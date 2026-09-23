// 原子操作：atomicAdd 统计直方图
//
// 直方图 = 数每个像素值出现的次数。多个线程同时给同一个桶 +1 时
// 普通的读-加-写会互相覆盖（race condition），要用 atomicAdd。
//
// 原子操作慢的原因：同一个地址的原子操作只能串行化。
// 小技巧：先在 shared memory 里做局部直方图（冲突范围缩小到一个 block），
// 最后每个 block 再往全局内存加一次，原子操作次数大幅减少。

#include <cstdio>
#include <cstdlib>

#define IMG_W 4096
#define IMG_H 4096
#define BINS 256

// 版本1：直接对全局内存做原子加
__global__ void histGlobal(const unsigned char *img, int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[img[i]], 1);
}

// 版本2：shared memory 局部直方图 + 汇总
__global__ void histShared(const unsigned char *img, int *hist, int n) {
    __shared__ int local[BINS];
    // 每个 block 先把自己的局部直方图清零
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) local[i] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        atomicAdd(&local[img[i]], 1);   // 冲突只发生在 block 内

    __syncthreads();
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&hist[b], local[b]);
}

int main() {
    int nPixels = IMG_W * IMG_H;
    unsigned char *h_img = (unsigned char *)malloc(nPixels);
    srand(1234);
    for (int i = 0; i < nPixels; i++) h_img[i] = rand() % BINS;

    // CPU 参考答案
    int ref[BINS] = {0};
    for (int i = 0; i < nPixels; i++) ref[h_img[i]]++;

    unsigned char *d_img;
    int *d_hist;
    cudaMalloc(&d_img, nPixels);
    cudaMalloc(&d_hist, BINS * sizeof(int));
    cudaMemcpy(d_img, h_img, nPixels, cudaMemcpyHostToDevice);

    int hist[BINS];

    // 两个版本都跑一遍
    const char *names[2] = {"global 原子", "shared 优化"};

    for (int v = 0; v < 2; v++) {
        cudaMemset(d_hist, 0, BINS * sizeof(int));
        if (v == 0)
            histGlobal<<<(nPixels + 255) / 256, 256>>>(d_img, d_hist, nPixels);
        else
            histShared<<<nPixels / 4 / 256, 256>>>(d_img, d_hist, nPixels);
        cudaDeviceSynchronize();
        cudaMemcpy(hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);

        int diff = 0;
        for (int b = 0; b < BINS; b++) diff += abs(hist[b] - ref[b]);
        printf("%s: 与 CPU 结果差异 %d %s\n", names[v], diff, diff == 0 ? "(对)" : "(错!)");
    }

    cudaFree(d_img); cudaFree(d_hist);
    free(h_img);
    return 0;
}
