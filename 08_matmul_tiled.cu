// 矩阵乘法优化版：共享内存分块（tiling）
// 对比 07 的 naive 版，能快好几倍
//
// 核心想法：naive 版每个元素要从全局显存读 A 的一行 + B 的一列，
// 大量重复读。改成把 A、B 各切成 TILE x TILE 的小块，
// 每次协作地把一块搬进 shared memory，block 内所有线程反复用。
//
// 记忆点：__syncthreads() 出现两次——搬完数据同步一次，
// 算完再同步一次（防止有的线程先进入下一轮把数据改了）。

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 512
#define TILE 16

__global__ void matmulTiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;

    int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // 合作搬运：本 block 的 16x16 个线程各搬一个元素
        int acol = t * TILE + threadIdx.x;              // A 中要的那一小块
        int brow = t * TILE + threadIdx.y;              // B 中要的那一小块
        As[threadIdx.y][threadIdx.x] = A[row * n + acol];
        Bs[threadIdx.y][threadIdx.x] = B[brow * n + col];
        __syncthreads();

        // 用 shared memory 里的小块算部分和
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < n && col < n) C[row * n + col] = sum;
}

int main() {
    int bytes = N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    srand(7);
    for (int i = 0; i < N * N; i++) {
        h_a[i] = rand() % 10 / 10.f;
        h_b[i] = rand() % 10 / 10.f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    cudaEventRecord(t0);
    matmulTiled<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("tiled 矩阵乘法 %dx%d, 耗时 %.3f ms, C[0][0]=%.2f\n",
           N, N, ms, h_c[0]);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
