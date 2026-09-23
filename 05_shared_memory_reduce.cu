// 共享内存入门：block 内归约求和
// 编译: nvcc 05_shared_memory_reduce.cu -o reduce
//
// shared memory 是同一个 block 内所有线程共享的高速片上内存，
// 比访问全局显存快得多。归约（reduction）是它的经典用法：
//   每一步把相邻元素两两相加，活动线程数每步减半。
// CPU 端最后把每个 block 的部分和再加起来。

#include <cstdio>
#include <cstdlib>

#define N (1 << 20)   // 1048576 个元素

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            printf("CUDA 错误: %s (文件 %s, 行 %d)\n",                 \
                   cudaGetErrorString(err), __FILE__, __LINE__);       \
            exit(1);                                                   \
        }                                                              \
    } while (0)

#define BLOCK_SIZE 256

__global__ void blockReduceSum(const float *in, float *blockSums, int n) {
    __shared__ float sdata[BLOCK_SIZE];   // 每个 block 一份共享内存

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 把自己负责的数据先读进共享内存（越界填 0，不影响求和）
    sdata[threadIdx.x] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();   // 等所有线程都把数据放好

    // 树形归约：stride 从 128, 64, 32 ... 减半到 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();   // 每轮都要同步
    }

    // 每个 block 的 0 号线程把结果写出去
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;   // 每个都是 1，和应等于 N

    float *d_in, *d_sums;
    CHECK(cudaMalloc(&d_in, bytes));
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK(cudaMalloc(&d_sums, numBlocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    blockReduceSum<<<numBlocks, BLOCK_SIZE>>>(d_in, d_sums, N);
    CHECK(cudaGetLastError());

    float *h_sums = (float *)malloc(numBlocks * sizeof(float));
    CHECK(cudaMemcpy(h_sums, d_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 上把各 block 的部分和加总
    double total = 0;
    for (int i = 0; i < numBlocks; i++) total += h_sums[i];
    printf("GPU 求和结果: %.0f，期望值: %d，%s\n",
           total, N, (int)total == N ? "正确!" : "有误!");

    cudaFree(d_in); cudaFree(d_sums);
    free(h_in); free(h_sums);
    return 0;
}
