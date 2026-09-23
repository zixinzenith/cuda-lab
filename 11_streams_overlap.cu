// CUDA Streams：让数据拷贝和计算重叠
//
// 默认所有操作都在一个"默认流"里排队，拷贝的时候 GPU 算力闲着。
// 开两个（或多个）stream 之后：
//   stream0 拷贝第 2 块数据的同时，stream1 可以算第 1 块
// 前提：数据要分块（ping-pong），显存用 cudaMallocHost 分配的
// 锁页内存（pinned memory），普通 malloc 的内存拷贝无法与 kernel 重叠。

#include <cstdio>

#define TOTAL (1 << 22)
#define CHUNK (TOTAL / 4)   // 数据切成 4 份流水处理

__global__ void scaleKernel(float *d, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= factor;
}

int main() {
    size_t bytes = TOTAL * sizeof(float);
    size_t chunkBytes = CHUNK * sizeof(float);

    // 锁页内存，异步拷贝的必要条件
    float *h_data;
    cudaMallocHost(&h_data, bytes);
    for (int i = 0; i < TOTAL; i++) h_data[i] = 1.0f;

    float *d_data;
    cudaMalloc(&d_data, bytes);

    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // 流水线：第 i 块在 s[i%2] 上 拷入 -> 计算 -> 拷出
    for (int i = 0; i < 4; i++) {
        cudaStream_t s = (i % 2 == 0) ? s0 : s1;
        float *chunkPtr = d_data + i * CHUNK;
        float *hostPtr = h_data + i * CHUNK;

        cudaMemcpyAsync(chunkPtr, hostPtr, chunkBytes, cudaMemcpyHostToDevice, s);
        scaleKernel<<<(CHUNK + 255) / 256, 256, 0, s>>>(chunkPtr, 2.0f, CHUNK);
        cudaMemcpyAsync(hostPtr, chunkPtr, chunkBytes, cudaMemcpyDeviceToHost, s);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("双流流水处理 %d 个元素, 总耗时 %.3f ms\n", TOTAL, ms);
    printf("检查 h[CHUNK*3] = %.1f (期望 2.0)\n", h_data[CHUNK * 3]);

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    return 0;
}
