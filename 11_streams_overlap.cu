// cuda streams: overlap copies with compute
// build: nvcc 11_streams_overlap.cu -o streams
//
// by default everything queues in one "default stream", so while data
// is copying, the SMs sit idle. split work into chunks and alternate
// them across two streams:
//   stream0 copies chunk 2 while stream1 computes chunk 1
// two requirements:
//   1. split the data (ping-pong buffers)
//   2. pinned memory via cudaMallocHost -- regular malloc'd memory
//      can't do async copies at all

#include <cstdio>

#define TOTAL (1 << 22)
#define CHUNK (TOTAL / 4)  // 4 chunks through the pipeline

__global__ void scaleKernel(float *d, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= factor;
}

int main() {
    size_t bytes = TOTAL * sizeof(float);
    size_t chunkBytes = CHUNK * sizeof(float);

    // pinned memory, required for async copies
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

    // pipeline: chunk i does copy -> kernel -> copy back, all on s[i%2]
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
    printf("2-stream pipeline over %d elements: %.3f ms total\n", TOTAL, ms);
    printf("check h[CHUNK*3] = %.1f (want 2.0)\n", h_data[CHUNK * 3]);

    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    return 0;
}
