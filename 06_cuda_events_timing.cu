// timing kernels with cuda events, comparing different block sizes
// build: nvcc 06_cuda_events_timing.cu -o timing
//
// don't time GPU work with the cpu clock -- launches are async, you'd only
// measure the launch. the right way is cudaEvent: two timestamps recorded
// ON the GPU, elapsed time between them.

#include <cstdio>
#include <cstdlib>

#define N (1 << 24)  // ~16M elements

#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s (%s, line %d)\n",             \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                             \
        }                                                        \
    } while (0)

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // sweep block sizes, watch how it plateaus. 64 is clearly too small
    // (not enough warps to hide latency) after that it barely matters
    // for a bandwidth bound kernel like this
    int sizes[] = {64, 128, 256, 512, 1024};
    for (int s = 0; s < 5; s++) {
        int threadsPerBlock = sizes[s];
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        CHECK(cudaEventRecord(start));
        vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("block size %4d : kernel time %.3f ms\n", threadsPerBlock, ms);
    }

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("spot check c[123456] = %.1f (want 3.0)\n", h_c[123456]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
