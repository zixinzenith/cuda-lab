// anchors for the roofline model, measured on the actual hardware
// build: nvcc 24_roofline_anchor.cu -lcublas -o roofline
//
// before claiming any kernel is "memory bound" or "compute bound" you
// need to know where the ceiling is on YOUR card (not the spec sheet --
// laptop TGP configs vary a lot). this measures:
//   1. D2D copy bandwidth        -> raw DRAM ceiling (read+write)
//   2. saxpy streaming bandwidth -> realistic read/write mix ceiling
//   3. cublas sgemm peak         -> practical FP32 compute ceiling
// those two numbers are the roof; PROFILING.md plots the matmul kernels
// against them.

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CHECK(e) do { \
    if ((e) != cudaSuccess) { printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; } \
} while (0)

__global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ---- 1. device-to-device copy: 2 GB moved (1 read + 1 write) ----
    size_t bytes = 1ull << 30;
    float *x, *y;
    CHECK(cudaMalloc(&x, bytes));
    CHECK(cudaMalloc(&y, bytes));
    cudaMemcpy(x, x, 1, cudaMemcpyDeviceToDevice);  // warm up pages
    float bestCopy = 1e9f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        cudaMemcpy(y, x, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        if (ms < bestCopy) bestCopy = ms;
    }
    printf("D2D copy : %8.1f GB/s  (1 GB in %.2f ms)\n",
           bytes * 2 / (bestCopy * 1e-3) / 1e9, bestCopy);
    cudaFree(x); cudaFree(y);

    // ---- 2. saxpy: 3 arrays touched, 2 reads + 1 write per element ----
    const int n = 1 << 26;  // 64M floats = 256 MB per array
    float *a, *b;
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMalloc(&b, n * sizeof(float)));
    cudaMemset(a, 1, n * sizeof(float));
    cudaMemset(b, 2, n * sizeof(float));
    saxpy<<<(n + 255) / 256, 256>>>(2.0f, a, b, n);
    cudaDeviceSynchronize();
    float bestSaxpy = 1e9f;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(t0);
        saxpy<<<(n + 255) / 256, 256>>>(2.0f, a, b, n);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        if (ms < bestSaxpy) bestSaxpy = ms;
    }
    printf("saxpy    : %8.1f GB/s  (3 * %d MB in %.2f ms)\n",
           3.0 * n * sizeof(float) / (bestSaxpy * 1e-3) / 1e9,
           (int)(n * sizeof(float) >> 20), bestSaxpy);
    cudaFree(a); cudaFree(b);

    // ---- 3. cublas sgemm: the practical compute ceiling ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    const int m = 4096;
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, (size_t)m * m * sizeof(float)));
    CHECK(cudaMalloc(&d_b, (size_t)m * m * sizeof(float)));
    CHECK(cudaMalloc(&d_c, (size_t)m * m * sizeof(float)));
    float alpha = 1, beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m,
                &alpha, d_b, m, d_a, m, &beta, d_c, m);
    cudaDeviceSynchronize();
    float bestGemm = 1e9f;
    for (int rep = 0; rep < 5; rep++) {
        cudaEventRecord(t0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m,
                    &alpha, d_b, m, d_a, m, &beta, d_c, m);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        if (ms < bestGemm) bestGemm = ms;
    }
    printf("sgemm %d : %8.1f GFLOP/s (%.3f ms) <- compute roof\n",
           m, 2.0 * m * m * m / (bestGemm * 1e-3) / 1e9, bestGemm);

    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}
