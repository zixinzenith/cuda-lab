// cublas: the library you'd actually use at work instead of hand-rolled GEMM
// build: nvcc 13_cublas_sgemm.cu -lcublas -o cublas
//
// two gotchas that cost me some time:
// 1. cublas is column-major (fortran style), C code is row-major.
//    trick: C = A*B in row-major equals C^T = B^T * A^T in column-major,
//    so pass the row-major buffers as-is with OP_N and the result
//    buffer already holds row-major C. no transposing of data needed.
// 2. create the cublasHandle_t once, reuse it everywhere.
//
// also compares timing against my tiled kernel in 08. humbling.

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define N 512

int main() {
    int bytes = N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    srand(99);
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

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warm up (cuBLAS picks kernels on first call, don't time that)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_b, N, d_a, N, &beta, d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_b, N, d_a, N, &beta, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("cuBLAS sgemm %dx%d: %.3f ms avg, C[0]=%.2f\n", N, N, ms / 10, h_c[0]);

    cublasDestroy(handle);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
