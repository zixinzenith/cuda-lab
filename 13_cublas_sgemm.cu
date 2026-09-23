// 用 cuBLAS：实际工作中除非学习目的，一般不手写 GEMM
// cuBLAS 是 NVIDIA 官方的高度优化 BLAS 库，底层用的是 tensor core
//
// 坑点提醒：
// 1. cuBLAS 默认按列主序（Fortran 风格），和 C 的行主序相反。
//    技巧：C = A*B (行主序) 等价于 C^T = B^T * A^T (列主序)，
//    所以直接把行主序的 A、B 按原样传进去，用 cublasSgemm('N','N') 即可，
//    拿到的结果缓冲区就是行主序的 C。
// 2. 句柄 cublasHandle_t 创建一次，重复使用。
//
// 这里顺便和 08 的手写 tiled 版比个时间，感受一下差距。

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

    // 暖机
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
    printf("cuBLAS sgemm %dx%d: 平均 %.3f ms, C[0]=%.2f\n", N, N, ms / 10, h_c[0]);

    cublasDestroy(handle);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
