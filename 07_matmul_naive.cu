// matrix multiply, naive version. C = A * B
// build: nvcc 07_matmul_naive.cu -o matmul_naive
//
// every thread computes ONE element of C:
//   C[i][j] = sum over k of A[i][k] * B[k][j]
// the problem: threads in a warp read B down a column, so consecutive
// threads touch addresses n*sizeof(float) apart -> zero coalescing.
// 08 fixes this with shared memory tiling.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define M 512
#define N 512
#define K 512

__global__ void matmulNaive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.f;
    for (int k = 0; k < n; k++) {
        // row major: A(row,k) = A[row*n+k], B(k,col) = B[k*n+col]
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

// reference on the cpu to check against
void cpuMatmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.f;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

int main() {
    int bytes = M * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes), *h_ref = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < M * N; i++) {
        h_a[i] = rand() % 10 / 10.f;
        h_b[i] = rand() % 10 / 10.f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmulNaive<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("kernel died: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cpuMatmul(h_a, h_b, h_ref, N);
    float maxErr = 0;
    for (int i = 0; i < M * N; i++)
        maxErr = fmaxf(maxErr, fabsf(h_c[i] - h_ref[i]));
    printf("naive matmul, max error %f %s\n", maxErr, maxErr < 1e-3 ? "(ok)" : "(WRONG!)");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return 0;
}
