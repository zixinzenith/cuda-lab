// matrix multiply, tiled with shared memory. the fix for 07.
// build: nvcc 08_matmul_tiled.cu -o matmul_tiled
//
// idea: naive does a full row of A + a full column of B from global memory
// per output element, tons of redundant reads. instead split A and B into
// TILE x TILE chunks, load one chunk of each into shared memory, let every
// thread in the block reuse them.
//
// __syncthreads() appears twice and both are needed:
//   once after loading the tile, once at the end of the loop
//   (otherwise a fast thread would overwrite the tile while slow threads
//   are still reading it). forgetting the second one = wrong results
//   only sometimes. good luck debugging that.

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
        // cooperative load: the block's 16x16 threads each fetch one element
        int acol = t * TILE + threadIdx.x;  // tile of A we need
        int brow = t * TILE + threadIdx.y;  // tile of B we need
        As[threadIdx.y][threadIdx.x] = A[row * n + acol];
        Bs[threadIdx.y][threadIdx.x] = B[brow * n + col];
        __syncthreads();

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

    // first launch includes module load / clock ramp stuff, run twice
    matmulTiled<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    matmulTiled<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("tiled matmul %dx%d: %.3f ms, C[0][0]=%.2f\n", N, N, ms, h_c[0]);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
