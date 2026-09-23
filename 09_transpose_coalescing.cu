// matrix transpose two ways, to show coalescing
//
// transpose itself is trivial. the point is memory coalescing:
// you want the 32 threads of a warp to touch 32 consecutive floats,
// so the hardware issues ONE memory transaction instead of 32.
//
//   readCoalescedWriteBad: reads A row-wise (good), writes C column-wise (bad)
//   readBadWriteCoalesced: the mirror image
// time them, the gap is the price of strided access.
//
// a real transpose wants BOTH sides coalesced: stage the tile through
// shared memory and transpose it there. that version lives in
// 14_transpose_shared.cu now

#include <cstdio>
#include <cstdlib>

#define N 4096  // big enough that the difference actually shows

#define cudaCheck(e) do { \
    if ((e) != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

// one thread moves one element
__global__ void transposeReadOk(const float *A, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) C[c * n + r] = A[r * n + c];  // writes are strided
}

__global__ void transposeWriteOk(const float *A, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) C[r * n + c] = A[c * n + r];  // reads are strided
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_a[i] = i % 1000;

    float *d_a, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));
    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    dim3 block(32, 8);  // 32 in x so a warp reads one full row
    dim3 grid(N / 32, N / 8);

    // warm up first, first launch has extra overhead
    transposeReadOk<<<grid, block>>>(d_a, d_c, N);
    cudaCheck(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeReadOk<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms1; cudaEventElapsedTime(&ms1, t0, t1);

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeWriteOk<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms2; cudaEventElapsedTime(&ms2, t0, t1);

    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("check C[100][5]=%.0f, expected %.0f\n", h_c[100 * N + 5], h_a[5 * N + 100]);

    printf("coalesced read / strided write : %.3f ms\n", ms1 / 10);
    printf("strided read  / coalesced write: %.3f ms\n", ms2 / 10);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_c);
    free(h_a); free(h_c);
    return 0;
}
