// matrix transpose staged through shared memory
// build: nvcc 14_transpose_shared.cu -o transpose_shared
//
// in 09 one side of the transpose was always strided. fix: load a 32x32
// tile into shared memory with coalesced reads, transpose the tile IN
// shared memory, write it out with coalesced writes. both sides fast now.
//
// the +1 in tile[32][33] is not a typo: it pads the row so column access
// during the transpose doesn't hit shared memory bank conflicts.

#include <cstdio>
#include <cstdlib>

#define N 4096
#define TILE 32

#define cudaCheck(e) do { \
    if ((e) != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

// naive version from 09, kept here for an honest side-by-side timing
__global__ void transposeNaive(const float *A, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) C[c * n + r] = A[r * n + c];
}

__global__ void transposeShared(const float *A, float *C, int n) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = A[y * n + x];
    __syncthreads();

    // this block's tile moves to the transposed position in the output,
    // so swap the roles of x and y between grid and tile coordinates
    int x2 = blockIdx.y * TILE + threadIdx.x;
    int y2 = blockIdx.x * TILE + threadIdx.y;
    C[y2 * n + x2] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_a[i] = i % 997;

    float *d_a, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));
    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    transposeShared<<<grid, block>>>(d_a, d_c, N);
    cudaCheck(cudaDeviceSynchronize());  // warm up

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeNaive<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float msNaive; cudaEventElapsedTime(&msNaive, t0, t1);

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeShared<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float msShared; cudaEventElapsedTime(&msShared, t0, t1);

    // check correctness against the naive result we just computed
    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < N && !bad; r++)
        for (int c = 0; c < N; c++)
            if (h_c[r * N + c] != h_a[c * N + r]) { bad = 1; break; }
    printf("shared transpose %s\n", bad ? "WRONG" : "correct");

    printf("naive  (one strided side): %.3f ms\n", msNaive / 10);
    printf("shared (both coalesced)  : %.3f ms\n", msShared / 10);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_c);
    free(h_a); free(h_c);
    return 0;
}
