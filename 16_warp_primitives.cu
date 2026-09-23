// warp-level primitives: butterfly reduction with __shfl_xor_sync
// build: nvcc 16_warp_primitives.cu -o warp_ops
//
// pre-Volta everyone hand-rolled reductions with shared memory + syncthreads.
// now a warp can exchange registers directly:
//   __shfl_xor_sync(mask, val, lane) : lane i gets val from lane i^offset
// xoring with 16, 8, 4, 2, 1 folds all 32 lanes onto lane 0 in 5 steps,
// no shared memory, no syncs. this is the fastest way to reduce within
// a warp and shows up in every modern reduction implementation.
//
// also demos __ballot_sync (which lanes have a true predicate, as a bitmask).

#include <cstdio>
#include <cstdlib>
#include <cstdlib>

#define N (1 << 20)
#define BLOCK 256

__global__ void sumWithShuffle(const float *in, float *out, int n) {
    float s = 0.f;
    // grid-stride: each thread sums a chunk in registers first
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        s += in[i];

    // warp reduction: 5 shuffle steps, all 32 lanes stay active
    for (int offset = 16; offset > 0; offset >>= 1)
        s += __shfl_xor_sync(0xffffffff, s, offset);

    int warp = threadIdx.x / 32;
    __shared__ float warpSums[BLOCK / 32];
    if ((threadIdx.x & 31) == 0) warpSums[warp] = s;
    __syncthreads();

    // first warp reduces the per-warp sums the same way
    if (warp == 0) {
        s = (threadIdx.x < BLOCK / 32) ? warpSums[threadIdx.x] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, offset);
        if (threadIdx.x == 0)
            atomicAdd(out, s);  // one atomic per block, negligible
    }
}

__global__ void ballotDemo(const int *flags, int *count, int n) {
    // grid-stride so the same launch config covers the whole array
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int base = 0; base < n; base += stride) {
        int i = base + i0;
        bool mine = (i < n) && flags[i];
        unsigned mask = __ballot_sync(0xffffffff, mine);  // bit per lane
        // count set bits in the mask, once per warp
        if ((threadIdx.x & 31) == 0)
            atomicAdd(count, __popc(mask));
    }
}

int main() {
    // part 1: shuffle sum
    size_t bytes = N * sizeof(float);
    float *h = (float *)malloc(bytes);
    double total = 0;  // double! float accumulation of 1M values drifts
    srand(11);
    for (int i = 0; i < N; i++) { h[i] = rand() % 100 / 100.f; total += h[i]; }

    float *d, *d_out;
    cudaMalloc(&d, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    sumWithShuffle<<<N / 4 / BLOCK, BLOCK>>>(d, d_out, N);
    cudaDeviceSynchronize();

    float got;
    cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("shuffle sum: %.2f, cpu: %.2f, %s\n", got, total,
           fabs(got - total) < 1.f ? "(ok)" : "(WRONG)");

    // part 2: ballot
    int *h_flags = (int *)malloc(N * sizeof(int));
    int nSet = 0;
    for (int i = 0; i < N; i++) { h_flags[i] = rand() % 3 == 0; nSet += h_flags[i]; }

    int *d_flags, *d_count;
    cudaMalloc(&d_flags, N * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemcpy(d_flags, h_flags, N * sizeof(int), cudaMemcpyHostToDevice);

    ballotDemo<<<N / 4 / BLOCK, BLOCK>>>(d_flags, d_count, N);
    cudaDeviceSynchronize();

    int gotCount;
    cudaMemcpy(&gotCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("ballot count: %d, cpu: %d, %s\n", gotCount, nSet,
           gotCount == nSet ? "(ok)" : "(WRONG)");

    cudaFree(d); cudaFree(d_out); cudaFree(d_flags); cudaFree(d_count);
    free(h); free(h_flags);
    return 0;
}
