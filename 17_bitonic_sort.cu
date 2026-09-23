// bitonic sort, the classic GPU sorting network
// build: nvcc 17_bitonic_sort.cu -o bitonic
//
// fixed comparison pattern, no data-dependent branching, perfect for GPUs.
// two nested loops on the host pick the (j, k) stage pair, each launch is
// one compare-exchange pass over the whole array:
//   k = the size of the sorted subsequence being built
//   j = the distance compared within it
// direction is decided by bit k of the index.
// n must be a power of two (pad if not).
//
// not the fastest sort on GPUs in practice (thrust uses radix) but it's
// the one interviewers know, and it shows off "control logic lives on
// the host, data-parallel work on the device" nicely.

#include <cstdio>
#include <cstdlib>
#include <cstdlib>

#define N (1 << 16)

__global__ void bitonicStep(int *data, int j, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;
    if (ixj > i) {
        // ascending subsequence if bit k of i is 0, descending otherwise
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) { int t = data[i]; data[i] = data[ixj]; data[ixj] = t; }
        } else {
            if (data[i] < data[ixj]) { int t = data[i]; data[i] = data[ixj]; data[ixj] = t; }
        }
    }
}

int main() {
    int *h = (int *)malloc(N * sizeof(int));
    long long sumBefore = 0;
    srand(3);
    for (int i = 0; i < N; i++) { h[i] = rand() % 100000; sumBefore += h[i]; }

    int *d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h, N * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = N / threads;
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            bitonicStep<<<blocks, threads>>>(d, j, k);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);

    // sorted AND same multiset (sum as a cheap proxy + full order check)
    long long sumAfter = 0;
    bool sorted = true;
    for (int i = 0; i < N; i++) {
        sumAfter += h[i];
        if (i && h[i - 1] > h[i]) { sorted = false; break; }
    }
    printf("bitonic sort: %s (sum %lld vs %lld)\n",
           sorted && sumBefore == sumAfter ? "sorted ok" : "FAILED",
           sumBefore, sumAfter);

    cudaFree(d);
    free(h);
    return 0;
}
