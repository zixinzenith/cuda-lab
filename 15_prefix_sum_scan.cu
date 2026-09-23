// parallel prefix sum (scan) with Hillis-Steele in shared memory
// build: nvcc 15_prefix_sum_scan.cu -o scan
//
// inclusive scan: out[i] = in[0] + ... + in[i].
// the parallel version does log2(n) steps, each thread grabs the element
// `offset` to its left and adds it. needs double buffering, otherwise
// threads read values that other threads already overwrote mid-step.
//
// this is the simple version: single block, n = blockDim.x.
// Hillis-Steele does O(n log n) total work; Blelloch's scan does O(n)
// work with two passes but is way fiddlier. for multi-block arrays you
// just chain block scans + offsets (that's what thrust/cub does).

#include <cstdio>
#include <cstdlib>

#define N 1024  // one block's worth on purpose, see note above

__global__ void scanHillisSteele(float *data, int n) {
    __shared__ float temp[2][N];
    int tid = threadIdx.x;
    int pin = 0, pout = 1;

    temp[pin][tid] = data[tid];
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        temp[pout][tid] = temp[pin][tid];   // carry my value over
        if (tid >= offset)
            temp[pout][tid] += temp[pin][tid - offset];
        __syncthreads();
        pout ^= 1;
        pin ^= 1;
    }

    data[tid] = temp[pin][tid];
}

int main() {
    float *h = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));
    srand(5);
    for (int i = 0; i < N; i++) h[i] = rand() % 10 / 10.f;

    // cpu reference
    ref[0] = h[0];
    for (int i = 1; i < N; i++) ref[i] = ref[i - 1] + h[i];

    float *d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice);

    scanHillisSteele<<<1, N>>>(d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxErr = 0;
    for (int i = 0; i < N; i++)
        maxErr = fmaxf(maxErr, fabsf(h[i] - ref[i]));
    printf("scan max error %f %s\n", maxErr, maxErr < 1e-3 ? "(ok)" : "(WRONG)");

    cudaFree(d);
    free(h); free(ref);
    return 0;
}
