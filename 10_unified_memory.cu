// unified memory: cudaMallocManaged
// build: nvcc 10_unified_memory.cu
//
// everything before this needed the malloc + 2x memcpy + free dance.
// with managed memory one pointer works on both sides, the driver
// migrates pages on demand.
//
// the catch: page migration isn't free. for performance code, explicit
// management still wins. great for prototypes though.

#include <cstdio>
#include <cmath>

__global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;

    // one pointer, usable from cpu and gpu
    float *x, *y;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    saxpy<<<(n + 255) / 256, 256>>>(3.0f, x, y, n);
    cudaDeviceSynchronize();  // must sync before the cpu reads results

    // cpu reads gpu output directly, no explicit copy anywhere
    double err = 0;
    for (int i = 0; i < n; i++) err += fabs(y[i] - 5.0f);
    printf("saxpy mean error %e\n", err / n);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
