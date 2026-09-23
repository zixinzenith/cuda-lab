// atomics: histogram with atomicAdd
// build: nvcc 12_histogram_atomics.cu -o hist
//
// histogram = count how often each value appears. many threads want to
// increment the same bucket; a plain read-add-write loses updates
// (race condition), so we need atomicAdd.
//
// atomics are slow because same-address operations serialize.
// trick: build a per-block histogram in shared memory first (conflicts
// only within a block), then each block merges into global memory once.
// cuts the number of global atomics by orders of magnitude.

#include <cstdio>
#include <cstdlib>

#define IMG_W 4096
#define IMG_H 4096
#define BINS 256

// v1: atomic straight into global memory
__global__ void histGlobal(const unsigned char *img, int *hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[img[i]], 1);
}

// v2: shared memory histogram per block, then merge
__global__ void histShared(const unsigned char *img, int *hist, int n) {
    __shared__ int local[BINS];
    // zero out this block's local histogram
    for (int i = threadIdx.x; i < BINS; i += blockDim.x) local[i] = 0;
    __syncthreads();

    // grid-stride loop so fewer blocks can still cover the whole image
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        atomicAdd(&local[img[i]], 1);  // contention limited to the block

    __syncthreads();
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&hist[b], local[b]);
}

int main() {
    int nPixels = IMG_W * IMG_H;
    unsigned char *h_img = (unsigned char *)malloc(nPixels);
    srand(1234);
    for (int i = 0; i < nPixels; i++) h_img[i] = rand() % BINS;

    // cpu reference
    int ref[BINS] = {0};
    for (int i = 0; i < nPixels; i++) ref[h_img[i]]++;

    unsigned char *d_img;
    int *d_hist;
    cudaMalloc(&d_img, nPixels);
    cudaMalloc(&d_hist, BINS * sizeof(int));
    cudaMemcpy(d_img, h_img, nPixels, cudaMemcpyHostToDevice);

    int hist[BINS];
    const char *names[2] = {"global atomics", "shared mem + merge"};

    for (int v = 0; v < 2; v++) {
        cudaMemset(d_hist, 0, BINS * sizeof(int));
        if (v == 0)
            histGlobal<<<(nPixels + 255) / 256, 256>>>(d_img, d_hist, nPixels);
        else
            histShared<<<nPixels / 4 / 256, 256>>>(d_img, d_hist, nPixels);
        cudaDeviceSynchronize();
        cudaMemcpy(hist, d_hist, BINS * sizeof(int), cudaMemcpyDeviceToHost);

        int diff = 0;
        for (int b = 0; b < BINS; b++) diff += abs(hist[b] - ref[b]);
        printf("%s: diff vs cpu = %d %s\n", names[v], diff, diff == 0 ? "(ok)" : "(WRONG)");
    }

    cudaFree(d_img); cudaFree(d_hist);
    free(h_img);
    return 0;
}
