// gemm, third version: register tiling (thread coarsening)
// build: nvcc 20_matmul_regtile.cu -o matmul_regtile
//
// progression so far:
//   07 naive      : 1 output/thread, everything from global
//   08 tiled      : 1 output/thread, tiles staged in shared memory
//   20 this one   : each thread computes a 4x4 patch, keeping partial
//                   sums in registers across the k loop
// why: even with shared memory, doing 1 FMA per k means shared memory
// bandwidth is the new bottleneck (a 16x16 block issues 256 shared reads
// per k). loading a[4] and b[4] once per k gives 16 FMAs per 8 shared
// reads. same trick every real GEMM kernel uses, just with bigger tiles
// and double buffering on top.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 512
#define MS 64   // tile height/width. 64*64*4B * 2 tiles = 32KB shared
#define RT 4    // each thread computes an RT x RT patch

__global__ void matmulRegTile(const float *A, const float *B, float *C, int n) {
    __shared__ float As[MS][MS];
    __shared__ float Bs[MS][MS];

    int row0 = blockIdx.y * MS + threadIdx.y * RT;
    int col0 = blockIdx.x * MS + threadIdx.x * RT;

    float acc[RT][RT];
    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++)
            acc[i][j] = 0.f;

    int linear = threadIdx.y * blockDim.x + threadIdx.x;
    int numTiles = N / MS;

    for (int t = 0; t < numTiles; t++) {
        // cooperative load of two 64x64 tiles: 256 threads x 16 elements
        for (int l = 0; l < 16; l++) {
            int idx = l * 256 + linear;
            int r = idx / MS, c = idx % MS;
            int gr = blockIdx.y * MS + r;
            int gc = t * MS + c;
            As[r][c] = A[gr * n + gc];
            Bs[r][c] = B[(t * MS + r) * n + blockIdx.x * MS + c];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MS; k++) {
            float a[RT], b[RT];
            #pragma unroll
            for (int i = 0; i < RT; i++) {
                a[i] = As[threadIdx.y * RT + i][k];
                b[i] = Bs[k][threadIdx.x * RT + i];
            }
            #pragma unroll
            for (int i = 0; i < RT; i++)
                #pragma unroll
                for (int j = 0; j < RT; j++)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RT; i++)
        #pragma unroll
        for (int j = 0; j < RT; j++)
            C[(row0 + i) * n + col0 + j] = acc[i][j];
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    srand(42);
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
    dim3 block(16, 16);   // 256 threads, 64/16 = 4 -> RT x RT patch each
    dim3 grid(N / MS, N / MS);

    matmulRegTile<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();  // warm up

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++)
        matmulRegTile<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // spot check a few elements against the cpu
    double maxErr = 0;
    for (int i = 0; i < N; i += 37)
        for (int j = 0; j < N; j += 41) {
            float s = 0;
            for (int k = 0; k < N; k++) s += h_a[i * N + k] * h_b[k * N + j];
            maxErr = fmax(maxErr, (double)fabsf(h_c[i * N + j] - s));
        }
    printf("regtile matmul: %.3f ms avg, max spot err %f %s\n",
           ms / 10, maxErr, maxErr < 1e-2 ? "(ok)" : "(WRONG)");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
