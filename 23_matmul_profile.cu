// all three matmul versions in one binary, made for profiling with ncu
// build: nvcc 23_matmul_profile.cu -o matmul_profile
//
// why one binary: `ncu ./matmul_profile` then profiles every kernel in
// one session and the reports are directly comparable (same launch order
// every time). kernels get named so -k can filter:
//
//   ncu -k regex:naive  --set full -c 1 ./matmul_profile
//   ncu -k regex:tiled  --set full -c 1 ./matmul_profile
//   ncu -k regex:regtile --set full -c 1 ./matmul_profile
//
// or just run ./profile_matmul.sh which collects the interesting
// sections for all three into one csv + a readable summary.
//
// the question this file exists to answer: WHY does 08 beat 07, and 20
// beat 08? numbers live in PROFILING.md next to this file.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 512
#define TILE 16
#define MS 64
#define RT 4

// ---------- v1: naive, 1 output per thread, all global memory ----------
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    float sum = 0.f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// ---------- v2: 16x16 shared memory tiles ----------
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < n / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

// ---------- v3: 64x64 tiles, 4x4 register patch per thread ----------
__global__ void matmul_regtile(const float *A, const float *B, float *C, int n) {
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

    for (int t = 0; t < n / MS; t++) {
        for (int l = 0; l < 16; l++) {
            int idx = l * 256 + linear;
            int r = idx / MS, c = idx % MS;
            int gr = blockIdx.y * MS + r;
            int gc = t * MS + c;
            if (gr < n && gc < n) {
                As[r][c] = A[gr * n + gc];
                Bs[r][c] = B[(t * MS + r) * n + blockIdx.x * MS + c];
            }
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
            if (row0 + i < n && col0 + j < n)
                C[(row0 + i) * n + col0 + j] = acc[i][j];
}

int main(int argc, char **argv) {
    // optional size on the command line, must stay a multiple of MS=64
    // for the regtile grid math: ./matmul_profile 1024
    int n = (argc > 1) ? atoi(argv[1]) : N;
    n = (n / MS) * MS;  // snap down to a multiple of 64

    size_t bytes = (size_t)n * n * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < n * n; i++) {
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

    // 2 launches each: ncu grabs the first (-c 1), the timer uses the
    // second one so JIT/ramp doesn't pollute the wall clock numbers
    dim3 b16(TILE, TILE), g16(n / TILE, n / TILE);
    dim3 b256(16, 16), g64(n / MS, n / MS);

    matmul_naive<<<g16, b16>>>(d_a, d_b, d_c, n);
    matmul_tiled<<<g16, b16>>>(d_a, d_b, d_c, n);
    matmul_regtile<<<g64, b256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    double gflop = 2.0 * n * n * n / 1e9;
    struct Named { const char *name; int which; } list[] = {
        {"naive", 0}, {"tiled", 1}, {"regtile", 2}
    };
    for (auto &r : list) {
        cudaEventRecord(t0);
        for (int rep = 0; rep < 5; rep++) {
            if (r.which == 0) matmul_naive<<<g16, b16>>>(d_a, d_b, d_c, n);
            if (r.which == 1) matmul_tiled<<<g16, b16>>>(d_a, d_b, d_c, n);
            if (r.which == 2) matmul_regtile<<<g64, b256>>>(d_a, d_b, d_c, n);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        printf("%-8s %8.3f ms  %8.1f GFLOP/s\n", r.name, ms / 5, gflop / (ms / 5 * 1e-3));
    }

    // sanity: all three must agree
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < N * N; i += 997) s += h_c[i];
    printf("checksum %.2f\n", s);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
