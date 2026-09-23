// the "hello world" of real cuda work: vector add, c[i] = a[i] + b[i]
// build: nvcc 03_vector_add.cu -o vector_add
//
// this walks through the full pipeline that every cuda program follows:
//   1. allocate + init on the host
//   2. cudaMalloc on the device
//   3. cudaMemcpy host -> device
//   4. launch the kernel
//   5. cudaMemcpy device -> host
//   6. cudaFree
// step 3 and 5 are the ones people forget. no copy, no data.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 1024 * 1024

// cuda calls return cudaSuccess on happy days. this macro bails out
// with the file/line otherwise, saved me a few times already
#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s (%s, line %d)\n",             \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                             \
        }                                                        \
    } while (0)

// one thread handles exactly one element
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // global thread id: which block am i in, times block size, plus my id in it
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {  // we launch a few extra threads, they do nothing
        c[i] = a[i] + b[i];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);  // h_ = host
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;  // d_ = device
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 256 threads per block is a safe default. round the block count UP
    // so we cover all N elements
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // verify on the cpu
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) { ok = false; break; }
    }
    printf("%s (N=%d)\n", ok ? "vector add passed!" : "WRONG RESULT", N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
