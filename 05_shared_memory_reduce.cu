// shared memory + parallel reduction (block sum)
// build: nvcc 05_shared_memory_reduce.cu -o reduce
//
// shared memory is a small scratchpad on the SM, shared by all threads
// of one block. way faster than hitting global memory.
// the reduction pattern: pairwise adds, active threads halve each step.
// each block produces a partial sum, the cpu adds those up at the end.
//
// note the #define below needs the parentheses! I once wrote it as
// `#define N 1 << 20` and N * sizeof(float) became 1 << (20*4) -> segfault.
// classic macro trap.

#include <cstdio>
#include <cstdlib>

#define N (1 << 20)

#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s (%s, line %d)\n",             \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                             \
        }                                                        \
    } while (0)

#define BLOCK_SIZE 256

__global__ void blockReduceSum(const float *in, float *blockSums, int n) {
    __shared__ float sdata[BLOCK_SIZE];  // one copy per block

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // load my element into shared memory (out of range -> 0, harmless for sum)
    sdata[threadIdx.x] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();  // everyone must finish loading before we start

    // tree reduction: stride goes 128, 64, 32, ... down to 1
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();  // sync every round
    }

    // thread 0 of the block writes the result out
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;  // all ones, sum should be N

    float *d_in, *d_sums;
    CHECK(cudaMalloc(&d_in, bytes));
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK(cudaMalloc(&d_sums, numBlocks * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    blockReduceSum<<<numBlocks, BLOCK_SIZE>>>(d_in, d_sums, N);
    CHECK(cudaGetLastError());

    float *h_sums = (float *)malloc(numBlocks * sizeof(float));
    CHECK(cudaMemcpy(h_sums, d_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // finish the sum on the cpu
    double total = 0;
    for (int i = 0; i < numBlocks; i++) total += h_sums[i];
    printf("GPU sum: %.0f, expected: %d, %s\n",
           total, N, (int)total == N ? "passed!" : "WRONG!");

    cudaFree(d_in); cudaFree(d_sums);
    free(h_in); free(h_sums);
    return 0;
}
