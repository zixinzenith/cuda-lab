// 矩阵转置：对比两种读写的访存效率
//
// 转置 C = A^T 本身逻辑很简单，重点在"合并访存"（coalescing）：
//   同一个 warp 的 32 个线程最好访问连续的 32 个 float，
//   这样硬件只发一次内存事务，否则拆成 32 次，带宽利用率暴跌。
//
// readCoalescedWriteBad: 读 A 是连续的（好），写 C 跨行（差）
// readBadWriteCoalesced: 反过来
// 实测两者的时间差就是合并访存的代价。面试可以聊这个。

#include <cstdio>
#include <cstdlib>

#define N 4096   // 4096x4096，够大才测得出差距

#define cudaCheck(e) do { \
    if ((e) != cudaSuccess) { printf("cuda error %s\n", cudaGetErrorString(e)); exit(1); } \
} while (0)

// 一个线程转置一个元素
__global__ void transposeReadOk(const float *A, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) C[c * n + r] = A[r * n + c];   // 写是跨行的
}

__global__ void transposeWriteOk(const float *A, float *C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) C[r * n + c] = A[c * n + r];   // 读是跨行的
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_a = (float *)malloc(bytes), *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_a[i] = i % 1000;

    float *d_a, *d_c;
    cudaCheck(cudaMalloc(&d_a, bytes));
    cudaCheck(cudaMalloc(&d_c, bytes));
    cudaCheck(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    dim3 block(32, 8);   // x 方向 32：保证 warp 内线程读同一行
    dim3 grid(N / 32, N / 8);

    // 暖机一下再测
    transposeReadOk<<<grid, block>>>(d_a, d_c, N);
    cudaCheck(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeReadOk<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms1; cudaEventElapsedTime(&ms1, t0, t1);

    cudaEventRecord(t0);
    for (int rep = 0; rep < 10; rep++) transposeWriteOk<<<grid, block>>>(d_a, d_c, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms2; cudaEventElapsedTime(&ms2, t0, t1);

    // 检查一下结果对不对
    cudaCheck(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("检查 C[100][5]=%.0f 应为 %.0f\n", h_c[100 * N + 5], h_a[5 * N + 100]);

    printf("读合并/写分散: %.3f ms\n", ms1 / 10);
    printf("读分散/写合并: %.3f ms\n", ms2 / 10);
    // 注：矩阵转置两边总有一个方向是分散的，实际优化要靠 shared memory 转置，
    // 在 shared 里做一次 32x32 转置让读写都合并，后面有空补上

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_c);
    free(h_a); free(h_c);
    return 0;
}
