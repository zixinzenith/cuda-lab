// 二维线程网格：用 2D block 处理矩阵
// 编译: nvcc 04_grid_2d.cu -o grid_2d
//
// 处理图像、矩阵时常用二维索引：
//   row    = blockIdx.y * blockDim.y + threadIdx.y
//   col    = blockIdx.x * blockDim.x + threadIdx.x
// 本例给矩阵每个元素乘 2，并演示二维索引的计算方式。

#include <cstdio>
#include <cmath>

#define ROWS 64
#define COLS 64

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            printf("CUDA 错误: %s (文件 %s, 行 %d)\n",                 \
                   cudaGetErrorString(err), __FILE__, __LINE__);       \
            exit(1);                                                   \
        }                                                              \
    } while (0)

__global__ void scaleMatrix(float *m, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        // 一维存储的矩阵，第 row 行第 col 列的地址是 row * cols + col
        m[row * cols + col] *= 2.0f;
    }
}

int main() {
    size_t bytes = ROWS * COLS * sizeof(float);
    float *h_m = (float *)malloc(bytes);
    for (int i = 0; i < ROWS * COLS; i++) h_m[i] = 1.0f;

    float *d_m;
    CHECK(cudaMalloc(&d_m, bytes));
    CHECK(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));

    // 16x16 的二维 block（常见写法），网格大小也按二维算
    dim3 block(16, 16);
    dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);
    scaleMatrix<<<grid, block>>>(d_m, ROWS, COLS);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < ROWS * COLS; i++)
        if (fabs(h_m[i] - 2.0f) > 1e-5) { ok = false; break; }
    printf("%s\n", ok ? "二维矩阵乘 2 结果正确!" : "结果有误!");

    cudaFree(d_m);
    free(h_m);
    return 0;
}
