// 2D thread grid, for when the data is a matrix/image instead of a flat array
// build: nvcc 04_grid_2d.cu -o grid_2d
//
// same idea as 1D but with a y component:
//   row = blockIdx.y * blockDim.y + threadIdx.y
//   col = blockIdx.x * blockDim.x + threadIdx.x
// this one multiplies every matrix element by 2, mostly to practice the indexing

#include <cstdio>
#include <cmath>

#define ROWS 64
#define COLS 64

#define CHECK(call)                                              \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            printf("CUDA error: %s (%s, line %d)\n",             \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1);                                             \
        }                                                        \
    } while (0)

__global__ void scaleMatrix(float *m, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        // the matrix is stored flat, element (row, col) lives at row*cols+col
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

    // 16x16 blocks are the usual choice for 2D problems
    dim3 block(16, 16);
    dim3 grid((COLS + block.x - 1) / block.x, (ROWS + block.y - 1) / block.y);
    scaleMatrix<<<grid, block>>>(d_m, ROWS, COLS);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < ROWS * COLS; i++)
        if (fabs(h_m[i] - 2.0f) > 1e-5) { ok = false; break; }
    printf("%s\n", ok ? "2D scale passed!" : "WRONG RESULT");

    cudaFree(d_m);
    free(h_m);
    return 0;
}
