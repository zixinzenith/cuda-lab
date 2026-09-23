// 统一内存（Unified Memory）：cudaMallocManaged
// 前面每个例子都要手写 cudaMalloc + 两次 cudaMemcpy，
// 用 managed 内存的话 CPU/GPU 直接共用一个指针，页错误时驱动自动搬数据。
//
// 代价：自动搬迁有开销，性能敏感的场合还是手动管理更好。
// 适合原型开发。这是个取舍问题，面试也常问。

#include <cstdio>
#include <cmath>

__global__ void saxpy(float a, float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int n = 1 << 20;

    // 一个指针，CPU 和 GPU 都能用
    float *x, *y;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    saxpy<<<(n + 255) / 256, 256>>>(3.0f, x, y, n);
    cudaDeviceSynchronize();   // 必须同步，否则 CPU 读的时候还没算完

    // CPU 直接检查 GPU 的结果，没有显式拷贝
    double err = 0;
    for (int i = 0; i < n; i++) err += fabs(y[i] - 5.0f);
    printf("saxpy 平均误差 %e\n", err / n);

    cudaFree(x);
    cudaFree(y);
    return 0;
}
