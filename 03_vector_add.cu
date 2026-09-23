// 经典入门例子：向量加法 C[i] = A[i] + B[i]
// 编译: nvcc 03_vector_add.cu -o vector_add
//
// 完整走一遍 CUDA 的标准流程：
//   1. CPU 上分配并初始化数据
//   2. cudaMalloc 在 GPU 上分配显存
//   3. cudaMemcpy 把数据从 CPU 拷到 GPU
//   4. 启动 kernel 计算
//   5. cudaMemcpy 把结果拷回 CPU
//   6. cudaFree 释放显存

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N 1024 * 1024   // 向量长度

// 检查 CUDA API 返回值的宏，出错了直接打印文件和行号
#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            printf("CUDA 错误: %s (文件 %s, 行 %d)\n",                 \
                   cudaGetErrorString(err), __FILE__, __LINE__);       \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// 一个线程只负责一个元素的加法
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // 全局线程编号：先算出自己在第几个 block，再乘上 block 的大小
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {          // 线程数可能比 n 多，越界的线程什么都不干
        c[i] = a[i] + b[i];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // 1. CPU 上分配内存
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }

    // 2. GPU 上分配显存（d_ 前缀表示 device）
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // 3. 把输入数据拷到 GPU
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 4. 启动 kernel：每个 block 256 个线程，block 数量由 N 算出
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;   // 向上取整
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());

    // 5. 把结果拷回 CPU
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) { ok = false; break; }
    }
    printf("%s （N=%d）\n", ok ? "向量加法结果正确!" : "结果有误!", N);

    // 6. 释放
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
