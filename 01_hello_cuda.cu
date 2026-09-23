// 第一个 CUDA 程序：在 GPU 上打印 Hello World
// 编译: nvcc 01_hello_cuda.cu -o hello
//
// 核心概念：
//   __global__ : 标记一个函数是 kernel，由 GPU 执行，CPU 调用
//   <<<blocks, threads>>> : 启动配置，决定开多少个线程
//   threadIdx / blockIdx : 每个线程自己的编号，用来区分干不同的活

#include <cstdio>

// kernel 函数：每个 GPU 线程都会执行一遍这个函数
__global__ void helloFromGPU() {
    // threadIdx.x 是线程在 block 内的编号，blockIdx.x 是 block 的编号
    printf("Hello World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // 启动 2 个 block，每个 block 4 个线程，一共 8 个线程
    helloFromGPU<<<2, 4>>>();

    // 等待 GPU 执行完毕（kernel 启动是异步的，CPU 不会自动等它）
    cudaDeviceSynchronize();

    printf("Done! CPU 也在工作。\n");
    return 0;
}
