// my first cuda program. launch a kernel that prints from the GPU
// build: nvcc 01_hello_cuda.cu -o hello
//
// things to remember:
// - __global__ marks a function as a kernel (runs on GPU, called from CPU)
// - <<<blocks, threads>>> is the launch config, i.e. how many threads to start
// - threadIdx / blockIdx let each thread know who it is

#include <cstdio>

__global__ void helloFromGPU() {
    // threadIdx.x = id inside the block, blockIdx.x = which block
    printf("Hello World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // 2 blocks x 4 threads = 8 threads total
    helloFromGPU<<<2, 4>>>();

    // kernel launch is async, have to wait for it or the printf output
    // gets cut off when main returns
    cudaDeviceSynchronize();

    printf("Done! (that line was from the CPU)\n");
    return 0;
}
