// 查询 GPU 设备信息
// 编译: nvcc 02_device_info.cu -o device_info
//
// 写 CUDA 程序前先弄清楚自己的 GPU 有多少核心、多少显存，
// 这些参数决定了后面如何选择 block / thread 数量。

#include <cstdio>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("检测到 %d 个 CUDA 设备\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("===== 设备 %d =====\n", i);
        printf("名字:               %s\n", prop.name);
        printf("计算能力:           %d.%d\n", prop.major, prop.minor);
        printf("SM 数量:            %d\n", prop.multiProcessorCount);
        printf("显存大小:           %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("每个 block 最大线程数: %d\n", prop.maxThreadsPerBlock);
        printf("每个 SM 最大线程数:  %d\n", prop.maxThreadsPerMultiProcessor);
        printf("shared memory/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("\n");
    }
    return 0;
}
