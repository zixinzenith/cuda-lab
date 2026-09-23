// print out GPU device properties
// build: nvcc 02_device_info.cu -o device_info
//
// good to know your hardware before picking block sizes.
// run this once and keep the numbers in mind (SM count matters a lot
// when you start reasoning about occupancy)

#include <cstdio>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("found %d CUDA device(s)\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("===== device %d =====\n", i);
        printf("name:                %s\n", prop.name);
        printf("compute capability:  %d.%d\n", prop.major, prop.minor);
        printf("SM count:            %d\n", prop.multiProcessorCount);
        printf("global memory:       %.1f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("max threads/block:   %d\n", prop.maxThreadsPerBlock);
        printf("max threads/SM:      %d\n", prop.maxThreadsPerMultiProcessor);
        printf("shared mem/block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("\n");
    }
    return 0;
}
