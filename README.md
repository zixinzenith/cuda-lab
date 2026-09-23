# cuda-lab

CUDA 入门学习代码，按编号从易到难，每个文件独立可编译运行，注释里写了核心概念。

环境：CUDA 12.6 + NVIDIA RTX 3060 Laptop

## 目录

| 文件 | 内容 | 关键概念 |
|---|---|---|
| `01_hello_cuda.cu` | 第一个 CUDA 程序 | `__global__`、kernel 启动 `<<<blocks, threads>>>`、`threadIdx`/`blockIdx` |
| `02_device_info.cu` | 查询 GPU 设备信息 | `cudaGetDeviceProperties`、SM、显存、计算能力 |
| `03_vector_add.cu` | 经典向量加法 | 完整流程：cudaMalloc → cudaMemcpy → kernel → 拷回 → cudaFree，错误检查宏 |
| `04_grid_2d.cu` | 二维线程网格处理矩阵 | `dim3`、二维 `blockIdx`/`threadIdx`、行主序地址计算 |
| `05_shared_memory_reduce.cu` | 共享内存归约求和 | `__shared__`、`__syncthreads()`、树形归约 |
| `06_cuda_events_timing.cu` | CUDA Event 计时 | `cudaEventRecord`/`cudaEventElapsedTime`，对比不同 block 大小性能 |
| `07_matmul_naive.cu` | 矩阵乘法朴素版 | 一个线程算一个元素，和 CPU 结果对比验证 |
| `08_matmul_tiled.cu` | 矩阵乘法共享内存分块版 | tiling、`__syncthreads()` 两次同步、比 naive 快数倍 |
| `09_transpose_coalescing.cu` | 矩阵转置两种写法对比 | 合并访存（coalescing）、warp 内存事务 |
| `10_unified_memory.cu` | 统一内存 | `cudaMallocManaged`，省掉显式拷贝的取舍 |
| `11_streams_overlap.cu` | 双流流水线 | `cudaStream_t`、`cudaMemcpyAsync`、锁页内存 `cudaMallocHost` |
| `12_histogram_atomics.cu` | 直方图统计 | `atomicAdd` 竞争问题、shared memory 局部直方图优化 |
| `13_cublas_sgemm.cu` | cuBLAS 库调用 | 列主序坑、句柄管理、和手写 kernel 的性能差距 |

## 编译运行

```bash
make          # 编译全部
make run      # 编译并依次运行
make clean
```

单个文件也可以直接编译：

```bash
nvcc 03_vector_add.cu -o vector_add
./vector_add
```

## 学习路线建议

1. 先跑 `01`、`02`，理解"CPU 发号施令、GPU 干活"的模型和自己的 GPU 参数
2. 精读 `03`，这是所有 CUDA 程序的骨架（分配、拷贝、计算、拷回、释放）
3. `04` 学会用二维索引处理矩阵/图像
4. `05`、`06` 开始接触性能优化：共享内存和正确的 GPU 计时方法
5. `07`~`09` 是面试重点：手写 GEMM、tiling 优化、合并访存
6. `10`~`13` 偏工程实践：统一内存、stream 重叠、原子操作、cuBLAS
