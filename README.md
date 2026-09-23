# cuda-lab

CUDA learning notes in code form. Numbered from easy to less easy,
each file builds and runs on its own, concepts explained in comments.

Tested with CUDA 12.6 on an RTX 3060 Laptop.

## contents

| file | what it covers |
|---|---|
| `01_hello_cuda.cu` | first kernel: `__global__`, launch config, threadIdx/blockIdx |
| `02_device_info.cu` | querying GPU properties (SMs, memory, compute capability) |
| `03_vector_add.cu` | the full pipeline: cudaMalloc -> memcpy -> kernel -> back -> free, plus error checking |
| `04_grid_2d.cu` | 2D grids with dim3, row-major indexing |
| `05_shared_memory_reduce.cu` | `__shared__`, `__syncthreads()`, tree reduction |
| `06_cuda_events_timing.cu` | cudaEvent timing, block size sweep |
| `07_matmul_naive.cu` | naive GEMM, one thread per output, verified vs CPU |
| `08_matmul_tiled.cu` | GEMM with shared memory tiling, much faster than 07 |
| `09_transpose_coalescing.cu` | transposed access patterns, memory coalescing cost |
| `10_unified_memory.cu` | cudaMallocManaged, when it's worth it |
| `11_streams_overlap.cu` | streams + async copies + pinned memory pipeline |
| `12_histogram_atomics.cu` | atomicAdd, shared memory histogram to cut atomic traffic |
| `13_cublas_sgemm.cu` | cuBLAS, column-major gotcha, vs my hand-written kernel |
| `14_transpose_shared.cu` | transpose staged through shared memory, bank conflict padding (+1 column) |
| `15_prefix_sum_scan.cu` | Hillis-Steele inclusive scan, double buffering |
| `16_warp_primitives.cu` | `__shfl_xor_sync` butterfly reduction, `__ballot_sync` + popc |
| `17_bitonic_sort.cu` | bitonic sorting network, host-driven stages |
| `18_image_rgb2gray.cu` | grayscale + sobel edge detection, writes a PGM you can open |
| `19_conv2d_3x3.cu` | 3x3 convolution with the filter in `__constant__` memory |
| `20_matmul_regtile.cu` | GEMM #3: 4x4 register tiling per thread, faster than 08 |
| `21_thrust_sandbox.cu` | thrust reduce/transform/sort vs std::sort, mixed device/host iterator trap |
| `22_cuda_graphs.cu` | stream capture + graph replay, launch overhead vs plain launches |

## build & run

```bash
make        # build everything
make run    # build and run everything in order
make clean
```

or one at a time:

```bash
nvcc 03_vector_add.cu -o vector_add
./vector_add
```

## suggested order

1. 01 + 02 first: the CPU-launches-GPU-work model, your hardware's numbers
2. 03 carefully -- it's the skeleton of every CUDA program
3. 04 for 2D indexing (images/matrices)
4. 05/06: shared memory and how to actually time GPU code
5. 07-09: interview territory. hand-written GEMM, tiling, coalescing
6. 14-20: the optimization toolbox -- shared-memory transpose, scan,
   warp shuffles, sorting, image ops, constant memory, register tiling
7. 10-13 + 21-22: engineering flavor: unified memory, streams, atomics,
   cuBLAS, thrust, cuda graphs
