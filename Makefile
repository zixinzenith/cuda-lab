# builds every .cu file in one go
NVCC = nvcc
TARGETS = hello device_info vector_add grid_2d reduce timing \
          matmul_naive matmul_tiled transpose unified streams hist cublas \
          transpose_shared scan warp_ops bitonic image conv2d \
          matmul_regtile thrust_demo graphs

all: $(TARGETS)

hello: 01_hello_cuda.cu
	$(NVCC) $< -o $@

device_info: 02_device_info.cu
	$(NVCC) $< -o $@

vector_add: 03_vector_add.cu
	$(NVCC) $< -o $@

grid_2d: 04_grid_2d.cu
	$(NVCC) $< -o $@

reduce: 05_shared_memory_reduce.cu
	$(NVCC) $< -o $@

timing: 06_cuda_events_timing.cu
	$(NVCC) $< -o $@

matmul_naive: 07_matmul_naive.cu
	$(NVCC) $< -o $@

matmul_tiled: 08_matmul_tiled.cu
	$(NVCC) $< -o $@

transpose: 09_transpose_coalescing.cu
	$(NVCC) $< -o $@

unified: 10_unified_memory.cu
	$(NVCC) $< -o $@

streams: 11_streams_overlap.cu
	$(NVCC) $< -o $@

hist: 12_histogram_atomics.cu
	$(NVCC) $< -o $@

cublas: 13_cublas_sgemm.cu
	$(NVCC) $< -lcublas -o $@

transpose_shared: 14_transpose_shared.cu
	$(NVCC) $< -o $@

scan: 15_prefix_sum_scan.cu
	$(NVCC) $< -o $@

warp_ops: 16_warp_primitives.cu
	$(NVCC) $< -o $@

bitonic: 17_bitonic_sort.cu
	$(NVCC) $< -o $@

image: 18_image_rgb2gray.cu
	$(NVCC) $< -o $@

conv2d: 19_conv2d_3x3.cu
	$(NVCC) $< -o $@

matmul_regtile: 20_matmul_regtile.cu
	$(NVCC) $< -o $@

thrust_demo: 21_thrust_sandbox.cu
	$(NVCC) $< -o $@

graphs: 22_cuda_graphs.cu
	$(NVCC) $< -o $@

run: all
	./hello
	./device_info
	./vector_add
	./grid_2d
	./reduce
	./timing
	./matmul_naive
	./matmul_tiled
	./transpose
	./unified
	./streams
	./hist
	./cublas
	./transpose_shared
	./scan
	./warp_ops
	./bitonic
	./image
	./conv2d
	./matmul_regtile
	./thrust_demo
	./graphs

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
