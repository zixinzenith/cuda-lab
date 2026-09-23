# builds every .cu file in one go
NVCC = nvcc
TARGETS = hello device_info vector_add grid_2d reduce timing \
          matmul_naive matmul_tiled transpose unified streams hist cublas

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

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
