# 一条命令编译所有 .cu 文件
NVCC = nvcc
TARGETS = hello device_info vector_add grid_2d reduce timing

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

run: all
	./hello
	./device_info
	./vector_add
	./grid_2d
	./reduce
	./timing

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
