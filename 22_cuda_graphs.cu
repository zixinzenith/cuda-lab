// cuda graphs: record a kernel sequence once, replay many times
// build: nvcc 22_cuda_graphs.cu -o graphs
//
// every <<<>>> launch costs a few microseconds of CPU time. if you run
// the same short kernels thousands of times per frame (inference loops,
// simulations), launch overhead adds up. a graph captures the whole
// sequence once and replays it with a single call.
//
// capture is easy: do the launches inside BeginCapture/EndCapture on a
// non-default stream, then instantiate + launch the graph.
// gotcha: nothing runs during capture, results only appear on launch.

#include <cstdio>

#define N (1 << 20)

__global__ void scaleBy(float *d, float f, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] *= f;
}

__global__ void addOne(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += 1.f;
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h[i] = 1.0f;

    float *d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    // capture: x2 -> +1 -> x2 == 4x + 2
    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    scaleBy<<<(N + 255) / 256, 256, 0, s>>>(d, 2.0f, N);
    addOne<<<(N + 255) / 256, 256, 0, s>>>(d, N);
    scaleBy<<<(N + 255) / 256, 256, 0, s>>>(d, 2.0f, N);
    cudaStreamEndCapture(s, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // time graph replay vs three individual launches
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int reps = 200;

    cudaGraphLaunch(exec, s);   // warm up
    cudaDeviceSynchronize();

    cudaEventRecord(t0, s);
    for (int i = 0; i < reps; i++) cudaGraphLaunch(exec, s);
    cudaEventRecord(t1, s);
    cudaEventSynchronize(t1);
    float msGraph; cudaEventElapsedTime(&msGraph, t0, t1);

    cudaEventRecord(t0, s);
    for (int i = 0; i < reps; i++) {
        scaleBy<<<(N + 255) / 256, 256, 0, s>>>(d, 2.0f, N);
        addOne<<<(N + 255) / 256, 256, 0, s>>>(d, N);
        scaleBy<<<(N + 255) / 256, 256, 0, s>>>(d, 2.0f, N);
    }
    cudaEventRecord(t1, s);
    cudaEventSynchronize(t1);
    float msSolo; cudaEventElapsedTime(&msSolo, t0, t1);

    // note: the solo loop runs AFTER the graph loop, so d holds
    // (3 reps + 200 reps + warmup) applications of the same transform.
    // only the correctness of a fresh run is checked below.
    printf("graph: %.3f ms for %d replays (%.1f us each)\n", msGraph, reps, msGraph * 1000 / reps);
    printf("3 launches: %.3f ms for %d rounds (%.1f us each)\n", msSolo, reps, msSolo * 1000 / reps);

    // fresh correctness check
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    cudaGraphLaunch(exec, s);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
    printf("check h[0] = %.1f (want 6.0: 1*2+1 then *2)\n", h[0]);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(d);
    free(h);
    return 0;
}
