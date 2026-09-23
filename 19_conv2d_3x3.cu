// 3x3 convolution with the filter in __constant__ memory
// build: nvcc 19_conv2d_3x3.cu -o conv2d
//
// sharpen kernel, one thread per output pixel.
// the filter (9 floats) is identical for every thread and never changes,
// which is exactly what constant memory is for: cached and broadcast,
// so all threads in a warp reading the same address pay essentially nothing.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define W 2048
#define H 2048

__constant__ float filt[9];   // row-major 3x3

__global__ void conv3x3(const float *in, float *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) return;  // skip border

    float sum = 0.f;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            sum += filt[(ky + 1) * 3 + (kx + 1)] * in[(y + ky) * w + (x + kx)];
    out[y * w + x] = sum;
}

void cpuConv3x3(const float *in, const float *f, float *out, int w, int h) {
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            float s = 0.f;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    s += f[(ky + 1) * 3 + (kx + 1)] * in[(y + ky) * w + (x + kx)];
            out[y * w + x] = s;
        }
}

int main() {
    int n = W * H;
    size_t bytes = n * sizeof(float);
    float *h_in = (float *)malloc(bytes), *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    srand(21);
    for (int i = 0; i < n; i++) h_in[i] = rand() % 256 / 255.f;

    float sharpen[9] = { 0, -1,  0,
                        -1,  5, -1,
                         0, -1,  0 };
    cudaMemcpyToSymbol(filt, sharpen, sizeof(sharpen));

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    conv3x3<<<grid, block>>>(d_in, d_out, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cpuConv3x3(h_in, sharpen, h_ref, W, H);
    float maxErr = 0;
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            maxErr = fmaxf(maxErr, fabsf(h_out[i] - h_ref[i]));
        }
    printf("conv3x3 max error %f %s\n", maxErr, maxErr < 1e-3 ? "(ok)" : "(WRONG)");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
