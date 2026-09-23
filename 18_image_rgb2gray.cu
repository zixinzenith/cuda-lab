// small image pipeline: rgb -> grayscale -> sobel edges, writes a PGM
// build: nvcc 18_image_rgb2gray.cu -o image
//
// two things to practice here:
// 1. per-pixel kernels with boundary handling (sobel clamps at borders)
// 2. luma weights 0.299/0.587/0.114 (BT.601) -- don't just average rgb,
//    the result looks wrong, greens dominate human perception
//
// output goes to sobel.pgm, view it with any image viewer:
//   python3 -c "from PIL import Image; Image.open('sobel.pgm').show()"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define W 1024
#define H 1024

__global__ void rgb2gray(const unsigned char *rgb, float *gray, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    gray[i] = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
}

__global__ void sobel(const float *gray, unsigned char *edges, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // clamp at the border instead of skipping it, keeps the image size
    int xm = max(x - 1, 0), xp = min(x + 1, w - 1);
    int ym = max(y - 1, 0), yp = min(y + 1, h - 1);

    float tl = gray[ym * w + xm], t = gray[ym * w + x], tr = gray[ym * w + xp];
    float l  = gray[y * w + xm],                    r  = gray[y * w + xp];
    float bl = gray[yp * w + xm], b = gray[yp * w + x], br = gray[yp * w + xp];

    float gx = (tr + 2.f * r + br) - (tl + 2.f * l + bl);
    float gy = (bl + 2.f * b + br) - (tl + 2.f * t + tr);
    float mag = sqrtf(gx * gx + gy * gy);

    edges[y * w + x] = (unsigned char)min(mag, 255.f);
}

int main() {
    int n = W * H;
    unsigned char *h_rgb = (unsigned char *)malloc(3 * n);
    unsigned char *h_edges = (unsigned char *)malloc(n);
    float *h_gray = (float *)malloc(n * sizeof(float));

    // synthetic input: a bright square on dark bg, sobel should outline it
    for (int i = 0; i < n; i++) {
        int x = i % W, y = i / W;
        bool square = (x > 200 && x < 500 && y > 200 && y < 400);
        unsigned char v = square ? 220 : 40;
        // slight noise so edges aren't perfectly clean
        v += rand() % 8;
        h_rgb[3 * i] = v; h_rgb[3 * i + 1] = v; h_rgb[3 * i + 2] = v;
    }

    unsigned char *d_rgb;
    float *d_gray;
    unsigned char *d_edges;
    cudaMalloc(&d_rgb, 3 * n);
    cudaMalloc(&d_gray, n * sizeof(float));
    cudaMalloc(&d_edges, n);
    cudaMemcpy(d_rgb, h_rgb, 3 * n, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    rgb2gray<<<grid, block>>>(d_rgb, d_gray, W, H);
    sobel<<<grid, block>>>(d_gray, d_edges, W, H);
    cudaDeviceSynchronize();

    cudaMemcpy(h_edges, d_edges, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gray, d_gray, n * sizeof(float), cudaMemcpyDeviceToHost);

    // verify gray on a handful of pixels
    double err = 0;
    for (int i = 0; i < n; i++) {
        float want = 0.299f * h_rgb[3 * i] + 0.587f * h_rgb[3 * i + 1] + 0.114f * h_rgb[3 * i + 2];
        err += fabs(h_gray[i] - want);
    }
    printf("gray mean error %.4f\n", err / n);

    FILE *f = fopen("sobel.pgm", "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", W, H);
        fwrite(h_edges, 1, n, f);
        fclose(f);
        printf("wrote sobel.pgm\n");
    }

    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_edges);
    free(h_rgb); free(h_edges); free(h_gray);
    return 0;
}
