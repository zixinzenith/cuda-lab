// thrust: the STL-flavored layer over CUDA
// build: nvcc 21_thrust_sandbox.cu -o thrust_demo
//
// when you just need sort/reduce/transform done, thrust is the answer --
// it picks good kernels (radix sort etc.) and nobody reviews you for
// reinventing them. good to know both worlds: hand-written for learning
// and hot loops, thrust for everything else.
//
// includes a quick timing against std::sort on the host to feel the gap.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/count.h>

int main() {
    const int n = 1 << 22;
    srand(8);

    thrust::host_vector<int> h(n);
    for (int i = 0; i < n; i++) h[i] = rand() % 100000;
    thrust::device_vector<int> d = h;   // copies host -> device

    // reduce
    long long sum = thrust::reduce(d.begin(), d.end(), (long long)0, thrust::plus<long long>());
    printf("thrust reduce: %lld\n", sum);

    // transform: scale by 2, then count how many exceed 100000
    thrust::transform(d.begin(), d.end(), d.begin(), 2 * thrust::placeholders::_1);
    int over = thrust::count_if(d.begin(), d.end(), thrust::placeholders::_1 > 100000);
    printf("count > 100000 after x2: %d\n", over);

    // sort, timed against std::sort on the cpu for scale
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double gpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::sort(h.begin(), h.end());
    auto t3 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    bool same = true;
    {   // learned the hard way: thrust::equal(d.begin(), d.end(), h.begin())
        // compiles but reads the host pointer from the GPU -> illegal address.
        // mixed device/host iterators need an explicit copy back first
        thrust::host_vector<int> h2 = d;
        // d holds the DOUBLED values, h was sorted above, so the sorted
        // result should be exactly 2 * h element-wise
        for (int i = 0; i < n && same; i++)
            if (h2[i] != 2 * h[i]) same = false;
    }
    printf("sort %d ints: gpu %.1f ms vs cpu %.1f ms, %s\n",
           n, gpuMs, cpuMs, same ? "results match" : "RESULTS DIFFER");

    return 0;
}
