#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <iostream>
#include <cuda_runtime.h>

#define N 128  // Vector length
#define ITERATIONS 1000  // Repeat the scan multiple times for stable timing

int main() {
    // Initialize host vector with values [1, 2, 3, ..., N]
    thrust::host_vector<int> h_vector(N);
    for (int i = 0; i < N; i++) {
        h_vector[i] = i + 1;
    }

    // Copy data to GPU
    thrust::device_vector<int> d_vector = h_vector;

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Run cumsum multiple times for performance stability
    for (int i = 0; i < ITERATIONS; i++) {
        thrust::inclusive_scan(d_vector.begin(), d_vector.end(), d_vector.begin());
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / ITERATIONS;

    std::cout << "Total Time for " << ITERATIONS << " iterations: " << milliseconds << " ms" << std::endl;
    std::cout << "Average Time per inclusive_scan: " << avg_time << " ms" << std::endl;

    // Copy result back to host
    thrust::host_vector<int> h_result = d_vector;

    // Print final cumsum result (optional)
    std::cout << "Cumulative Sum (Thrust GPU):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
