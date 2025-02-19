
//pass: 0.28432 ms
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>

#define N 32768   // Test with large dataset
#define NUM_BINS 256  // Unique values range from 1 to 40

int main() {
    // Allocate host and device vectors
    thrust::host_vector<int> h_array(N);
    thrust::device_vector<int> d_array(N);
    thrust::device_vector<int> d_unique_keys(NUM_BINS);
    thrust::device_vector<int> d_histogram(NUM_BINS, 0);

    // Initialize host array with random values from 1 to 40
    for (int i = 0; i < N; i++) {
        h_array[i] = (rand() % NUM_BINS) + 1;
    }

    // Copy data to device
    d_array = h_array;

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start timer

    // Step 1: Sort the array (Thrust optimized sorting on GPU)
    thrust::sort(d_array.begin(), d_array.end());

    // Step 2: Reduce-by-key to count occurrences
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> end_positions;
    end_positions = thrust::reduce_by_key(
        d_array.begin(), d_array.end(),                         // Keys: sorted numbers
        thrust::make_constant_iterator(1),                      // Values: All 1s for counting
        d_unique_keys.begin(), d_histogram.begin()              // Output: Unique keys + their counts
    );

    cudaEventRecord(stop);  // Stop timer
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    int num_unique = end_positions.first - d_unique_keys.begin();  // Get actual unique count

    // Copy result back to host
    thrust::host_vector<int> h_unique_keys(d_unique_keys.begin(), d_unique_keys.begin() + num_unique);
    thrust::host_vector<int> h_histogram(d_histogram.begin(), d_histogram.begin() + num_unique);

    // Print histogram results
    std::cout << "Value Counts (Thrust GPU):" << std::endl;
    for (int i = 0; i < num_unique; i++) {
        std::cout << "Value " << h_unique_keys[i] << ": " << h_histogram[i] << std::endl;
    }

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
