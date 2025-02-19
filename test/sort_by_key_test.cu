// pass: 0.157696 ms
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define N 32768  // Test with 32,768 elements
#define NUM_COLS 256  // Example: 128 unique column values
#define NUM_ROWS 4096  // Example: 256 unique row values

// Custom comparator: First by column, then by row
struct compare_col_row {
    __host__ __device__ bool operator()(const thrust::tuple<int, int, float>& a, 
                                        const thrust::tuple<int, int, float>& b) {
        int col_a = thrust::get<1>(a), col_b = thrust::get<1>(b);
        int row_a = thrust::get<0>(a), row_b = thrust::get<0>(b);

        if (col_a != col_b) return col_a < col_b;  // Sort by column first
        return row_a < row_b;  // If column is the same, sort by row
    }
};

int main() {
    // Allocate host memory
    std::vector<int> h_rows(N);
    std::vector<int> h_cols(N);
    std::vector<float> h_vals(N);

    // Randomly generate (row, col, value) pairs
    for (int i = 0; i < N; i++) {
        h_rows[i] = rand() % NUM_ROWS;  // Random row index
        h_cols[i] = rand() % NUM_COLS;  // Random column index
        h_vals[i] = static_cast<float>(rand()) / RAND_MAX;  // Random float value
    }

    // Allocate device memory
    thrust::device_vector<int> d_rows = h_rows;
    thrust::device_vector<int> d_cols = h_cols;
    thrust::device_vector<float> d_vals = h_vals;

    // Create tuple of (row, col, value)
    thrust::device_vector<thrust::tuple<int, int, float>> d_tuples(N);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_rows.begin(), d_cols.begin(), d_vals.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_rows.end(), d_cols.end(), d_vals.end())),
        d_tuples.begin(),
        thrust::identity<thrust::tuple<int, int, float>>()
    );

    // **Timing setup**
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // **Start timing**
    cudaEventRecord(start);

    // Sort by (col, row)
    thrust::sort(d_tuples.begin(), d_tuples.end(), compare_col_row());

    // **Stop timing**
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // **Measure execution time**
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Sorting Time (32,768 elements): " << milliseconds << " ms" << std::endl;

    // Copy sorted results back to host
    std::vector<thrust::tuple<int, int, float>> h_tuples(N);
    thrust::copy(d_tuples.begin(), d_tuples.end(), h_tuples.begin());

    // Print a few sorted results to verify correctness
    std::cout << "First 10 sorted (row, col, value) pairs:" << std::endl;
    for (int i = 0; i < 10; i++) {
        int row = thrust::get<0>(h_tuples[i]);
        int col = thrust::get<1>(h_tuples[i]);
        float val = thrust::get<2>(h_tuples[i]);
        std::cout << "(" << row << ", " << col << ", " << val << ")" << std::endl;
    }

    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
