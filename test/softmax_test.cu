//0.070656 ms
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define ROWS 4096
#define COLS 256
#define THREADS_PER_BLOCK 256

__global__ void rowwise_softmax(float* d_matrix, float* d_output, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= rows || col >= cols) return;

    __shared__ float row_max;
    __shared__ float sum_exp;

    // Initialize shared memory
    if (col == 0) {
        row_max = -1e9f;
        sum_exp = 0.0f;
    }
    __syncthreads();

    // Compute max value (use atomicMax for parallel safety)
    atomicMax((int*)&row_max, __float_as_int(d_matrix[row * cols + col]));
    __syncthreads();

    // Compute exponentials and sum them
    d_output[row * cols + col] = expf(d_matrix[row * cols + col] - row_max);
    atomicAdd(&sum_exp, d_output[row * cols + col]);
    __syncthreads();

    // Normalize
    d_output[row * cols + col] /= sum_exp;
}

int main() {
    float *d_matrix, *d_output;
    size_t matrix_size = ROWS * COLS * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_matrix, matrix_size);
    cudaMalloc(&d_output, matrix_size);

    // Initialize matrix with random values on the host
    float *h_matrix = new float[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) {
        h_matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy matrix to device
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start timer

    // Launch kernel
    rowwise_softmax<<<ROWS, 1>>>(d_matrix, d_output, ROWS, COLS);

    cudaEventRecord(stop);  // Stop timer
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaFree(d_matrix);
    cudaFree(d_output);
    delete[] h_matrix;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
