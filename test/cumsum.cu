
//correct! 0.005 ms
#include <cuda_runtime.h>
#include <iostream>

#define NUM_EXPERTS 256
#define NUM_TOKEN 32768
#define BLOCK_X 256
#define GRID_X 128
#define ITERATIONS 100 // Run multiple iterations for stable timing
__global__ void cumsum(int* token_place, int* cum_histogram, const int num_experts, const int num_token) {
    __shared__ int smem[NUM_EXPERTS];  // Increased shared memory size to avoid overflow

    int global_threadID = blockIdx.x * blockDim.x + threadIdx.x;

    // Load histogram into shared memory
    if (threadIdx.x < NUM_EXPERTS) {
        smem[threadIdx.x] = cum_histogram[threadIdx.x];
    }
    __syncthreads();

    // Ensure thread does not exceed number of tokens
    if (global_threadID >= NUM_TOKEN) return;

    int offset = 0;

    // Find the appropriate expert bin for the token
    for (int i = 0; i < NUM_EXPERTS; i++) {
        if (global_threadID < smem[i]) {
            offset = (i == 0) ? global_threadID : (global_threadID - smem[i - 1]);
            //offset = global_threadID;
            break;
        }
    }
    __syncthreads();
    // Store computed offset for each token
    token_place[global_threadID] = offset;
}


void print_array(const char* label, int* array, int size) {
    std::cout << label << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

//check correctness
// int main() {
    
//     int cum_histogram[NUM_EXPERTS] = {12,25,55,64};
//     int *d_cum_histogram, *d_token_place;
//     int *h_token_place = new int[NUM_TOKEN];

//     // Allocate GPU memory
//     cudaMalloc(&d_cum_histogram, NUM_EXPERTS * sizeof(int));
//     cudaMalloc(&d_token_place, NUM_TOKEN * sizeof(int));

//     // Copy data to GPU
//     cudaMemcpy(d_cum_histogram, cum_histogram, NUM_EXPERTS * sizeof(int), cudaMemcpyHostToDevice);

//     // Launch kernel
//     dim3 gridDim(GRID_X);
//     dim3 blockDim(BLOCK_X);
//     cumsum<<<gridDim, blockDim>>>(d_token_place, d_cum_histogram, NUM_EXPERTS, NUM_TOKEN);
//     cudaDeviceSynchronize();

//     // Copy back results
//     cudaMemcpy(h_token_place, d_token_place, NUM_TOKEN * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print result
//     print_array("Token Place", h_token_place, NUM_TOKEN);

//     // Free memory
//     delete[] h_token_place;
//     cudaFree(d_cum_histogram);
//     cudaFree(d_token_place);

//     return 0;
// }

//check performance
int main() {
    int cum_histogram[NUM_EXPERTS] = {500, 1000, 1500, 2000, 2500, 3000, 3500, NUM_TOKEN};
    int *d_cum_histogram, *d_token_place;
    int *h_token_place = new int[NUM_TOKEN];

    // Allocate GPU memory
    cudaMalloc(&d_cum_histogram, NUM_EXPERTS * sizeof(int));
    cudaMalloc(&d_token_place, NUM_TOKEN * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_cum_histogram, cum_histogram, NUM_EXPERTS * sizeof(int), cudaMemcpyHostToDevice);

    // Setup CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Run kernel multiple times for stable measurements
    for (int i = 0; i < ITERATIONS; i++) {
        cumsum<<<GRID_X, BLOCK_X>>>(d_token_place, d_cum_histogram, NUM_EXPERTS, NUM_TOKEN);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / ITERATIONS;

    std::cout << "Average Execution Time per Kernel Launch: " << avg_time << " ms" << std::endl;

    // Copy back results
    cudaMemcpy(h_token_place, d_token_place, NUM_TOKEN * sizeof(int), cudaMemcpyDeviceToHost);


    // Cleanup
    delete[] h_token_place;
    cudaFree(d_cum_histogram);
    cudaFree(d_token_place);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
