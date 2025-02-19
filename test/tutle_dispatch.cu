#include <cuda_runtime.h>
#include <iostream>

#define CAPACITY 140
#define NUM_TOKEN 4096
#define K 8
#define HIDDEN_SIZE 3584

// Device function to get dispatched input
__device__ void get_dispatched_input(
    const float* input, //[num_token, hidden_size]
    const int* t_eid,  // experts
    const int* t_eoffset,  // [tokenid][k] = the ith element of the expert
    float* dispatched_input, // [e, c, hidden_size]
    const int num_token,
    const int k, // Number of selected experts
    const int hidden_size,
    const int capacity) {  
        for (int tid = blockIdx.x; tid < num_token; tid += gridDim.x) {
          #pragma unroll
          for (int l = 0; l < k; l += 1) {
            int id = tid * k + l;
          if (t_eoffset[id] > capacity) continue;
              #pragma unroll
              for (int j = threadIdx.x; j < hidden_size; j += blockDim.x)
                  dispatched_input[(t_eid[id] + t_eoffset[id]) * hidden_size + j] = input[tid * hidden_size + j];
          }
        }
}

// Kernel wrapper to launch get_dispatched_input
__global__ void dispatch_kernel(
    const float* input,
    const int* t_eid,
    const int* t_eoffset,
    float* dispatched_input) {
    get_dispatched_input(input, t_eid, t_eoffset, dispatched_input, NUM_TOKEN, K, HIDDEN_SIZE, CAPACITY);
}

// Test function
void test_dispatch() {
    // Allocate host memory
    float *h_input_f = new float[NUM_TOKEN * HIDDEN_SIZE];
    int *h_t_eid = new int[NUM_TOKEN * K];
    int *h_t_eoffset = new int[NUM_TOKEN * K];
    float *h_dispatched_input = new float[CAPACITY * K * HIDDEN_SIZE];

    // Initialize with random values
    for (int i = 0; i < NUM_TOKEN * HIDDEN_SIZE; i++) {
        h_input_f[i] = rand();
    }
    for(int i=0; i<256; i++){
        for(int j=0; j<CAPACITY; j++){
            if (i * CAPACITY + j >= NUM_TOKEN * K) break;
            h_t_eid[i * CAPACITY + j] = i;
            h_t_eoffset[i * CAPACITY + j] = j;
        }
    }

    // Allocate device memory
    float *d_input_f, *d_dispatched_input_f;
    int *d_t_eid, *d_t_eoffset;
    cudaMalloc(&d_input_f, NUM_TOKEN * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_t_eid, NUM_TOKEN * K * sizeof(int));
    cudaMalloc(&d_t_eoffset, NUM_TOKEN * K * sizeof(int));
    cudaMalloc(&d_dispatched_input_f, CAPACITY * K * HIDDEN_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input_f, h_input_f, NUM_TOKEN * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_eid, h_t_eid, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_eoffset, h_t_eoffset, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice);

    // Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel and measure execution time
    dim3 gridDim(512); // Adjust based on GPU capability
    dim3 blockDim(1024);
    cudaEventRecord(start);
    dispatch_kernel<<<gridDim, blockDim>>>(d_input_f, d_t_eid, d_t_eoffset, d_dispatched_input_f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_dispatched_input, d_dispatched_input_f, CAPACITY * K * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_input_f;
    delete[] h_t_eid;
    delete[] h_t_eoffset;
    delete[] h_dispatched_input;
    cudaFree(d_input_f);
    cudaFree(d_t_eid);
    cudaFree(d_t_eoffset);
    cudaFree(d_dispatched_input_f);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_dispatch();
    return 0;
}
