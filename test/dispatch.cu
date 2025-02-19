//5.96 ms
#include <cuda_runtime.h>
#include <iostream>

#define CAPACITY 400
#define NUM_TOKEN 4096
#define K 8
#define HIDDEN_SIZE 7168

// Device function to get dispatched input
__device__ void get_dispatched_input(
    const int4* input, //[num_token, hidden_size]
    const int* t_eid,  // [tokenid][k] = expertid
    const int* t_eoffset,  // [tokenid][k] = the ith element of the expert
    int4* dispatched_input, // [e, c, hidden_size]
    const int num_token,
    const int k, // Number of selected experts
    const int hidden_size,
    const int capacity) {  

    int THREAD_TOKEN_NUM = num_token / gridDim.x;
    int start_token = THREAD_TOKEN_NUM * blockIdx.x;
    const int hidden_stride = hidden_size / 8;

    // Use dynamic shared memory (Up to 164 KB on A100)
    extern __shared__ int shared_buffer[];  // at most k = 10; otherwise we cannot hold it
    int* eid_cache = shared_buffer;
    int* eoffset_cache = &shared_buffer[THREAD_TOKEN_NUM * K];

    // Cooperative loading of index data
    if (threadIdx.x < THREAD_TOKEN_NUM * k) {
        eid_cache[threadIdx.x] = t_eid[threadIdx.x + start_token * k];
        eoffset_cache[threadIdx.x] = t_eoffset[threadIdx.x + start_token * k];
    }
    __syncthreads();

    #pragma unroll
    for (int tid = 0; tid < THREAD_TOKEN_NUM; tid += 1) {
        if (tid + start_token >= num_token) return;

        #pragma unroll
        for (int j = 0; j < k; ++j) {
            int eid = eid_cache[tid * k + j];
            int eoffset = eoffset_cache[tid * k + j];
            if (eoffset >= capacity) continue;
            #pragma unroll
            for (int s = threadIdx.x; s < hidden_size; s += blockDim.x) {
                dispatched_input[(eid * capacity + eoffset) * hidden_stride + s] = 
                    input[(start_token + tid) * hidden_stride + s];
            }
        }
    }
}



// Kernel wrapper to launch get_dispatched_input
__global__ void dispatch_kernel(
    const int4* input,
    const int* t_eid,
    const int* t_eoffset,
    int4* dispatched_input) {
    get_dispatched_input(input, t_eid, t_eoffset, dispatched_input, NUM_TOKEN, K, HIDDEN_SIZE, CAPACITY);
}

// Test function
void test_dispatch() {
    // Allocate host memory
    int4 *h_input = new int4[NUM_TOKEN * (HIDDEN_SIZE / 8)];
    int *h_t_eid = new int[NUM_TOKEN * K];
    int *h_t_eoffset = new int[NUM_TOKEN * K];
    int4 *h_dispatched_input = new int4[CAPACITY * K * (HIDDEN_SIZE / 8)];

    // Initialize with random values
    for (int i = 0; i < NUM_TOKEN * (HIDDEN_SIZE / 8); i++) {
        h_input[i] = make_int4(rand() % 100, rand() % 100, rand() % 100, rand() % 100);
    }
    for (int i = 0; i < NUM_TOKEN * K; i++) {
        h_t_eid[i] = rand() % K;
        h_t_eoffset[i] = rand() % CAPACITY;
    }

    // Allocate device memory
    int4 *d_input, *d_dispatched_input;
    int *d_t_eid, *d_t_eoffset;
    cudaMalloc(&d_input, NUM_TOKEN * (HIDDEN_SIZE / 8) * sizeof(int4));
    cudaMalloc(&d_t_eid, NUM_TOKEN * K * sizeof(int));
    cudaMalloc(&d_t_eoffset, NUM_TOKEN * K * sizeof(int));
    cudaMalloc(&d_dispatched_input, CAPACITY * K * (HIDDEN_SIZE / 8) * sizeof(int4));

    // Copy data to device
    cudaMemcpy(d_input, h_input, NUM_TOKEN * (HIDDEN_SIZE / 8) * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_eid, h_t_eid, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t_eoffset, h_t_eoffset, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice);

    // Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Set kernel attribute to use 164 KB shared memory on A100
    cudaFuncSetAttribute(dispatch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

    // Launch the kernel and measure execution time
    dim3 gridDim(128); // Adjust based on GPU capability
    dim3 blockDim(256);
    cudaEventRecord(start);
    dispatch_kernel<<<gridDim, blockDim, 163840>>>(d_input, d_t_eid, d_t_eoffset, d_dispatched_input);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_dispatched_input, d_dispatched_input, CAPACITY * K * (HIDDEN_SIZE / 8) * sizeof(int4), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_input;
    delete[] h_t_eid;
    delete[] h_t_eoffset;
    delete[] h_dispatched_input;
    cudaFree(d_input);
    cudaFree(d_t_eid);
    cudaFree(d_t_eoffset);
    cudaFree(d_dispatched_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    test_dispatch();
    return 0;
}
