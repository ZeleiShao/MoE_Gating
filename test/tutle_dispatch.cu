#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err); \
    } \
}

// Problem dimensions
#define CAPACITY 300
#define NUM_TOKEN 4096
#define K 8
#define HIDDEN_SIZE 7168
#define NUM_EXPERT 256
#define GRID_SIZE 512
#define BLOCK_SIZE 896

__global__ void dispatch_kernel(
    const int4* __restrict__ input,
    const int* __restrict__ expert_id, //[um_token, k]
    const int* __restrict__ index,  //[num_token, k]
    int4* dispatched_input, //[num_expert,capacity,hidden_stride]
    const int num_token,
    const int k,
    const int hidden_size,
    const int capacity
) {
    int hidden_stride = hidden_size / 8;
    for (int tid = blockIdx.x; tid < num_token; tid += GRID_SIZE) {
            #pragma unroll
            for (int l = 0; l < k; l++) {
                int id = tid * k + l;
                int idx = index[id];
                int eid = expert_id[id];
                if (idx >= capacity) continue;
                for (int j = threadIdx.x; j < hidden_stride; j += BLOCK_SIZE){
                  dispatched_input[(eid * capacity + idx) * hidden_stride + j] = 
                    input[tid * hidden_stride + j];
                }           
        }
    }
}


// __global__ void dispatch_kernel(
//     const int4* input,
//     const int* expert_id,
//     const int* index,
//     int4* dispatched_input,
//     int num_token,
//     int k,
//     int hidden_size,
//     int capacity
// ) {
//     const int hidden_stride = hidden_size / 8;
    
//     // Each thread handles a single token's K assignments
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_token) return;

//     // Process all K experts for this token
//     for (int l = 0; l < k; ++l) {
//         const int assignment_idx = tid * k + l;
//         const int expert = expert_id[assignment_idx];
//         const int idx = index[assignment_idx];

//         // Skip invalid indices
//         if (idx >= capacity) continue;

//         // Calculate output position
//         const int4* src = &input[tid * hidden_stride];
//         int4* dst = &dispatched_input[(expert * capacity + idx) * hidden_stride];

//         // Copy entire hidden_stride using all threads in the block
//         for (int j = threadIdx.x; j < hidden_stride; j += BLOCK_SIZE) {
//             dst[j] = src[j];
//         }
//     }
// }
void test_correctness() {
    // Initialize host data
    const int hidden_stride = HIDDEN_SIZE / 8;
    int4* h_input = new int4[NUM_TOKEN * hidden_stride];
    int* h_expert_id = new int[NUM_TOKEN * K];
    int* h_index = new int[NUM_TOKEN * K];
    int4* h_dispatched = new int4[NUM_EXPERT * CAPACITY * hidden_stride]{};
    int4* h_reference = new int4[NUM_EXPERT * CAPACITY * hidden_stride]{};

    // Generate test data
    std::default_random_engine rng(42);
    std::uniform_int_distribution<int> expert_dist(0, NUM_EXPERT-1);
    std::uniform_int_distribution<int> index_dist(0, CAPACITY);  // Some invalid indices

    // Initialize input with unique patterns
    for (int t = 0; t < NUM_TOKEN; t++) {
        for (int j = 0; j < hidden_stride; j++) {
            h_input[t * hidden_stride + j] = make_int4(t, j, t+j, t-j);
        }
    }

// Assuming these macros are defined:
// NUM_TOKEN, NUM_EXPERT, K, CAPACITY

// Allocate the arrays h_expert_id and h_index (of size NUM_TOKEN * K) before this block.
{
  // Create a vector for each expert listing all available indices.
  std::vector<std::vector<int>> available_indices(NUM_EXPERT);
  for (int e = 0; e < NUM_EXPERT; e++) {
      for (int idx = 0; idx < CAPACITY; idx++) {
          available_indices[e].push_back(idx);
      }
  }
  
  // For each token, select K distinct experts and assign them a random available index.
  std::default_random_engine rng(42);  // or use your existing rng
  for (int t = 0; t < NUM_TOKEN; t++) {
      // Create a candidate list of expert IDs [0, 1, 2, ..., NUM_EXPERT-1]
      std::vector<int> candidate_experts(NUM_EXPERT);
      for (int e = 0; e < NUM_EXPERT; e++) {
          candidate_experts[e] = e;
      }
      // Shuffle to get a random order.
      std::shuffle(candidate_experts.begin(), candidate_experts.end(), rng);
      
      // For this token, assign the first K experts from the shuffled list.
      for (int l = 0; l < K; l++) {
          int expert = candidate_experts[l];
          // Check that there is at least one available index for this expert.
          if (available_indices[expert].empty()) {
              std::cerr << "Not enough capacity for expert " << expert << std::endl;
              exit(1);
          }
          // Randomly select an available index from available_indices[expert]
          std::uniform_int_distribution<int> dist(0, available_indices[expert].size() - 1);
          int pos = dist(rng);
          int index_val = available_indices[expert][pos];
          // Remove the chosen index (swap with last and pop_back)
          available_indices[expert][pos] = available_indices[expert].back();
          available_indices[expert].pop_back();
          
          // Store the assignment.
          h_expert_id[t * K + l] = expert;
          h_index[t * K + l] = index_val;
      }
  }
}


    // Create reference on CPU
    for (int t = 0; t < NUM_TOKEN; t++) {
        for (int l = 0; l < K; l++) {
            int id = t * K + l;
            int expert = h_expert_id[id];
            int idx = h_index[id];
            
            if (idx >= CAPACITY) continue;
            
            int4* src = &h_input[t * hidden_stride];
            int4* dst = &h_reference[(expert * CAPACITY + idx) * hidden_stride];
            
            // Last writer wins for overlapping positions
            memcpy(dst, src, hidden_stride * sizeof(int4));
        }
    }

    // Allocate device memory
    int4 *d_input, *d_dispatched;
    int *d_expert_id, *d_index;
    
    CUDA_CHECK(cudaMalloc(&d_input, NUM_TOKEN * hidden_stride * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_expert_id, NUM_TOKEN * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_index, NUM_TOKEN * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dispatched, NUM_EXPERT * CAPACITY * hidden_stride * sizeof(int4)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, NUM_TOKEN * hidden_stride * sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_id, h_expert_id, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_index, h_index, NUM_TOKEN * K * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dispatched, 0, NUM_EXPERT * CAPACITY * hidden_stride * sizeof(int4)));

    // Launch kernel
    dispatch_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_expert_id, d_index, d_dispatched,
                                             NUM_TOKEN, K, HIDDEN_SIZE, CAPACITY);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_dispatched, d_dispatched, NUM_EXPERT * CAPACITY * hidden_stride * sizeof(int4), 
                         cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int e = 0; e < NUM_EXPERT; e++) {
        for (int c = 0; c < CAPACITY; c++) {
            for (int j = 0; j < hidden_stride; j++) {
                int4 gpu = h_dispatched[(e * CAPACITY + c) * hidden_stride + j];
                int4 ref = h_reference[(e * CAPACITY + c) * hidden_stride + j];
                
                if (gpu.x != ref.x || gpu.y != ref.y || 
                    gpu.z != ref.z || gpu.w != ref.w) {
                    if (errors++ < 10) {
                        std::cerr << "Mismatch at expert=" << e 
                                  << " capacity=" << c << " element=" << j
                                  << " GPU: (" << gpu.x << ", " << gpu.y << ", " << gpu.z << ", " << gpu.w << ")"
                                  << " REF: (" << ref.x << ", " << ref.y << ", " << ref.z << ", " << ref.w << ")\n";
                    }
                }
            }
        }
    }

    std::cout << "Correctness test: " 
              << (errors ? std::to_string(errors) + " errors" : "PASSED") 
              << std::endl;

    // Cleanup
    delete[] h_input;
    delete[] h_expert_id;
    delete[] h_index;
    delete[] h_dispatched;
    delete[] h_reference;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_expert_id));
    CUDA_CHECK(cudaFree(d_index));
    CUDA_CHECK(cudaFree(d_dispatched));
}

void test_performance() {
    const int hidden_stride = HIDDEN_SIZE / 4;
    
    // Allocate device memory
    int4 *d_input, *d_dispatched;
    int *d_expert_id, *d_index;
    
    CUDA_CHECK(cudaMalloc(&d_input, NUM_TOKEN * hidden_stride * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_expert_id, NUM_TOKEN * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_index, NUM_TOKEN * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dispatched, NUM_EXPERT * CAPACITY * hidden_stride * sizeof(int4)));

    // Warmup
    dispatch_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_expert_id, d_index, d_dispatched,
                                             NUM_TOKEN, K, HIDDEN_SIZE, CAPACITY);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Timing
    const int iterations = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        dispatch_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_expert_id, d_index, d_dispatched,
                                                 NUM_TOKEN, K, HIDDEN_SIZE, CAPACITY);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    std::cout << "Performance results:\n"
              << "Average latency: " << elapsed_ms / iterations << " ms\n"
              << "Throughput: " 
              << (NUM_TOKEN * iterations) / (elapsed_ms / 1000) << " tokens/sec\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_expert_id));
    CUDA_CHECK(cudaFree(d_index));
    CUDA_CHECK(cudaFree(d_dispatched));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "Running correctness test...\n";
    test_correctness();
    
    std::cout << "\nRunning performance test...\n";
    test_performance();
    
    return 0;
}