#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(err); \
    } \
}
// Number of experts per token.

#define CAPACITY 300
#define NUM_TOKEN 4096
#define K 8
#define HIDDEN_SIZE 7168
#define NUM_EXPERT 256
#define GRID_SIZE 512
#define BLOCK_SIZE 1024
__global__ void gather_kernel_optimized(
    const __nv_bfloat162* __restrict__ expert_output,  // [num_expert, capacity, half_hiddensize]
    const int* __restrict__ expert_id,       // [num_token * k]
    const float* __restrict__ gates, // [num_token * k] (bf16)
    const int* __restrict__ index,           // [num_token * k]
    const int num_token,
    const int capacity,
    const int k,
    __nv_bfloat162* __restrict__ gathered             // [num_token, half_hiddensize]
) {
    const int half_hidden = HIDDEN_SIZE / 2;
        __nv_bfloat162 sum[(half_hidden + BLOCK_SIZE - 1) / BLOCK_SIZE];
        __nv_bfloat162 zero = __float2bfloat162_rn(0.0f);
        for (int tokenid = blockIdx.x; tokenid < num_token; tokenid += GRID_SIZE){
            #pragma unroll
            for (int i=0;i<(half_hidden + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
                sum[i] = zero;
            #pragma unroll
            for (int j=0; j<k; ++j){
                int id = tokenid * k + j;
                int idx = index[id];
                if (idx >= capacity) continue;
                int expert = expert_id[id];
                __nv_bfloat162 gate = __float2bfloat162_rn(gates[id]);
                #pragma unroll
                for (int s=threadIdx.x; s<half_hidden; s+=BLOCK_SIZE){
                    sum[s / BLOCK_SIZE] = __hfma2(gate, expert_output[(expert * capacity + idx) * half_hidden + s], sum[s / BLOCK_SIZE]);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int s=threadIdx.x; s<half_hidden; s+=blockDim.x)
                gathered[tokenid * half_hidden + s] = sum[s / BLOCK_SIZE];
        }
}

//deepseek:
#include <cooperative_groups.h>

// template <int VEC_SIZE, int K, int HIDDEN_SIZE>
// __global__ void gather_kernel_optimized(
//     const int4* __restrict__ expert_output,  // [num_expert, capacity, hiddensize]
//     const int* __restrict__ token_id,        // [num_token * k] 
//     const int* __restrict__ expert_id,       // [num_token * k]
//     const __nv_bfloat16* __restrict__ gates, // [num_token * k] (bf16)
//     const int* __restrict__ index,           // [num_token * k]
//     const int num_token,
//     const int capacity,
//     int4* __restrict__ gathered             // [num_token, hiddensize]
// ) {
//     // 协作组和共享内存定义
//     namespace cg = cooperative_groups;
//     const cg::thread_block block = cg::this_thread_block();
    
//     const int tid = block.thread_rank();
//     const int vec_lane = tid % VEC_SIZE; // 向量通道
//     const int vec_group = tid / VEC_SIZE; // 向量组

//     // 初始化共享内存
//     if (tid < HIDDEN_SIZE) shmem_sums[tid] = 0.0f;
//     block.sync();

//     // 主循环处理每个token
//     for (int token_idx = blockIdx.x; token_idx < num_token; token_idx += gridDim.x) {
//         float local_sum[HIDDEN_SIZE/VEC_SIZE] = {0};

//         // 处理K个专家
//         for (int j = 0; j < K; ++j) {
//             const int flat_idx = token_idx * K + j;
//             const int cap_idx = index[flat_idx];
//             if (cap_idx >= capacity) continue;

//             // 计算专家数据地址
//             const int expert_offset = (expert_id[flat_idx] * capacity + cap_idx) * HIDDEN_SIZE;
//             const int4* expert_data = expert_output + (expert_offset / VEC_SIZE);
            
//             // 向量化加载和计算
//             const float gate = __bfloat162float(gates[flat_idx]);
//             #pragma unroll
//             for (int v = vec_group; v < HIDDEN_SIZE/VEC_SIZE; v += blockDim.x/VEC_SIZE) {
//                 int4 vec = expert_data[v];
//                 float* elements = reinterpret_cast<float*>(&vec);
//                 #pragma unroll
//                 for (int s = 0; s < VEC_SIZE; ++s) {
//                     local_sum[v] += elements[s] * gate;
//                 }
//             }
//         }

//         // 归约到共享内存
//         #pragma unroll
//         for (int v = 0; v < HIDDEN_SIZE/VEC_SIZE; ++v) {
//             atomicAdd(&shmem_sums[v * VEC_SIZE + vec_lane], local_sum[v]);
//         }
//         block.sync();

//         // 写入全局内存
//         if (vec_group < HIDDEN_SIZE/VEC_SIZE) {
//             int4 vec;
//             float* ptr = reinterpret_cast<float*>(&vec);
//             #pragma unroll
//             for (int s = 0; s < VEC_SIZE; ++s) {
//                 ptr[s] = shmem_sums[vec_group * VEC_SIZE + s];
//             }
//             gathered[token_idx * (HIDDEN_SIZE/VEC_SIZE) + vec_group] = vec;
//         }

//         // 重置共享内存
//         if (tid < HIDDEN_SIZE) shmem_sums[tid] = 0.0f;
//         block.sync();
//     }
// }

int main() {
    // Problem dimensions.
    const int num_token = NUM_TOKEN;
    const int capacity = CAPACITY;
    const int hidden = HIDDEN_SIZE;
    const int half_hidden = HIDDEN_SIZE / 2; //forgot to change, here it's actually half_hidden
    const int num_expert = NUM_EXPERT;

    // Calculate sizes.
    size_t size_expert_output = num_expert * capacity * half_hidden * sizeof(__nv_bfloat162);
    size_t size_expert_id   = num_token * K * sizeof(int);
    size_t size_gates       = num_token * K * sizeof(float);
    size_t size_index       = num_token * K * sizeof(int);
    size_t size_gathered    = num_token * half_hidden * sizeof(__nv_bfloat162);

    // Allocate host memory.
    __nv_bfloat162* h_expert_output = new __nv_bfloat162[num_expert * capacity * half_hidden];
    int* h_expert_id             = new int[num_token * K];
    float* h_gates       = new float[num_token * K];
    int* h_index                 = new int[num_token * K];
    __nv_bfloat162* h_gathered    = new __nv_bfloat162[num_token * half_hidden];
    // For reference, we store results in float.
    __nv_bfloat162* h_gathered_ref   = new __nv_bfloat162[num_token * half_hidden];

    // Initialize host data with random values.
    std::default_random_engine rng(0);
    std::uniform_real_distribution<float> dist_float(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_expert(0, num_expert - 1);
    std::uniform_int_distribution<int> dist_index(0, capacity - 1);

    // Initialize expert_output.
    for (int i = 0; i < num_expert * capacity * half_hidden; ++i) {
        float val = dist_float(rng);
        h_expert_output[i] = __float2bfloat162_rn(val);
    }
    // Initialize gates.
    for (int i = 0; i < num_token * K; ++i) {
        h_gates[i] = dist_float(rng);
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


    // Compute the reference result on the CPU.
    for (int t = 0; t < num_token; ++t) {
        // Zero initialize the output for this token.
        for (int s = 0; s < half_hidden; ++s) {
            h_gathered_ref[t * half_hidden + s] = __float2bfloat162_rn(0.0f);
        }
        // Accumulate contributions from each expert.
        for (int j = 0; j < K; ++j) {
            int idx = t * K + j;
            if (h_index[idx] >= capacity) continue;
            int expert = h_expert_id[idx];
            float gate_val = h_gates[idx]; 
            __nv_bfloat162 gate = __float2bfloat162_rn(gate_val);
            for (int s = 0; s < half_hidden; ++s) {
                int offset = (expert * capacity + h_index[idx]) * half_hidden + s;
                __nv_bfloat162 expert_val = h_expert_output[offset];
                h_gathered_ref[t * half_hidden + s] = __hadd2(__hmul2(gate, expert_val) , h_gathered_ref[t * half_hidden + s]);
            }
        }
    }

    // Allocate device memory.
    __nv_bfloat162* d_expert_output;
    int* d_expert_id;
    float* d_gates;
    int* d_index;
    __nv_bfloat162* d_gathered;

    CUDA_CHECK(cudaMalloc(&d_expert_output, size_expert_output));
    CUDA_CHECK(cudaMalloc(&d_expert_id,   size_expert_id));
    CUDA_CHECK(cudaMalloc(&d_gates,       size_gates));
    CUDA_CHECK(cudaMalloc(&d_index,       size_index));
    CUDA_CHECK(cudaMalloc(&d_gathered,    size_gathered));

    // Copy host data to device.
    CUDA_CHECK(cudaMemcpy(d_expert_output, h_expert_output, size_expert_output, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_id,   h_expert_id,     size_expert_id,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gates,       h_gates,         size_gates,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_index,       h_index,         size_index,       cudaMemcpyHostToDevice));

    // Launch the kernel.
    gather_kernel_optimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_expert_output,
                                                     d_expert_id,
                                                     d_gates,
                                                     d_index,
                                                     num_token,
                                                     capacity,
                                                     K,
                                                     d_gathered);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the gathered results back to the host.
    CUDA_CHECK(cudaMemcpy(h_gathered, d_gathered, size_gathered, cudaMemcpyDeviceToHost));

    // Verify correctness by comparing against the CPU reference.
    float max_error = 0.0f;
    for (int i = 0; i < num_token * half_hidden; ++i) {
        float gpu_high = __high2float(h_gathered[i]);
        float cpu_high = __high2float(h_gathered_ref[i]);
        float gpu_low = __low2float(h_gathered[i]);
        float cpu_low = __low2float(h_gathered_ref[i]);
        float err = max(fabs(gpu_high - cpu_high)/fabs(cpu_high) , fabs(gpu_low - cpu_low)/fabs(cpu_low));
        if (err > max_error)
            max_error = err;
    }
    std::cout << "Max relative error between GPU and CPU results: " << max_error << std::endl;

    // Efficiency measurement: run the kernel many times and average the runtime.
    const int iterations = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gather_kernel_optimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_expert_output,
                                                         d_expert_id,
                                                         d_gates,
                                                         d_index,
                                                         num_token,
                                                         capacity,
                                                         K,
                                                         d_gathered);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Average kernel execution time: " << (elapsed_ms / iterations) << " ms" << std::endl;

    // Cleanup.
    delete[] h_expert_output;
    delete[] h_expert_id;
    delete[] h_gates;
    delete[] h_index;
    delete[] h_gathered;
    delete[] h_gathered_ref;
    CUDA_CHECK(cudaFree(d_expert_output));
    CUDA_CHECK(cudaFree(d_expert_id));
    CUDA_CHECK(cudaFree(d_gates));
    CUDA_CHECK(cudaFree(d_index));
    CUDA_CHECK(cudaFree(d_gathered));

    return 0;
}