#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include<float.h>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cutlass/bfloat16.h>

// Define the CUTLASS GEMM operation with BF16 input and FP32 output
using ElementInputA = cutlass::bfloat16_t;  // BF16 for A
using ElementInputB = cutlass::bfloat16_t;  // BF16 for B
using ElementOutput = float;                // FP32 for C
using ElementAccumulator = float;           // FP32 for computation (accumulator)
using Layout = cutlass::layout::RowMajor;

// Define CUTLASS GEMM with BF16 Input and FP32 Output
using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, Layout,   // A matrix (BF16)
    ElementInputB, Layout,   // B matrix (BF16)
    ElementOutput, Layout,   // C matrix (FP32)
    ElementAccumulator       // Accumulator (FP32)
>;

void gemm_cutlass_bf16_fp32(const cutlass::bfloat16_t* d_A,
                             const cutlass::bfloat16_t* d_B,
                             float* d_C, int M, int N, int K) {
    // Define GEMM operator
    Gemm gemm_op;

    // Define GEMM arguments
    typename Gemm::Arguments args(
        {M, N, K},  // Matrix dimensions (MxN = MxK * KxN)
        {d_A, K},   // A matrix (MxK) - BF16
        {d_B, N},   // B matrix (KxN) - BF16
        {nullptr, N}, // No initial C matrix
        {d_C, N},   // Output matrix C (MxN) - FP32
        {1.0f, 0.0f} // Scaling factors: alpha = 1, beta = 0
    );

    // Run GEMM
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed!" << std::endl;
        return;
    }
}
void gemv_cutlass_col_means(float* d_ones,float* d_matrix,float* d_mean, int M, int N){
// Initialize ones vector
    // CUTLASS GEMV (Matrix-Vector Multiplication)
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,    // Matrix A
        float, cutlass::layout::ColumnMajor, // Vector x
        float, cutlass::layout::ColumnMajor, // Output y
        float>;

    Gemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(1,  N, M); // (M, N, K)
    typename Gemm::Arguments args(  
        problem_size,
        d_matrix, M,
        d_ones, M,
        d_mean, N,
        d_mean, N,
        {1.0f, 0.0f} // alpha = 1, beta = 0
    );

    gemm_op(args); // Run CUTLASS GEMM

    // Normalize by M
    float scale = 1.0f / M;
    cudaMemcpy(h_mean, d_mean, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Scale results
    for (int j = 0; j < N; j++) {  
        h_mean[j] *= scale;
    }

}

// first kernel: topk + softmax + l_aux; 
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}
// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

__device__ void cumsum(int* token_place,int* expert_tokens,const int num_experts, const int num_token, const int topk){ 
  __shared__ int smem[num_experts];
  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x;
  
  if (threadIdx.x < num_experts){
    smem[threadIdx.x] = expert_tokens[threadIdx.x];
  }
   
  token_place[threadIdx.x] = 0;
  cp_async_fence();
}

// kernel3: from a sparse representation get dispatch & combine  
__device__ void get_dispatched_input(
  const int4* input, //[num_token, hidden_size]
  const int* t_eid, // [tokenid][k] = expertid
  const int* t_eoffset, // [tokenid][k] = the ith element of the expert
  int4* dispatched_input, // [e, c, hidden_size]
  const int num_token,
  const int k, //number of the selected experts
  const int hidden_size,
  const int capacity
){
  int THREAD_TOKEN_NUM = num_token / gridDim.x;
  int start_token = THREAD_TOKEN_NUM * blockIdx.x;
  const int hidden_stride = hidden_size / 8;
    __shared__ struct {
        int eid[THREAD_TOKEN_NUM *k];
        int eoffset[THREAD_TOKEN_NUM *k];
    } cache;  
    // 协作加载索引数据 works only when blockDim.x< THREAD_TOKEN_NUM * k
    if (threadIdx.x < THREAD_TOKEN_NUM * k){ //4 * 40
      cache.eid[threadIdx.x] = t_eid[threadIdx.x + start_token * k];
      cache.eoffset[threadIdx.x] = t_eoffset[threadIdx.x + start_token * k];
    }
    __syncthreads();

  // for i in range(thread_token_num*blockDim.x,thread_token_num*blockDim.x+thread_token_num):
  //     for j in range(k): k个expert
  //      for l in range(hidden_size):
  //        dispatched[t_eid[i][j]][t_eoffset[i][j]][l] = input[i][j][l]
  #pragma unroll
  for (int tid=0; tid < THREAD_TOKEN_NUM; tid += 1){ 
    if (tid + start_token >= num_token) return;
   #pragma unroll
   for (int j=0; j<k; ++j){
    int eid = cache.eid[tid* k + j];
    if(eid == -1) continue; //边界检查，某些token被drop了
    int eoffset = cache.eoffset[tid* k + j];
    #pragma unroll
    for (int s = threadIdx.x; s < hidden_size; s += blockDim.x)
      dispatched_input[(eid * capacity + eoffset) * hidden_stride + s] = input_start[(start_token + tid) * hidden_stride + s];
  }
}
}

// kernel 4: combined kernel



int main() {
    int num_tokens = 4096, num_expert = 128, hidden_size = 7168, topk=4;  // Matrix dimensions

    // Allocate and initialize matrices on GPU
    cutlass::bfloat16_t *input, *weight;
    float *logits;
    float *d_ones;
    float *d_mean;
    cudaMalloc(&input, num_tokens * hidden_size * sizeof(cutlass::bfloat16_t));
    cudaMalloc(&weight, num_expert * hidden_size * sizeof(cutlass::bfloat16_t));
    cudaMalloc(&logits, num_tokens * num_expert * sizeof(float));
    cudaMalloc(&d_topk, num_tokens * topk * sizeof(float)); //val
    cudaMalloc(&d_token_id, num_tokens * topk * sizeof(int)); //row
    cudaMalloc(&d_expert_id, num_tokens * topk * sizeof(int)); // col
    cudaMalloc(&d_ones, num_tokens * sizeof(float));
    cudaMalloc(&d_mean, num_expert * sizeof(float));

    gemm_cutlass_bf16_fp32(input, weight, logits, num_tokens, num_expert, hidden_size);
    rowwise_softmax(logits, num_tokens, num_expert);
    rowwise_topk(logits, d_topk, d_token_id, d_expert_id, num_tokens, num_expert, topk);
    gemv_cutlass_col_means(d_ones,logits,d_mean, num_tokens, num_expert);

    //zip(row, col, val), thrust 
    //thrust reduce histogram to get "ce", and also get whether this expert is full
    //cumsum
    //dispatch
    




    




    // Free memory
    cudaFree(input);
    cudaFree(weight);
    cudaFree(logits);
    cudaFree(d_topk);
    cudaFree(d_token_id);
    cudaFree(d_expert_id);
    return 0;
}
