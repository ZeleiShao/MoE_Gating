import numpy as np
import pycuda.autoinit
from pycuda import gpuarray, compiler
import time
import torch
def test_gather_kernel():
    # 参数配置
    num_expert = 256
    capacity = 140
    num_token = 4096
    k = 8
    hidden = 7168  
    VEC_SIZE = 4

    # 生成测试数据
    expert_output = torch.randn((num_expert, capacity, hidden), dtype=torch.bfloat16)
    expert_id = torch.randint(0, num_expert, (num_token, k))
    index = torch.randint(0, capacity+1, (num_token, k)) # 包含越界情况
    gates = torch.randn((num_token, k), dtype=torch.bfloat16) # bf16模拟

    # GPU数据准备
    expert_output_gpu = gpuarray.to_gpu(expert_output)
    expert_id_gpu = gpuarray.to_gpu(expert_id.flatten())
    index_gpu = gpuarray.to_gpu(index.flatten())
    gates_gpu = gpuarray.to_gpu(gates.view(np.int16))
    gathered_gpu = gpuarray.zeros((num_token, hidden), dtype=torch.bfloat16)

    # 编译内核
    mod = compiler.SourceModule("""
    #include <cuda_bf16.h>
    #include <cuda_runtime.h>

__global__ void gather_kernel_optimized(
    const __nv_bfloat16* __restrict__ expert_output,  // [num_expert, capacity, hiddensize]
    const int* __restrict__ expert_id,       // [num_token * k]
    const __nv_bfloat16* __restrict__ gates, // [num_token * k] (bf16)
    const int* __restrict__ index,           // [num_token * k]
    const int num_token,
    const int capacity,
    const int hidden,
    const int k,
    __nv_bfloat16* __restrict__ gathered             // [num_token, hiddensize]
) {
        float sum[hidden / blockDim.x];
        #pragma unroll
        for (int i=0;i<hidden / blockDim.x; ++i)
            sum[i] = 0;
        for (int tokenid = blockIdx.x; tokenid < num_token; tokenid += gridDim.x){
            #pragma unroll
            for (int j=0; j<k; ++j){
                if (index[tokenid * k + j] > capacity) continue;
                #pragma unroll
                for (int s=threadIdx.x; s<hidden; s+=blockDim.x)
                    sum[s / blockDim.x] += gates[tokenid * k + j] * expert_output[(expert_id[tokenid * k + j] * capacity + index[tokenid * k + j]) * hidden + s];
            }
            __syncthreads();
            #pragma unroll
            for (int s=threadIdx.x; s<hidden; s+=blockDim.x)
                gathered[tokenid * hidden + s] = sum[s / blockDim.x];
        }
}
    """, options=["-std=c++17"])
    print("before")
    # 运行内核
    func = mod.get_function("gather_kernel_optimized")
    block = (256, 1, 1)
    grid = ((num_token + 31) // 32, 1)
    start = time.time()
    func(expert_output_gpu, expert_id_gpu, gates_gpu, index_gpu,
         np.int32(num_token), np.int32(capacity), np.int32(hidden),
         np.int32(k),
         gathered_gpu, 
         block=block, grid=grid)
    end = time.time()
    print(f"Kernel execution time: {end - start} seconds")

    # CPU参考计算
    ref_output = np.zeros((num_token, hidden), dtype=np.float32)
    for t in range(num_token):
        for j in range(k):
            if index[t,j] >= capacity: continue
            eid = expert_id[t,j]
            idx = index[t,j]
            ref_output[t] += expert_output[eid,idx] * gates[t,j].astype(np.float32)

    # 验证结果
    gathered_cpu = gathered_gpu.get().view(np.float32).reshape(num_token, hidden)
    assert np.allclose(gathered_cpu, ref_output, atol=1e-3)

test_gather_kernel()