__global__ void gather_kernel(
    const int4* expert_output,//[num_expert, capacity, hiddensize]
    const int* token_id, //[num_token * k]
    const int* expert_id, //[num_token * k]
    const int* gates, //[num_token * k]  bf16
    const int* index,//[num_token * k]
    const int* device_expertid,
    const int device_expert_num,
    const int capacity,
    const int k,
    const int hidden,
    int4* gathered //[num_token, hiddensize]
    ) {
        __shared__ sum[hidden / blockDim.x]

        for (int i = 0; i < device_expert_num; ++i){
            int expert_id = device_expertid[i];
        }
}