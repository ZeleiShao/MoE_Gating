#include <cuda_runtime.h>
#include <iostream>

#define ROWS 3
#define COLS 5
#define K 3

__device__ void insert_top_k(float *top_k, float val, int k) {
    // Insert in sorted order
    for (int i = 0; i < k; i++) {
        if (val > top_k[i]) {
            for (int j = k - 1; j > i; j--) {
                top_k[j] = top_k[j - 1];
            }
            top_k[i] = val;
            break;
        }
    }
}

__global__ void find_top_k(float *matrix, float *topk, int* token_id, int* expert_id, int rows, int cols,int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float top_k[K];
    for (int i = 0; i < k; i++) top_k[i] = -1e9f;  // Initialize with small values

    for (int col = 0; col < cols; col++) {
        insert_top_k(top_k, matrix[row * cols + col], k);
    }

    for (int i = 0; i < k; i++) {
        topk[row * k + i] = top_k[i];
        token_id[row * k + i] = row;
        expert_id[row * k + i] =col;
    }

}

int main() {
    float h_matrix[ROWS][COLS] = {
        {1.0, 4.5, 2.3, 7.2, 5.1},
        {3.1, 6.2, 1.9, 8.0, 2.4},
        {9.3, 3.3, 6.6, 2.1, 7.8}
    };

    float h_topk[ROWS][K];
    int h_token_id[ROWS][K];
    int h_expert_id[ROWS][K];

    float *d_matrix, *d_topk;
    cudaMalloc(&d_matrix, ROWS * COLS * sizeof(float));
    cudaMalloc(&d_topk, ROWS * K * sizeof(float));
    cudaMalloc(&d_token_id, ROWS * K * sizeof(int));
    cudaMalloc(&d_expert_id, ROWS * K * sizeof(int));
    cudaMemcpy(d_matrix, h_matrix, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (ROWS + threadsPerBlock - 1) / threadsPerBlock;
    find_top_k<<<numBlocks, threadsPerBlock>>>(d_matrix, d_topk, d_token_id, d_expert_id, ROWS, COLS, K);

    cudaMemcpy(h_topk, d_topk, ROWS * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_token_id, d_token_id, ROWS * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expert_id, d_expert_id, ROWS * K * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Top-K values per row:" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < K; j++) {
            std::cout << h_topk[i][j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_matrix);
    cudaFree(d_topk);
    cudaFree(d_token_id);
    cudaFree(d_expert_id);

    return 0;
}
