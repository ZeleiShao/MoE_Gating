//pass: 2.12 ms
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

int main() {
    int M = 4096, N = 256, K = 7168;  // Matrix dimensions

    // Allocate and initialize matrices on GPU
    cutlass::bfloat16_t *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(cutlass::bfloat16_t));
    cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start recording time
    gemm_cutlass_bf16_fp32(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);  // Stop recording time

    // Synchronize and measure time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    std::cout << "CUTLASS GEMM with BF16 input and FP32 output completed successfully!" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
