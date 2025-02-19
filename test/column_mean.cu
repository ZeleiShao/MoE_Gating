#include <cutlass/cutlass.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/reduction/device/tensor_reduce.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>

#include <iostream>

//nvcc -I /scratch/bcjw/zshao3/cutlass/include -I /scratch/bcjw/zshao3/cutlass/tools/util/include -o column_mean column_mean.cu 

int main() {
    // Define matrix dimensions
    const int M = 4; // Rows
    const int N = 3; // Columns

    // Define the matrix in row-major layout
    using Layout = cutlass::layout::RowMajor;
    using Element = float;
    using TensorRef = cutlass::TensorRef<Element, Layout>;

    // Allocate and initialize host matrix
    cutlass::HostTensor<Element, Layout> host_matrix({M, N});
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            host_matrix.at({i, j}) = static_cast<Element>(i + j);
        }
    }

    // Allocate device memory and copy data to device
    cutlass::DeviceAllocation<Element> device_matrix(M * N);
    device_matrix.copy_from_host(host_matrix.host_data());

    // Create a TensorRef for the device matrix
    TensorRef tensor_ref(device_matrix.get(), Layout(M, N));

    // Allocate device memory for the result (column sums)
    cutlass::DeviceAllocation<Element> device_column_sums(N);
    device_column_sums.fill(0);

    // Perform column-wise reduction
    for (int j = 0; j < N; ++j) {
        // Define a TensorRef for the j-th column
        TensorRef column_ref(tensor_ref.data() + j, Layout(M, 1));

        // Reduce the column to a single value
        cutlass::reduction::device::Reduce<
            Element, // Data type
            cutlass::plus<Element>, // Reduction operator
            1 // Vector length
        >(
            column_sums.get() + j, // Output
            column_ref, // Input
            M // Number of elements to reduce
        );
    }

    // Copy the result back to the host
    cutlass::HostTensor<Element, Layout> host_column_sums({1, N});
    host_column_sums.copy_from_device(device_column_sums.get());

    // Print the result
    std::cout << "Column sums: ";
    for (int j = 0; j < N; ++j) {
        std::cout << host_column_sums.at({0, j}) << " ";
    }
    std::cout << std::endl;

    return 0;
}