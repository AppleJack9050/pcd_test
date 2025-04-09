#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>  // New header for CUDA context
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "grouping_gpu.h"

// Remove the deprecated THCState reference

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int grouping_wrapper(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(xyz_tensor);

    const int *xyz = xyz_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    // Use the new CUDA stream from ATen
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouping_kernel_launcher(b, n, m, nsample, xyz, idx, stream);
    return 1;
}
