#include <CL/cl.h>

#include "../gpu/gpu.h"
#include "../io/io.h"
#include "../utils/listWithSize.h"

#include "sgemm.h"

extern map<const char*, cl_kernel> *available_kernels;

cl_int linalg::sgemm(double *A, double *B, double *C, unsigned int M, unsigned int N, unsigned int K) {

    // Compile the sgemm kernel
    const char *sgemm_name = "sgemm";

    cl_int err;

    if (!gpu::kernelExists(sgemm_name)) {
        const char *kernel_string = io::readKernel("kernels/sgemm.cl");
        gpu::compileKernelIfNotExists(kernel_string, sgemm_name, &err); RETURN_ON_ERROR;
    } 

    cl_kernel sgemm_kernel = (*available_kernels)[sgemm_name];

    utils::ListWithSize<double> A_info;
    A_info.items = A;
    A_info.num_items = M * K;

    utils::ListWithSize<double> B_info;
    B_info.items = B;
    B_info.num_items = K * N;

    utils::ListWithSize<double> C_info;
    C_info.items = C;
    C_info.num_items = M * N;

    // Allocate GPU memory
    cl_mem A_d = gpu::hostToDeviceCopy<double>(A_info, &err); RETURN_ON_ERROR;
    cl_mem B_d = gpu::hostToDeviceCopy<double>(B_info, &err); RETURN_ON_ERROR;
    cl_mem C_d = gpu::deviceOutputAllocate(M * N * sizeof(double), &err); RETURN_ON_ERROR;

    // FIXME - Optimize this dynamically
    // the max in OpenCL is 256
    size_t blockSize = 16;

    utils::ListWithSize<size_t> gridDim = utils::ListWithSize<size_t>();
    gridDim.num_items = 2;
    gridDim.items = new size_t[2];
    gridDim.items[0] = ceil((M * 1.0) / blockSize) * blockSize;
    gridDim.items[1] = ceil((N * 1.0) / blockSize) * blockSize;

    utils::ListWithSize<size_t> blockDim = utils::ListWithSize<size_t>();
    blockDim.num_items = 2;
    blockDim.items = new size_t[2];
    blockDim.items[0] = blockSize;
    blockDim.items[1] = blockSize;

    // Set kernel arguments
    err = clSetKernelArg(sgemm_kernel, 0, sizeof(A_d), (void*) &A_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 1, sizeof(B_d), (void*) &B_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 2, sizeof(C_d), (void*) &C_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 3, sizeof(M), (void*) &M); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 4, sizeof(N), (void*) &N); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 5, sizeof(K), (void*) &K); RETURN_ON_ERROR;

    // Launch the kernel
    err = gpu::launchKernel(sgemm_kernel, gridDim, blockDim); RETURN_ON_ERROR;

    // Copy the output back
    err = gpu::copyDeviceToHost<double>(C_d, C_info); RETURN_ON_ERROR;
    
    // Cleanup
    clReleaseMemObject(A_d);
    clReleaseMemObject(B_d);
    clReleaseMemObject(C_d);

    return err;
}