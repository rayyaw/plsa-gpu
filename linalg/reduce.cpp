#include "reduce.h"

#include "../gpu/gpu.h"
#include "../utils/listWithSize.h"

#include <CL/cl.h>

using utils::ListWithSize;

#define RETURN_ON_ERROR if (CL_SUCCESS != err) return err;
#define RETURN_NULL_ON_ERROR if (CL_SUCCESS != *err) return (cl_mem) 0;

cl_int linalg::reduceWide(cl_mem &input, cl_mem &output, size_t num_major, size_t num_minor, size_t block_size) {
    cl_int err = CL_SUCCESS;

    cl_kernel reductionKernel = gpu::compileKernelFromFile("kernels/normalize.cl", "reduceAlongMajorAxisWide", &err); RETURN_ON_ERROR;

    cl_mem *argList[4] = {&input, &output, (cl_mem*) &num_major, (cl_mem*) &num_minor};
    ListWithSize<cl_mem*> args(4, argList);

    err = gpu::setKernelArgs(reductionKernel, args); RETURN_ON_ERROR;
    err = gpu::launch1dKernelWithRoundup(reductionKernel, num_major, block_size); RETURN_ON_ERROR;

    return err;

}

cl_mem linalg::reduceTall(cl_mem &input, size_t num_major, size_t num_minor, size_t block_size, cl_int *err) {
    cl_kernel reductionKernel = gpu::compileKernelFromFile("kernels/normalize.cl", "reduceAlongMajorAxisTall", err); RETURN_NULL_ON_ERROR;
    
    cl_mem old = input;

    // Iterate until only one element left
    for (size_t num_inputs = num_minor; num_inputs > 1; num_inputs = ceil((num_inputs * 1.0) / block_size)) {
        // Alloc extra data
        // FIXME - use more efficient methods like double buffering
        size_t num_outputs = ceil((1.0 * num_inputs) / block_size);
        cl_mem modelSums = gpu::deviceIntermediateAllocate(sizeof(double) * num_major * num_outputs, err); RETURN_NULL_ON_ERROR;

        cl_mem *argList[4] = {&old, &modelSums, (cl_mem*) &num_major, (cl_mem*) &num_inputs};
        ListWithSize<cl_mem*> args(4, argList);

        *err = gpu::setKernelArgs(reductionKernel, args); RETURN_NULL_ON_ERROR;
        *err = gpu::launch2dKernelWithRoundup(reductionKernel, num_major, num_inputs, 1, block_size); RETURN_NULL_ON_ERROR;

        if (num_inputs != num_minor) {
            clReleaseMemObject(old);
        }

        old = modelSums;

        if (num_outputs == 1) break;
    }

    return old;

}