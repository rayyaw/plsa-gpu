#include "reduce.h"

#include "../gpu/gpu.h"
#include "../utils/listWithSize.h"

#include <CL/cl.h>

using utils::ListWithSize;

#define RETURN_NULL_ON_ERROR if (CL_SUCCESS != *err) return (cl_mem) 0;
cl_mem linalg::reduceTall(cl_mem &input, size_t num_major, size_t num_minor, size_t blockSize, cl_kernel kernel, cl_int *err) {
    cl_mem old = input;

    // Iterate until only one element left
    for (size_t num_inputs = num_minor; num_inputs > 1; num_inputs = ceil((num_inputs * 1.0) / blockSize)) {
        // Alloc extra data
        // FIXME - use more efficient methods like double buffering
        size_t num_outputs = ceil((1.0 * num_inputs) / blockSize);
        cl_mem modelSums = gpu::deviceIntermediateAllocate(sizeof(double) * num_major * num_outputs, err); RETURN_NULL_ON_ERROR;

        cl_mem *argList[4] = {&old, &modelSums, (cl_mem*) &num_major, (cl_mem*) &num_inputs};
        ListWithSize<cl_mem*> args(4, argList);

        *err = gpu::setKernelArgs(kernel, args); RETURN_NULL_ON_ERROR;
        *err = gpu::launch2dKernelWithRoundup(kernel, num_major, num_inputs, 1, blockSize); RETURN_NULL_ON_ERROR;

        if (num_inputs != num_minor) {
            clReleaseMemObject(old);
        }

        old = modelSums;

        if (num_outputs == 1) break;
    }

    return old;

}