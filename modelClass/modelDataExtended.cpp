// Local headers
#include "modelDataExtended.h"

#include "../gpu/gpu.h"

// C headers
#include <CL/cl.h>

cl_int ModelDataExtended::mirrorGpu() {
    cl_int to_return = CL_SUCCESS;

    is_gpu_mirrored = true;

    document_counts_d = gpu::hostToDeviceCopy<size_t>(document_counts, document_count * vocab_size, &to_return);
    background_lm_d = gpu::hostToDeviceCopy<double>(background_lm, vocab_size, &to_return);

    return to_return;
}

ModelDataExtended::~ModelDataExtended() {
    if (is_gpu_mirrored) {
        clReleaseMemObject(document_counts_d);
        clReleaseMemObject(background_lm_d);
    }

    delete[] document_counts;
    delete[] background_lm;
}