// Local headers
#include "emStep.h"
#include "emStepExtended.h"
#include "modelData.h"

#include "../gpu/gpu.h"

// C headers
#include <CL/cl.h>
#include <string.h>

// C++ headers
#include <cstdlib>

EMstepExtended::EMstepExtended(size_t num_topics, size_t num_documents, size_t vocab_size) : EMstep(num_topics, num_documents, vocab_size) {}

EMstepExtended::~EMstepExtended() {
    if (is_gpu_stored) {
        clReleaseMemObject(document_coverage_d);
        clReleaseMemObject(topic_models_d);
    }

    delete[] document_coverage;
    delete[] topic_models;
}

cl_int EMstepExtended::cpuToGpuCopy() {
    cl_int err = CL_SUCCESS;
    document_coverage_d = gpu::hostToDeviceCopyWithRw<double>(document_coverage, num_documents * num_topics, &err);
    topic_models_d = gpu::hostToDeviceCopyWithRw<double>(topic_models, num_topics * vocab_size, &err);
    is_gpu_stored = true;

    return err;
}

cl_int EMstepExtended::gpuToCpuCopy() {
    cl_int err = CL_SUCCESS;
    err = gpu::deviceToHostCopy<double>(document_coverage_d, document_coverage, num_documents * num_topics);
    err = gpu::deviceToHostCopy<double>(topic_models_d, topic_models, num_topics * vocab_size);

    return err;
}