#pragma once

// Local include
#include "emStep.h"

// C headers
#include <CL/cl.h>

class EMstepExtended : public EMstep {
    public:

    cl_mem document_coverage_d = nullptr;
    cl_mem topic_models_d = nullptr;

    bool is_gpu_stored = false;

    EMstepExtended(size_t, size_t, size_t);
    ~EMstepExtended();

    cl_int cpuToGpuCopy();
    cl_int gpuToCpuCopy();
};