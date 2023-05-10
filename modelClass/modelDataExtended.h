#pragma once

#include "modelData.h"

// C headers
#include <CL/cl.h>
#include <stddef.h>

class ModelDataExtended : public ModelData {
    public:
    cl_mem document_counts_d = NULL;
    cl_mem background_lm_d = NULL;

    bool is_gpu_mirrored = false;

    /**
     * @brief Construct a new Model Data object
     * First parameter is number of documents, second is the vocabulary size
     */
    ModelDataExtended(size_t a, size_t b) : ModelData(a, b) {};

    // Copy the data to the GPU (by default the cl_mem's are unused)
    // Returns err if it was unable to be replicated
    cl_int mirrorGpu();

    ~ModelDataExtended();
};