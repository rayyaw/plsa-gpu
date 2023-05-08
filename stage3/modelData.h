#pragma once

#include <CL/cl.h>
#include <stddef.h>

class ModelData {
    public:
    size_t document_count = 0;
    size_t vocab_size = 0;

    // Doc counts will be in document-major order
    size_t *document_counts = NULL;
    double *background_lm = NULL;

    cl_mem document_counts_d = NULL;
    cl_mem background_lm_d = NULL;

    bool is_gpu_mirrored = false;

    /**
     * @brief Construct a new Model Data object
     * First parameter is number of documents, second is the vocabulary size
     */
    ModelData(size_t, size_t);
    ModelData(const ModelData &);

    // Copy the data to the GPU (by default the cl_mem's are unused)
    // Returns err if it was unable to be replicated
    cl_int mirrorGpu();

    ~ModelData();
};