#pragma once

#include <stddef.h>

class ModelData {
    public:
    size_t document_count = 0;
    size_t vocab_size = 0;

    // Doc counts will be in document-major order
    size_t *document_counts = NULL;
    double *background_lm = NULL;

    /**
     * @brief Construct a new Model Data object
     * First parameter is number of documents, second is the vocabulary size
     */
    ModelData(size_t, size_t);
    ModelData(const ModelData &);

    ~ModelData();
};