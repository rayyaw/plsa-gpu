#pragma once

#include <cstddef>

class EMstep {
    public:
    size_t num_topics = 0;
    size_t num_documents = 0;
    size_t vocab_size = 0;

    // Topic models will be in topic-major order
    double *document_coverage = nullptr;
    double *topic_models = nullptr;

    /**
     * @brief Construct a new EM Step object
     * First parameter is number of topics, second is number of documents, third is vocabulary size 
     */
    EMstep(size_t, size_t, size_t);
    EMstep(const EMstep &);
    ~EMstep();

    /**
     * @brief Generate the EM Step parameters randomly
     * If a long is specified, it will be used as seed
     */
    void genrandom();
    void genrandom(long);
};