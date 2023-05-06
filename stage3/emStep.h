#pragma once

// C headers
#include <stddef.h>

// Local headers
#include "modelData.h"

class EMstep {
    public:
    size_t num_topics = 0;
    size_t num_documents = 0;
    size_t vocab_size = 0;

    // Topic models will be in topic-major order
    double *document_coverage = NULL;
    double *topic_models = NULL;

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

/**
 * @brief Perform a single EM algorithm update step on the CPU.
 * Note that the two 
 * 
 * @param previous The previous update step.
 * @param current The current update step. You may not have current == previous. Should already have size parameters set.
 * @param ModelData Information including the counts and background LM.
 * @param backgroundLmProb Probability of the background LM.
 * @param P_zdw_B Preallocated scratchpad memory. Should have size=vocab_size * num_documents.
 * @param P_zdw_j Preallocated scratchpad memory. Should have size=vocab_size * num_documents * num_topics.
 * @return EMstep The probabilities after an update step.
 * 
 */
void cpuUpdate(EMstep &current, const EMstep &previous, ModelData &ModelData, double backgroundLmProb,
    double *P_zdw_B, double *P_zdw_j);

// You should comment this out when compiling without a GPU
void gpuUpdate(EMstep &current, const EMstep &previous, ModelData &ModelData, double backgroundLmProb,
    double *P_zdw_B, double *P_zdw_j, double *doc_coverage_T, double *denoms_common);

/**
 * @brief Check if the EM algorithm has converged. The order of parameters does not matter.
 * 
 * @param first The latest iteration values.
 * @param second The second-latest iteration values.
 * @return true if the algorithm has converged to a local maximum, and false otherwise.
 */
bool isConverged(const EMstep &first, const EMstep &second);