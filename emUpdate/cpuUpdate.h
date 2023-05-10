#pragma once

#include "../modelClass/emStep.h"
#include "../modelClass/modelData.h"

namespace EMupdate {
    /**
     * @brief Perform a single EM algorithm update step on the CPU.
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

    /**
     * @brief Check if the EM algorithm has converged. The order of parameters does not matter.
     * 
     * @param first The latest iteration values.
     * @param second The second-latest iteration values.
     * @return true if the algorithm has converged to a local maximum, and false otherwise.
     */
    bool isConverged(const EMstep &first, const EMstep &second);
}