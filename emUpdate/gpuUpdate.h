#pragma once

#include "../modelClass/emstepExtended.h"
#include "../modelClass/ModelDataExtended.h"

#include <CL/cl.h>

namespace EMupdate {
    /**
     * @brief Perform a single EM algorithm update step on the GPU.
     * 
     * @param previous The previous update step.
     * @param current The current update step. You may not have current == previous. Should already have size parameters set.
     * @param ModelDataExtended Information including the counts and background LM.
     * @param backgroundLmProb Probability of the background LM.
     * @param P_zdw_B Preallocated scratchpad memory. Should have size=vocab_size * num_documents.
     * @param P_zdw_j Preallocated scratchpad memory. Should have size=vocab_size * num_documents * num_topics.
     * @return EMstepExtended The probabilities after an update step.
     */
    void gpuUpdate(EMstepExtended &current, EMstepExtended &previous, ModelDataExtended &ModelDataExtended, double backgroundLmProb, 
        cl_mem &P_zdw_B_d, cl_mem &P_zdw_j_d, cl_mem &denoms_common_d);

    /**
     * @brief Check if the EM algorithm has converged. The order of parameters does not matter.
     * 
     * @param first The latest iteration values.
     * @param second The second-latest iteration values.
     * @return true if the algorithm has converged to a local maximum, and false otherwise.
     */
    bool isConvergedGpu(EMstepExtended &first, EMstepExtended &second, cl_mem coveragebuf_d, cl_mem modelbuf_d);
}