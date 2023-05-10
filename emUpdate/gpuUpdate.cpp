#include "gpuUpdate.h"

// Local includes
#include "../gpu/gpu.h"
#include "../linalg/reduce.h"
#include "../linalg/sgemm.h"
#include "../modelClass/emStepExtended.h"
#include "../utils/listWithSize.h"

// C includes
#include <CL/cl.h>

// C++ includes
#include <iostream>

// Std using
using std::cerr;
using std::cout;
using std::endl;

// Local using
using utils::ListWithSize;

#define RETURN_ON_ERROR if (err != CL_SUCCESS) return err;
#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

#define BLOCK_SIZE 256

// Correct values
// Model Error: 3.38041
// Coverage Error: 132.969
// FIXME - Allow async kernels/ multiple command queues
void EMupdate::gpuUpdate(EMstepExtended &current, EMstepExtended &previous, ModelDataExtended &ModelDataExtended, double backgroundLmProb,
    cl_mem &P_zdw_B_d, cl_mem &P_zdw_j_d, cl_mem &denoms_common_d) {
    
    // Overhead - GPU setup
    cl_int err = CL_SUCCESS;
    
    // Initialize kernels
    // These are lazy-loaded (if already loaded, just reuse) so not much overhead to call in every iteration
    cl_kernel backgroundPriorKernel = gpu::compileKernelFromFile("kernels/estep.cl", "computeBackgroundPrior", &err); PRINT_ON_ERROR;
    cl_kernel topicPriorKernel = gpu::compileKernelFromFile("kernels/estep.cl", "computeTopicPrior", &err); PRINT_ON_ERROR;

    cl_kernel documentUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeDocumentUpdate", &err); PRINT_ON_ERROR;
    cl_kernel topicUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeTopicUpdate", &err); PRINT_ON_ERROR;

    cl_kernel normalizationKernel = gpu::compileKernelFromFile("kernels/normalize.cl", "normalizeAlongMajorAxis", &err); PRINT_ON_ERROR;

    // E-step

    // Document coverage is passed in as transposed
    err = linalg::sgemmDevice(previous.document_coverage_d, previous.topic_models_d, denoms_common_d, previous.num_documents, previous.vocab_size, previous.num_topics); PRINT_ON_ERROR;
    
    // P(Z_d,w | B)
    cl_mem *backgroundPriorArgsList[6] = {&ModelDataExtended.background_lm_d, &denoms_common_d, &P_zdw_B_d, 
        (cl_mem*) &backgroundLmProb, (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size};
        
    ListWithSize<cl_mem*> backgroundPriorArgs(6, backgroundPriorArgsList);
    err = gpu::setKernelArgs(backgroundPriorKernel, backgroundPriorArgs); PRINT_ON_ERROR;

    err = gpu::launch2dKernelWithRoundup(backgroundPriorKernel,
        previous.num_documents, previous.vocab_size, 
        1, BLOCK_SIZE); PRINT_ON_ERROR;

    // P(Z_d,w | theta_j)
    cl_mem *topicPriorArgsList[9] = {&previous.document_coverage_d, &previous.topic_models_d, &denoms_common_d, 
        &ModelDataExtended.background_lm_d, &P_zdw_j_d, 
        (cl_mem*) &backgroundLmProb, (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size, (cl_mem*) &previous.num_topics};
    
    ListWithSize<cl_mem*> topicPriorArgs(9, topicPriorArgsList);
    err = gpu::setKernelArgs(topicPriorKernel, topicPriorArgs); PRINT_ON_ERROR;

    err = gpu::launch3dKernelWithRoundup(topicPriorKernel,
        previous.num_topics, previous.num_documents, previous.vocab_size,
        1, 1, BLOCK_SIZE); PRINT_ON_ERROR;

    // M-step
    // Document coverage
    cl_mem *documentUpdateArgsList[7] = {&P_zdw_B_d, &P_zdw_j_d, &ModelDataExtended.document_counts_d, &current.document_coverage_d,
        (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size, (cl_mem*) &previous.num_topics};

    ListWithSize<cl_mem*> documentUpdateArgs(7, documentUpdateArgsList);
    err = gpu::setKernelArgs(documentUpdateKernel, documentUpdateArgs); PRINT_ON_ERROR;

    err = gpu::launch2dKernelWithRoundup(documentUpdateKernel,
        previous.num_topics, previous.num_documents,
        1, BLOCK_SIZE); PRINT_ON_ERROR;


    // Topic models
    cl_mem *topicUpdateArgsList[7] = {&P_zdw_B_d, &P_zdw_j_d, &ModelDataExtended.document_counts_d, &current.topic_models_d,
        (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size, (cl_mem*) &previous.num_topics};
    
    ListWithSize<cl_mem*> topicUpdateArgs(7, topicUpdateArgsList);
    err = gpu::setKernelArgs(topicUpdateKernel, topicUpdateArgs); PRINT_ON_ERROR;

    err = gpu::launch2dKernelWithRoundup(topicUpdateKernel,
        previous.num_topics, previous.vocab_size,
        1, BLOCK_SIZE); PRINT_ON_ERROR;
        

    // Alloc extra data
    cl_mem coverageSums = gpu::deviceIntermediateAllocate(sizeof(double) * previous.num_documents, &err); PRINT_ON_ERROR;
    // Calculate normalization denominators
    cl_mem *normalizeCoverageArgsList[4] = {&current.document_coverage_d, &coverageSums, (cl_mem*) &previous.num_documents, (cl_mem*) &previous.num_topics};
    ListWithSize<cl_mem*> normalizeCoverageArgs(4, normalizeCoverageArgsList);
 
    err = linalg::reduceWide(current.document_coverage_d, coverageSums, previous.num_documents, previous.num_topics, BLOCK_SIZE);

    cl_mem modelSums = linalg::reduceTall(current.topic_models_d, previous.num_topics, previous.vocab_size, BLOCK_SIZE, &err); PRINT_ON_ERROR;

    cl_mem *normalizeModelArgsList[4] = {&current.topic_models_d, &modelSums, (cl_mem*) &previous.num_topics, (cl_mem*) &previous.vocab_size};
    ListWithSize<cl_mem*> normalizeModelArgs(4, normalizeModelArgsList);
    
    // Normalize the output
    err = gpu::setKernelArgs(normalizationKernel, normalizeCoverageArgs); PRINT_ON_ERROR;
    err = gpu::launch2dKernelWithRoundup(normalizationKernel, previous.num_documents, previous.num_topics, BLOCK_SIZE, 1); PRINT_ON_ERROR;

    err = gpu::setKernelArgs(normalizationKernel, normalizeModelArgs); PRINT_ON_ERROR;
    err = gpu::launch2dKernelWithRoundup(normalizationKernel, previous.num_topics, previous.vocab_size, 1, BLOCK_SIZE); PRINT_ON_ERROR;

    clReleaseMemObject(modelSums); PRINT_ON_ERROR;
    clReleaseMemObject(coverageSums); PRINT_ON_ERROR;
}

bool EMupdate::isConvergedGpu(EMstepExtended &first, EMstepExtended &second, cl_mem coveragebuf_d, cl_mem modelbuf_d) {
    // Check for convergence by subtracting the vectors and using an L1-norm over all values
    cl_int err = CL_SUCCESS;
    cl_kernel differenceKernel = gpu::compileKernelFromFile("kernels/basic.cl", "vectorAbsDiff", &err); PRINT_ON_ERROR;

    // Vector of absolute differences - document coverage
    size_t num_coverage = first.num_documents * first.num_topics;
    
    cl_mem *coverageArgsList[4] = {&first.document_coverage_d, &second.document_coverage_d, &coveragebuf_d, (cl_mem*) &num_coverage};
    ListWithSize<cl_mem*> coverageArgs(4, coverageArgsList);

    err = gpu::setKernelArgs(differenceKernel, coverageArgs); PRINT_ON_ERROR;
    err = gpu::launch1dKernelWithRoundup(differenceKernel, num_coverage, BLOCK_SIZE);

    // Vector of absolute differences - topic models 
    size_t num_models = first.num_topics * first.vocab_size;

    cl_mem *modelArgsList[4] = {&first.topic_models_d, &second.topic_models_d, &modelbuf_d, (cl_mem*) &num_models};
    ListWithSize<cl_mem*> modelArgs(4, modelArgsList);

    err = gpu::setKernelArgs(differenceKernel, modelArgs); PRINT_ON_ERROR;
    err = gpu::launch1dKernelWithRoundup(differenceKernel, num_models, BLOCK_SIZE); PRINT_ON_ERROR;

    // Reduce to get final error
    cl_mem coverage_err = linalg::reduceTall(coveragebuf_d, 1, num_coverage, BLOCK_SIZE, &err); PRINT_ON_ERROR;
    cl_mem model_err = linalg::reduceTall(modelbuf_d, 1, num_models, BLOCK_SIZE, &err); PRINT_ON_ERROR;

    double error_norm_coverage, error_norm_model;

    gpu::deviceToHostCopy<double>(coverage_err, &error_norm_coverage, 1);
    gpu::deviceToHostCopy<double>(model_err, &error_norm_model, 1);

    cout << "Model error: " << error_norm_model << endl;
    cout << "Coverage error: " << error_norm_coverage << endl;
    cout << endl;

    clReleaseMemObject(coverage_err);
    clReleaseMemObject(model_err);

    return (error_norm_model < 2 && error_norm_coverage < 30);
}