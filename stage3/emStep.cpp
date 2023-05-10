// Local headers
#include "emStep.h"
#include "modelData.h"

#include "../gpu/gpu.h"
#include "../linalg/reduce.h"
#include "../linalg/sgemm.h"

// C headers
#include <CL/cl.h>
#include <string.h>

// C++ headers
#include <cstdlib>
#include <iostream>

#define RETURN_ON_ERROR if (err != CL_SUCCESS) return err;
#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

#define BLOCK_SIZE 256

// Local ussing
using linalg::sgemm;
using utils::ListWithSize;

// STD using
using std::cerr;
using std::cout;
using std::endl;
using std::rand;
using std::srand;


EMstep::EMstep(size_t num_topics, size_t num_documents, size_t vocab_size) {
    this -> num_topics = num_topics;
    this -> num_documents = num_documents;
    this -> vocab_size = vocab_size;

    document_coverage = new double[num_documents * num_topics];
    topic_models = new double[num_topics * vocab_size];
}

EMstep::EMstep(const EMstep &other) {
    num_topics = other.num_topics;
    num_documents = other.num_documents;
    vocab_size = other.vocab_size;

    document_coverage = new double[num_documents * num_topics];
    topic_models = new double[num_topics * vocab_size];

    memcpy(document_coverage, other.document_coverage, num_documents *  num_topics * sizeof(double));
    memcpy(topic_models, other.topic_models, num_topics * vocab_size * sizeof(double));
}

EMstep::~EMstep() {
    if (is_gpu_stored) {
        clReleaseMemObject(document_coverage_d);
        clReleaseMemObject(topic_models_d);
    }

    delete[] document_coverage;
    delete[] topic_models;
}

void EMstep::genrandom() {
    for (size_t document = 0; document < num_documents; document++) {
        size_t prob_total = 0;

        for (size_t topic = 0; topic < num_topics; topic++) {
            unsigned int val = rand();

            document_coverage[document * num_topics + topic] = val;
            prob_total += val;
        }

        for (size_t topic = 0; topic < num_topics; topic++) {
            document_coverage[document * num_topics + topic] /= prob_total;
        }
    }

    for (size_t topic = 0; topic < num_topics; topic++) {
        size_t prob_total = 0;

        for (size_t word = 0; word < vocab_size; word++) {
            unsigned int val = rand();

            topic_models[topic * vocab_size + word] = val;
            prob_total += val;
        }

        for (size_t word = 0; word < vocab_size; word++) {
            topic_models[topic * vocab_size + word] /= prob_total;
        }
    }
}

void EMstep::genrandom(long seed) {
    srand(seed);
    genrandom();
}

cl_int EMstep::cpuToGpuCopy() {
    cl_int err = CL_SUCCESS;
    document_coverage_d = gpu::hostToDeviceCopyWithRw<double>(document_coverage, num_documents * num_topics, &err);
    topic_models_d = gpu::hostToDeviceCopyWithRw<double>(topic_models, num_topics * vocab_size, &err);
    is_gpu_stored = true;

    return err;
}

cl_int EMstep::gpuToCpuCopy() {
    cl_int err = CL_SUCCESS;
    err = gpu::deviceToHostCopy<double>(document_coverage_d, document_coverage, num_documents * num_topics);
    err = gpu::deviceToHostCopy<double>(topic_models_d, topic_models, num_topics * vocab_size);

    return err;
}

void cpuUpdate(EMstep &current, const EMstep &previous, ModelData &modelData, double backgroundLmProb,
    double *P_zdw_B, double *P_zdw_j) {
    // E-step
    // Topic-major, then document-major order
    double topicLmProb = 1 - backgroundLmProb;

    // We don't allocate our scratchpad memory since malloc() is slow and we can reuse across iterations

    // P(Z_d,w | B)
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            double P_zdw_B_num = backgroundLmProb * modelData.background_lm[word];
            double P_zdw_B_denom = backgroundLmProb * modelData.background_lm[word];

            double sum_of_all_topics = 0;

            for (size_t i = 0; i < previous.num_topics; i++) {
                sum_of_all_topics += previous.document_coverage[i * previous.num_documents + document] 
                                    * previous.topic_models[i * previous.vocab_size + word];
            }

            sum_of_all_topics *= topicLmProb;

            P_zdw_B_denom += sum_of_all_topics;

            P_zdw_B[document * previous.vocab_size] = P_zdw_B_num / P_zdw_B_denom;
        }
    }

    // P(Z_d,w | theta_j)
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            double P_zdw_j_denom_common = 0;

            // Sum over all topics
            for (size_t i = 0; i < previous.num_topics; i++) {
                P_zdw_j_denom_common += previous.document_coverage[i * previous.num_documents + document] * previous.topic_models[i * previous.vocab_size + word];
            }

            // For each topic/document pair
            for (size_t topic = 0; topic < previous.num_topics; topic++) {
                double P_zdw_j_num = previous.document_coverage[topic * previous.num_documents + document] * previous.topic_models[topic * previous.vocab_size + word];
                double P_zdw_j_denom = P_zdw_j_denom_common + (backgroundLmProb * modelData.background_lm[word]);

                P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word] = P_zdw_j_num / P_zdw_j_denom;
            }
        }
    }

    // M-step
    // Document coverage
    for (size_t document = 0; document < previous.num_documents; document++) {
        double denom = 0;

        // Sum over all topics
        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                denom += smooth_ct * (1 - P_zdw_B[document * previous.vocab_size + word]) * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }
        }

        // For each topic/word pair
        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            double num = 0;
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * (1 - P_zdw_B[document * previous.vocab_size + word]) * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }

            current.document_coverage[topic * previous.num_documents + document] = num / denom;
        }
    }

    // Topic models
    for (size_t topic = 0; topic < previous.num_topics; topic++) {
        double denom = 0;

        // Sum over all words in the collection
        for (size_t document = 0; document < previous.num_documents; document++) {    
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                denom += smooth_ct * (1 - P_zdw_B[document * previous.vocab_size + word]) * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }
        }

        for (size_t word = 0; word < previous.vocab_size; word++) {
            double num = 0;
            for (size_t document = 0; document < previous.num_documents; document++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * (1 - P_zdw_B[document * previous.vocab_size + word]) * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }

            current.topic_models[topic * previous.vocab_size + word] = num / denom;
        }
    }
}

// Correct values
// Model Error: 3.38041
// Coverage Error: 132.969
// FIXME - Allow async kernels/ multiple command queues
// FIXME - Refactor code into separate files
void gpuUpdate(EMstep &current, EMstep &previous, ModelData &modelData, double backgroundLmProb,
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
    cl_mem *backgroundPriorArgsList[6] = {&modelData.background_lm_d, &denoms_common_d, &P_zdw_B_d, 
        (cl_mem*) &backgroundLmProb, (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size};
        
    ListWithSize<cl_mem*> backgroundPriorArgs(6, backgroundPriorArgsList);
    err = gpu::setKernelArgs(backgroundPriorKernel, backgroundPriorArgs); PRINT_ON_ERROR;

    err = gpu::launch2dKernelWithRoundup(backgroundPriorKernel,
        previous.num_documents, previous.vocab_size, 
        1, BLOCK_SIZE); PRINT_ON_ERROR;

    // P(Z_d,w | theta_j)
    cl_mem *topicPriorArgsList[9] = {&previous.document_coverage_d, &previous.topic_models_d, &denoms_common_d, 
        &modelData.background_lm_d, &P_zdw_j_d, 
        (cl_mem*) &backgroundLmProb, (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size, (cl_mem*) &previous.num_topics};
    
    ListWithSize<cl_mem*> topicPriorArgs(9, topicPriorArgsList);
    err = gpu::setKernelArgs(topicPriorKernel, topicPriorArgs); PRINT_ON_ERROR;

    err = gpu::launch3dKernelWithRoundup(topicPriorKernel,
        previous.num_topics, previous.num_documents, previous.vocab_size,
        1, 1, BLOCK_SIZE); PRINT_ON_ERROR;

    // M-step
    // Document coverage
    cl_mem *documentUpdateArgsList[7] = {&P_zdw_B_d, &P_zdw_j_d, &modelData.document_counts_d, &current.document_coverage_d,
        (cl_mem*) &previous.num_documents, (cl_mem*) &previous.vocab_size, (cl_mem*) &previous.num_topics};

    ListWithSize<cl_mem*> documentUpdateArgs(7, documentUpdateArgsList);
    err = gpu::setKernelArgs(documentUpdateKernel, documentUpdateArgs); PRINT_ON_ERROR;

    err = gpu::launch2dKernelWithRoundup(documentUpdateKernel,
        previous.num_topics, previous.num_documents,
        1, BLOCK_SIZE); PRINT_ON_ERROR;


    // Topic models
    cl_mem *topicUpdateArgsList[7] = {&P_zdw_B_d, &P_zdw_j_d, &modelData.document_counts_d, &current.topic_models_d,
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

bool isConverged(const EMstep &first, const EMstep &second) {
    // Check for convergence by subtracting the vectors and using an L1-norm over all values

    long double error_norm_coverage = 0;
    long double error_norm_model = 0;

    for (size_t i = 0; i < first.num_documents * first.num_topics; i++) {
        error_norm_coverage += abs(first.document_coverage[i] - second.document_coverage[i]);
    }

    for (size_t i = 0; i < first.num_topics * first.vocab_size; i++) {
        error_norm_model += abs(first.topic_models[i] - second.topic_models[i]);
    }

    cout << "Model error: " << error_norm_model << endl;
    cout << "Coverage error: " << error_norm_coverage << endl;
    cout << endl;

    return (error_norm_model < 2 && error_norm_coverage < 30);
}

bool isConvergedGpu(EMstep &first, EMstep &second, cl_mem coveragebuf_d, cl_mem modelbuf_d) {
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