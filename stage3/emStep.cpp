// Local headers
#include "emStep.h"
#include "modelData.h"

#include "../gpu/gpu.h"
#include "../linalg/sgemm.h"

// C headers
#include <CL/cl.h>
#include <string.h>

// C++ headers
#include <cstdlib>
#include <iostream>

#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

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

// You should comment this out when compiling without a GPU
// FIXME - Do all normalizations on the GPU
// FIXME - All computations on the GPU to avoid HtoD and DtoH copies
// FIXME - memleak when using ListWithSize for grid and block dims (this is small, so only a minor issue)

// Correct values
// Model Error: 3.38041
// Coverage Error: 132.969
void gpuUpdate(EMstep &current, const EMstep &previous, ModelData &modelData, double backgroundLmProb,
    double *scratchpad, cl_mem &P_zdw_B_d, cl_mem &P_zdw_j_d, cl_mem &denoms_common_d) {

    // Scratchpad offsets
    double *denoms_common = scratchpad;
    
    // Overhead - GPU setup
    cl_int err = CL_SUCCESS;
    size_t blockSize = 256;
    
    // Initialize kernels
    // These are lazy-loaded (if already loaded, just reuse) so not much overhead to do in every iteration
    cl_kernel backgroundPriorKernel = gpu::compileKernelFromFile("kernels/estep.cl", "computeBackgroundPrior", &err); PRINT_ON_ERROR;
    cl_kernel topicPriorKernel = gpu::compileKernelFromFile("kernels/estep.cl", "computeTopicPrior", &err); PRINT_ON_ERROR;

    cl_kernel documentUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeDocumentUpdate", &err); PRINT_ON_ERROR;
    cl_kernel topicUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeTopicUpdate", &err); PRINT_ON_ERROR;

    // Copy data to the GPU
    cl_mem document_counts_d = gpu::hostToDeviceCopy<cl_ulong>((cl_ulong*) modelData.document_counts, previous.num_documents * previous.vocab_size, &err); PRINT_ON_ERROR;
    cl_mem background_lm_d = gpu::hostToDeviceCopy<double>(modelData.background_lm, previous.vocab_size, &err); PRINT_ON_ERROR;

    cl_mem prev_document_coverage_d = gpu::hostToDeviceCopy<double>(previous.document_coverage, previous.num_topics * previous.num_documents, &err); PRINT_ON_ERROR;
    cl_mem prev_topic_models_d = gpu::hostToDeviceCopy<double>(previous.topic_models, previous.num_topics * previous.vocab_size, &err); PRINT_ON_ERROR;

    // E-step

    // Document coverage is passed in as transposed
    err = linalg::sgemmDevice(prev_document_coverage_d, prev_topic_models_d, denoms_common_d, previous.num_documents, previous.vocab_size, previous.num_topics); PRINT_ON_ERROR;

    // P(Z_d,w | B)
    err = clSetKernelArg(backgroundPriorKernel, 0, sizeof(background_lm_d), (void*) &background_lm_d); PRINT_ON_ERROR;
    err = clSetKernelArg(backgroundPriorKernel, 1, sizeof(denoms_common_d), (void*) &denoms_common_d); PRINT_ON_ERROR;
    err = clSetKernelArg(backgroundPriorKernel, 2, sizeof(P_zdw_B_d), (void*) &P_zdw_B_d); PRINT_ON_ERROR;
    err = clSetKernelArg(backgroundPriorKernel, 3, sizeof(backgroundLmProb), (void*) &backgroundLmProb); PRINT_ON_ERROR;
    err = clSetKernelArg(backgroundPriorKernel, 4, sizeof(previous.num_documents), (void*) &previous.num_documents); PRINT_ON_ERROR;
    err = clSetKernelArg(backgroundPriorKernel, 5, sizeof(previous.vocab_size), (void*) &previous.vocab_size); PRINT_ON_ERROR;

    ListWithSize<size_t> gridDimBackgroundPrior = gpu::makeDim2(previous.num_documents, ceil((previous.vocab_size * 1.0) / blockSize) * blockSize);
    ListWithSize<size_t> blockDimBackgroundPrior = gpu::makeDim2(1, blockSize);

    err = gpu::launchKernel(backgroundPriorKernel, gridDimBackgroundPrior, blockDimBackgroundPrior);

    // P(Z_d,w | theta_j)
    // FIXME - Create a function so I don't need to call clSetKernelArg a million times
    err = clSetKernelArg(topicPriorKernel, 0, sizeof(prev_document_coverage_d), (void*) &prev_document_coverage_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 1, sizeof(prev_topic_models_d), (void*) &prev_topic_models_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 2, sizeof(denoms_common_d), (void*) &denoms_common_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 3, sizeof(background_lm_d), (void*) &background_lm_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 4, sizeof(P_zdw_j_d), (void*) &P_zdw_j_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 5, sizeof(backgroundLmProb), (void*) &backgroundLmProb); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 6, sizeof(previous.num_documents), (void*) &previous.num_documents); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 7, sizeof(previous.vocab_size), (void*) &previous.vocab_size); PRINT_ON_ERROR;
    err = clSetKernelArg(topicPriorKernel, 8, sizeof(previous.num_topics), (void*) &previous.num_topics); PRINT_ON_ERROR;

    ListWithSize<size_t> gridDimTopicPrior = gpu::makeDim3(previous.num_topics, previous.num_documents, ceil((previous.vocab_size * 1.0) / blockSize) * blockSize);
    ListWithSize<size_t> blockDimTopicPrior = gpu::makeDim3(1, 1, blockSize);

    err = gpu::launchKernel(topicPriorKernel, gridDimTopicPrior, blockDimTopicPrior); PRINT_ON_ERROR;

    // M-step

    // Copy all data to the GPU
    cl_mem topic_models_d = gpu::deviceOutputAllocate(sizeof(double) * previous.num_topics * previous.vocab_size, &err); PRINT_ON_ERROR;
    cl_mem document_coverage_d = gpu::deviceOutputAllocate(sizeof(double) * previous.num_topics * previous.num_documents, &err); PRINT_ON_ERROR;

    // Set arguments
    err = clSetKernelArg(documentUpdateKernel, 0, sizeof(P_zdw_B_d), (void*) &P_zdw_B_d); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 1, sizeof(P_zdw_j_d), (void*) &P_zdw_j_d); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 2, sizeof(document_counts_d), (void*) &document_counts_d); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 3, sizeof(document_coverage_d), (void*) &document_coverage_d); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 4, sizeof(previous.num_documents), (void*) &previous.num_documents); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 5, sizeof(previous.vocab_size), (void*) &previous.vocab_size); PRINT_ON_ERROR;
    err = clSetKernelArg(documentUpdateKernel, 6, sizeof(previous.num_topics), (void*) &previous.num_topics); PRINT_ON_ERROR;

    err = clSetKernelArg(topicUpdateKernel, 0, sizeof(P_zdw_B_d), (void*) &P_zdw_B_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 1, sizeof(P_zdw_j_d), (void*) &P_zdw_j_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 2, sizeof(document_counts_d), (void*) &document_counts_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 3, sizeof(topic_models_d), (void*) &topic_models_d); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 4, sizeof(previous.num_documents), (void*) &previous.num_documents); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 5, sizeof(previous.vocab_size), (void*) &previous.vocab_size); PRINT_ON_ERROR;
    err = clSetKernelArg(topicUpdateKernel, 6, sizeof(previous.num_topics), (void*) &previous.num_topics); PRINT_ON_ERROR;

    // Launch kernel - document counts
    ListWithSize<size_t> gridDimDocument = gpu::makeDim2(previous.num_topics, ceil((previous.num_documents * 1.0) / blockSize) * blockSize);
    ListWithSize<size_t> blockDimDocument = gpu::makeDim2(1, blockSize);

    err = gpu::launchKernel(documentUpdateKernel, gridDimDocument, blockDimDocument); PRINT_ON_ERROR;

    err = gpu::copyDeviceToHost<double>(document_coverage_d, current.document_coverage, previous.num_topics * previous.num_documents); PRINT_ON_ERROR;

    // Launch kernel - topic updates
    ListWithSize<size_t> gridDimTopic = gpu::makeDim2(previous.num_topics, ceil((previous.vocab_size * 1.0) / blockSize) * blockSize);
    ListWithSize<size_t> blockDimTopic = gpu::makeDim2(1, blockSize);

    err = gpu::launchKernel(topicUpdateKernel, gridDimTopic, blockDimTopic); PRINT_ON_ERROR;

    err = gpu::copyDeviceToHost<double>(topic_models_d, current.topic_models, previous.num_topics * previous.vocab_size); PRINT_ON_ERROR;

    // Cleanup
    clReleaseMemObject(topic_models_d); PRINT_ON_ERROR;
    clReleaseMemObject(document_counts_d); PRINT_ON_ERROR;
    clReleaseMemObject(prev_document_coverage_d); PRINT_ON_ERROR;
    clReleaseMemObject(prev_topic_models_d); PRINT_ON_ERROR;
    clReleaseMemObject(background_lm_d); PRINT_ON_ERROR;

    // Normalize the outputs
    // FIXME - Perform this step on the GPU instead
    for (size_t document = 0; document < previous.num_documents; document++) {
        double denom = 0;

        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            denom += current.document_coverage[document * previous.num_topics + topic];
        }

        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            current.document_coverage[document * previous.num_topics + topic] /= denom;
        }
    }

    for (size_t i = 0; i < previous.num_topics; i++) {
        double denom = 0;

        for (size_t j = 0; j < previous.vocab_size; j++) {
            denom += current.topic_models[i * previous.vocab_size + j];
        }

        for (size_t j = 0; j < previous.vocab_size; j++) {
            current.topic_models[i * previous.vocab_size + j] /= denom;
        }
    }
}

bool isConverged(const EMstep &first, const EMstep &second) {
    // Check for convergence by subtracting the vectors and using an L1-norm over all values
    // FIXME - is L-inf norm faster on GPU despite control divergence?

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

    // FIXME - Tweak these values
    return (error_norm_model < 1 && error_norm_coverage < 2);
}