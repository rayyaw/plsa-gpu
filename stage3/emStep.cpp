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

#define SMOOTHING_FACTOR 0.99
#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

// Local ussing
using linalg::sgemm;

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
    // FIXME - Precompute 1 - x to go faster
    // Use J-M smoothing of the counts to get more accurate
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
// FIXME - Pass in a single pointer to use as a scratchpad
void gpuUpdate(EMstep &current, const EMstep &previous, ModelData &modelData, double backgroundLmProb,
    double *P_zdw_B, double *P_zdw_j, double *doc_coverage_T, double *denoms_common) {
    
    cl_int err;

    // E-step
    // Topic-major, then document-major order
    double topicLmProb = 1 - backgroundLmProb;

    // Transpose document coverage in preparation for sgemm
    for (int i = 0; i < previous.num_documents; i++) {
        for (int j = 0; j < previous.num_topics; j++) {
            doc_coverage_T[i * previous.num_topics + j] = previous.document_coverage[j * previous.num_documents + i];
        }
    }

    err = linalg::sgemm(doc_coverage_T, previous.topic_models, denoms_common, previous.num_documents, previous.vocab_size, previous.num_topics);
    PRINT_ON_ERROR;

    // P(Z_d,w | B)
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            double P_zdw_B_num = backgroundLmProb * modelData.background_lm[word];
            double P_zdw_B_denom = (backgroundLmProb * modelData.background_lm[word]) +
                (denoms_common[document * previous.vocab_size + word] * topicLmProb);

            P_zdw_B[document * previous.vocab_size] = P_zdw_B_num / P_zdw_B_denom;
        }
    }

    // P(Z_d,w | theta_j)
    // FIXME - accelerate this
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {

            // For each topic/document pair
            for (size_t topic = 0; topic < previous.num_topics; topic++) {
                double P_zdw_j_num = previous.document_coverage[topic * previous.num_documents + document] * previous.topic_models[topic * previous.vocab_size + word];
                double P_zdw_j_denom = denoms_common[document * previous.vocab_size + word] + (backgroundLmProb * modelData.background_lm[word]);

                P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word] = P_zdw_j_num / P_zdw_j_denom;
            }
        }
    }

    // M-step
    // Document coverage
    // FIXME - accelerate this
    // The correct output gives coverage error of 132.969 and model error of 3.38041
    /*for (size_t document = 0; document < previous.num_documents; document++) {
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
    }*/

    // Some overhead required - compile the GPU code and copy over necessary variables
    cl_kernel documentUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeDocumentUpdate", &err); PRINT_ON_ERROR;
    cl_kernel topicUpdateKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "computeTopicUpdate", &err); PRINT_ON_ERROR;
    //cl_kernel reductionKernel = gpu::compileKernelFromFile("kernels/mstep.cl", "reduceSum", &err); PRINT_ON_ERROR;
    
    // Copy all data to the GPU
    cl_mem document_counts_d = gpu::hostToDeviceCopy<cl_ulong>((cl_ulong*) modelData.document_counts, previous.num_documents * previous.vocab_size, &err); PRINT_ON_ERROR;
    cl_mem P_zdw_B_d = gpu::hostToDeviceCopy<double>(P_zdw_B, previous.num_documents * previous.vocab_size, &err); PRINT_ON_ERROR;
    cl_mem P_zdw_j_d = gpu::hostToDeviceCopy<double>(P_zdw_j, previous.num_topics * previous.num_documents * previous.vocab_size, &err); PRINT_ON_ERROR;
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
    
    size_t blockSize = 256;

    // Launch kernel - document counts
    utils::ListWithSize<size_t> gridDimDocument = utils::ListWithSize<size_t>();
    gridDimDocument.num_items = 2;
    gridDimDocument.items = new size_t[2];
    gridDimDocument.items[0] = previous.num_topics;
    gridDimDocument.items[1] = ceil((previous.num_documents * 1.0) / blockSize) * blockSize;

    utils::ListWithSize<size_t> blockDimDocument = utils::ListWithSize<size_t>();
    blockDimDocument.num_items = 2;
    blockDimDocument.items = new size_t[2];
    blockDimDocument.items[0] = 1;
    blockDimDocument.items[1] = blockSize;

    // FIXME - this gives incorrect output of all 0s
    gpu::launchKernel(documentUpdateKernel, gridDimDocument, blockDimDocument);

    err = gpu::copyDeviceToHost<double>(document_coverage_d, current.document_coverage, previous.num_topics * previous.num_documents); PRINT_ON_ERROR;

    utils::ListWithSize<size_t> gridDimTopic = utils::ListWithSize<size_t>();
    gridDimTopic.num_items = 2;
    gridDimTopic.items = new size_t[2];
    gridDimTopic.items[0] = previous.num_topics;
    gridDimTopic.items[1] = ceil((previous.vocab_size * 1.0) / blockSize) * blockSize;

    utils::ListWithSize<size_t> blockDimTopic = utils::ListWithSize<size_t>();
    blockDimTopic.num_items = 2;
    blockDimTopic.items = new size_t[2];
    blockDimTopic.items[0] = 1;
    blockDimTopic.items[1] = blockSize;

    gpu::launchKernel(topicUpdateKernel, gridDimTopic, blockDimTopic);

    err = gpu::copyDeviceToHost<double>(topic_models_d, current.topic_models, previous.num_topics * previous.vocab_size); PRINT_ON_ERROR;

    clReleaseMemObject(topic_models_d);
    clReleaseMemObject(P_zdw_B_d);
    clReleaseMemObject(P_zdw_j_d);
    clReleaseMemObject(document_counts_d);
    clReleaseMemObject(topic_models_d);

    // Normalize the outputs
    // FIXME - Perform this step on the GPU instead
    for (size_t document = 0; document < previous.num_documents; document++) {
        double denom = 0;

        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            denom += current.document_coverage[topic * previous.num_documents + document];
        }

        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            current.document_coverage[topic * previous.num_documents + document] /= denom;
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