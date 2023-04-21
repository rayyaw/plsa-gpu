// Local headers
#include "emStep.h"
#include "modelData.h"

// Delete this line to compile without a GPU
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
    double *P_zdw_B, double *P_zdw_j) {
    
    cl_int err;

    // E-step
    // Topic-major, then document-major order
    double topicLmProb = 1 - backgroundLmProb;

    // We don't allocate our scratchpad memory since malloc() is slow and we can reuse across iterations
    // OR DO WE???
    double *doc_coverage_T = new double[previous.num_topics * previous.num_documents];
    double *denoms_common = new double[previous.num_documents * previous.vocab_size];

    // Transpose document coverage in preparation for sgemm
    // FIXME - GPU-accelerated transpose
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

            // Precomputing 1-x allows us to call sgemm directly in the next steps
            P_zdw_B[document * previous.vocab_size + word] = 1 - (P_zdw_B_num / P_zdw_B_denom);
        }
    }

    // P(Z_d,w | theta_j)
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
    for (size_t document = 0; document < previous.num_documents; document++) {
        double denom = 0;

        // Sum over all topics
        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                denom += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }
        }

        // For each topic/word pair
        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            double num = 0;
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
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
                denom += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }
        }

        for (size_t word = 0; word < previous.vocab_size; word++) {
            double num = 0;
            for (size_t document = 0; document < previous.num_documents; document++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }

            current.topic_models[topic * previous.vocab_size + word] = num / denom;
        }
    }

    delete[] doc_coverage_T;
    delete[] denoms_common;
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