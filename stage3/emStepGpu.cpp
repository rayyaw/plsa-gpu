// Local headers
#include "emStep.h"
#include "modelData.h"

#include "../linalg/sgemm.h"
#include "../utils/listWithSize.h"

// C headers
#include <CL/cl.h>
#include <string.h>

// C++ headers
#include <cstdlib>
#include <iostream>

// Local using
using linalg::sgemm;

// Std using
using std::cerr;
using std::endl;

#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

// FIXME - Pass in a single pointer to use as a scratchpad
// and create a function to generate a scratchpad of the correct size
void gpuUpdate(EMstep &current, const EMstep &previous, const ModelData &modelData, double backgroundLmProb,
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
        size_t word = previous.vocab_size - 1;
        double P_zdw_B_num = backgroundLmProb * modelData.background_lm[word];
        double P_zdw_B_denom = (backgroundLmProb * modelData.background_lm[word]) +
            (denoms_common[document * previous.vocab_size + word] * topicLmProb);

        // Precomputed to go faster
        P_zdw_B[document * previous.vocab_size] = P_zdw_B_num / P_zdw_B_denom;
    }

    // P(Z_d,w | theta_j)
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            double P_zdw_j_denom = denoms_common[document * previous.vocab_size + word] + (backgroundLmProb * modelData.background_lm[word]);

            // For each topic/document pair
            for (size_t topic = 0; topic < previous.num_topics; topic++) {
                double P_zdw_j_num = previous.document_coverage[topic * previous.num_documents + document] * previous.topic_models[topic * previous.vocab_size + word];

                P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word] = P_zdw_j_num / P_zdw_j_denom;
            }
        }
    }

    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            P_zdw_B[document * previous.vocab_size + word] = 1 - P_zdw_B[document * previous.vocab_size + word];
        }
    }

    // M-step
    // Document coverage
    for (size_t document = 0; document < previous.num_documents; document++) {
        double tot = 0;

        // For each topic/word pair
        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            double num = 0;
            for (size_t word = 0; word < previous.vocab_size; word++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }

            current.document_coverage[topic * previous.num_documents + document] = num;
            tot += num;
        }

        for (size_t topic = 0; topic < previous.num_topics; topic++) {
            current.document_coverage[topic * previous.num_documents + document] /= tot;
        }
    }

    // Topic models
    // This should be the main priority for GPU acceleration
    // FIXME - Rephrase this as a sgemm problem
    for (size_t topic = 0; topic < previous.num_topics; topic++) {
        double tot = 0;

        for (size_t word = 0; word < previous.vocab_size; word++) {
            double num = 0;
            for (size_t document = 0; document < previous.num_documents; document++) {
                double smooth_ct = modelData.document_counts[(document * modelData.vocab_size) + word];
                num += smooth_ct * P_zdw_B[document * previous.vocab_size + word] * P_zdw_j[((topic * previous.num_documents) + document) * previous.vocab_size + word];
            }

            current.topic_models[topic * previous.vocab_size + word] = num;
            tot += num;
        }

        for (size_t word = 0; word < previous.vocab_size; word++) {
            current.topic_models[topic * previous.vocab_size + word] /= tot;
        }
    }


    delete[] doc_coverage_T;
    delete[] denoms_common;
}