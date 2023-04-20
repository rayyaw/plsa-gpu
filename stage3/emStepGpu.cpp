// C headers
#include <CL/cl.h>

// C++ headers
#include <iostream>

// Local headers
#include "emStep.h"
#include "../gpu/gpu.h"
#include "../io/io.h"

using std::cerr;
using std::endl;

#define RETURN_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1); }

extern map<const char*, cl_kernel> *available_kernels;

// FIXME - This cannot handle matrices of the required size
cl_int sgemm(double *A, double *B, double *C, unsigned int M, unsigned int N, unsigned int K) {
    const char *sgemm_name = "sgemm";

    cl_int err;

    if (!gpu::kernelExists(sgemm_name)) {
        const char *kernel_string = io::readKernel("kernels/sgemm.cl");
        gpu::compileKernelIfNotExists(kernel_string, sgemm_name, &err); RETURN_ON_ERROR;
    } 

    cl_kernel sgemm_kernel = (*available_kernels)[sgemm_name];

    utils::ListWithSize<double> A_info;
    A_info.items = A;
    A_info.num_items = M * K;

    utils::ListWithSize<double> B_info;
    B_info.items = B;
    B_info.num_items = K * N;

    utils::ListWithSize<double> C_info;
    C_info.items = C;
    C_info.num_items = M * N;

    // Allocate GPU memory
    cl_mem A_d = gpu::hostToDeviceCopy<double>(A_info, &err); RETURN_ON_ERROR;
    cl_mem B_d = gpu::hostToDeviceCopy<double>(B_info, &err); RETURN_ON_ERROR;
    cl_mem C_d = gpu::deviceOutputAllocate(M * K * sizeof(double), &err); RETURN_ON_ERROR;

    // the max in OpenCL is 256
    size_t blockSize = 16;

    utils::ListWithSize<size_t> gridDim = utils::ListWithSize<size_t>();
    gridDim.num_items = 2;
    gridDim.items = new size_t[2];
    gridDim.items[0] = ceil((M * 1.0) / blockSize) * blockSize;
    gridDim.items[1] = ceil((N * 1.0) / blockSize) * blockSize;

    utils::ListWithSize<size_t> blockDim = utils::ListWithSize<size_t>();
    blockDim.num_items = 2;
    blockDim.items = new size_t[2];
    blockDim.items[0] = blockSize;
    blockDim.items[1] = blockSize;

    // Set kernel arguments
    err = clSetKernelArg(sgemm_kernel, 0, sizeof(A_d), (void*) &A_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 1, sizeof(B_d), (void*) &B_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 2, sizeof(C_d), (void*) &C_d); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 3, sizeof(M), (void*) &M); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 4, sizeof(N), (void*) &N); RETURN_ON_ERROR;
    err = clSetKernelArg(sgemm_kernel, 5, sizeof(K), (void*) &K); RETURN_ON_ERROR;

    // Launch the kernel
    // FIXME - issues here (this bluescreens my PC)
    err = gpu::launchKernel(sgemm_kernel, gridDim, blockDim); RETURN_ON_ERROR;

    // Copy the output back
    err = gpu::copyDeviceToHost<double>(C_d, C_info); RETURN_ON_ERROR;
    
    // Cleanup
    clReleaseMemObject(A_d);
    clReleaseMemObject(B_d);
    clReleaseMemObject(C_d);

    return err;
}

// FIXME
void gpuUpdate(EMstep &current, const EMstep &previous, const ModelData &modelData, double backgroundLmProb) {
    // E-step
    // Topic-major, then document-major order
    double topicLmProb = 1 - backgroundLmProb;

    double *P_zdw_B = new double[modelData.document_count * modelData.vocab_size];
    double *P_zdw_j = new double[modelData.document_count * modelData.vocab_size * previous.num_topics];

    // Defining OpenCL stuff
    cl_int err;

    // First reduction kernel
    double *denoms_p_z_dw_B = new double[previous.num_documents * previous.vocab_size];

    // O[doc][word], I1 = doccov[doc][topic] * models[topic][word]
    sgemm(previous.document_coverage, previous.topic_models, denoms_p_z_dw_B, previous.num_topics, previous.vocab_size, previous.num_topics);

    cerr << "sgemm call successful" << endl;
    // P(Z_d,w | B)
    // FIXME - Precompute 1 - x to go faster
    for (size_t document = 0; document < previous.num_documents; document++) {
        for (size_t word = 0; word < previous.vocab_size; word++) {
            double P_zdw_B_num = backgroundLmProb * modelData.background_lm[word];
            double P_zdw_B_denom = backgroundLmProb * modelData.background_lm[word];

            double sum_of_all_topics = 0;

            // FIXME - Rephrase this as matrix multiplication
            for (size_t i = 0; i < previous.num_topics; i++) {
                sum_of_all_topics += previous.document_coverage[i * previous.num_documents + document] 
                                    * previous.topic_models[i * previous.vocab_size + word];
            }

            sum_of_all_topics *= topicLmProb;

            P_zdw_B_denom += sum_of_all_topics;

            P_zdw_B[document * previous.vocab_size] = backgroundLmProb * modelData.background_lm[word];
            P_zdw_B[document * previous.vocab_size] /= P_zdw_B_denom;
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

    delete[] denoms_p_z_dw_B;

    delete[] P_zdw_B;
    delete[] P_zdw_j;
}