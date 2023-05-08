/**
 * @file stage3_cpu_processing.cpp
 * @author Riley
 * @brief Perform stage 3 of the PLSA pipeline. Here, we use the EM algorithm to derive our model.
 * @version 1.0
 * @date 2023-04-12
 * 
 * @copyright Copyright (c) 2023. Licensed under CC-BY-SA.
 * 
 */

// C headers
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// C++ headers
#include <cstddef>
#include <iostream>

// Local headers
#include "gpu/gpu.h"
#include "stage3/emStep.h"
#include "stage3/modelData.h"

using std::cerr;
using std::cout;
using std::endl;

#define MAXITER 2
#define PRINT_ON_ERROR if (err != CL_SUCCESS) { cerr << "CL ERROR: " << err << endl; exit(1);}

unsigned long long currentTimeMillis() {
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (unsigned long long)(tv.tv_sec) * 1000 + (unsigned long long)(tv.tv_usec) / 1000;
}

ModelData loadModelFromFile() {
    // Load the counts and background LM from file.
    FILE *count_file = fopen("model/counts.bin", "rb");

    if (!count_file) {
        cerr << "Unable to open counts file. Please check your data." << endl;
        exit(-1);
    }

    // The first 2 entries
    size_t *count_data = new size_t[2];
    fread(count_data, sizeof(size_t), 2, count_file);

    ModelData model = ModelData(count_data[0], count_data[1]);

    delete[] count_data;

    cout << "Detected " << model.document_count << " documents and " << model.vocab_size << " words. Loading background LM..." << endl;

    fread(model.document_counts, sizeof(size_t), model.vocab_size * model.document_count, count_file);
    fclose(count_file);

    FILE *background_lm_file = fopen("model/bg.bin", "rb");

    if (!background_lm_file) {
        cerr << "Unable to open background LM file. Please check your data." << endl;
        exit(-1);
    }

    fread(model.background_lm, sizeof(size_t), model.vocab_size, background_lm_file);
    fclose(background_lm_file);

    cout << "File loading completed. Proceeding to computation phase." << endl;

    return model;
}

// Save the EM results to files. Generated by ChatGPT.
void saveEmToFile(const EMstep &data) {
    // Open files
    FILE *doc_file = fopen("model/document_coverage.bin", "wb+");
    FILE *topic_file = fopen("model/topic_models.bin", "wb+");

    // Write num_topics, num_documents, and vocab_size to topic file
    fwrite(&data.num_topics, sizeof(size_t), 1, topic_file);
    fwrite(&data.num_documents, sizeof(size_t), 1, topic_file);
    fwrite(&data.vocab_size, sizeof(size_t), 1, topic_file);

    // Write document_coverage to file
    fwrite(data.document_coverage, sizeof(double), data.num_topics * data.num_documents, doc_file);

    // Write topic_models to file
    fwrite(data.topic_models, sizeof(double), data.num_topics * data.vocab_size, topic_file);

    // Close files
    fclose(doc_file);
    fclose(topic_file);

    cout << "Saved models to files. Stage 3 completed." << endl;
}

// Transpose the document coverage matrix.
// This allows us to run matrix multiplication
void transposeDocumentCoverage(EMstep &model) {
    double *doc_coverage_T = new double[model.num_topics * model.num_documents];

    for (int i = 0; i < model.num_documents; i++) {
        for (int j = 0; j < model.num_topics; j++) {
            doc_coverage_T[i * model.num_topics + j] = model.document_coverage[j * model.num_documents + i];
        }
    }

    delete[] model.document_coverage;
    model.document_coverage = doc_coverage_T;
}


EMstep runEm(ModelData &model, size_t num_topics, double prob_of_bg) {
    // Double buffering!
    EMstep first = EMstep(num_topics, model.document_count, model.vocab_size);
    EMstep second = EMstep(num_topics, model.document_count, model.vocab_size);

    EMstep to_return = EMstep(num_topics, model.document_count, model.vocab_size);
    
    first.genrandom();

    // Required to run sgemm
    transposeDocumentCoverage(first);

    bool update_first = false;

    double *scratchpad = new double[first.num_documents * first.vocab_size];
    
    // Setup GPU scratchpad memory
    cl_int err = CL_SUCCESS;
    cl_mem P_zdw_j_d = gpu::deviceIntermediateAllocate(sizeof(double) * first.num_documents * first.num_topics * first.vocab_size, &err); PRINT_ON_ERROR;
    cl_mem P_zdw_B_d = gpu::deviceIntermediateAllocate(sizeof(double) * first.num_documents * first.vocab_size, &err); PRINT_ON_ERROR;

    cl_mem denoms_common_d = gpu::deviceIntermediateAllocate(sizeof(double) * first.num_documents * first.vocab_size, &err); PRINT_ON_ERROR;


    for (size_t i = 0; i < MAXITER; i++) {
        // This takes 42s per iteration, assuming 500 books (on the CPU)
        // On the GPU this takes 34s per iteration, assuming 400 books
        unsigned long long start_t = currentTimeMillis();
        if (update_first) {
            gpuUpdate(first, second, model, prob_of_bg, scratchpad, P_zdw_B_d, P_zdw_j_d, denoms_common_d);
        } else {
            gpuUpdate(second, first, model, prob_of_bg, scratchpad, P_zdw_B_d, P_zdw_j_d, denoms_common_d);
        }
        unsigned long long end_t = currentTimeMillis();

        cout << "Iteration number: " << i << endl;
        cout << "Time taken: " << end_t - start_t << "ms" << endl;

        if (isConverged(first, second)) {
            if (update_first) {
                transposeDocumentCoverage(first);
                to_return = first;
                break;
            } else {
                transposeDocumentCoverage(second);
                to_return = second;
                break;
            }
        }

        update_first = !update_first;
    }

    delete[] scratchpad;
    clReleaseMemObject(denoms_common_d);
    clReleaseMemObject(P_zdw_B_d);
    clReleaseMemObject(P_zdw_j_d);

    cout << "Completed EM phase. Saving results to file..." << endl;
    return to_return;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " [num_topics] [background probability]" << endl;
        exit(-1);
    }

    gpu::initializeGpuData(1);
    
    ModelData model = loadModelFromFile();
    cl_int err = model.mirrorGpu(); PRINT_ON_ERROR;

    unsigned int num_topics = atoi(argv[1]);
    float bg_prob = atof(argv[2]);

    EMstep output = runEm(model, num_topics, bg_prob);

    // Save output to file
    saveEmToFile(output);

    gpu::destroyGpuData();
}