#include "modelData.h"
#include "../gpu/gpu.h"

#include <CL/cl.h>
#include <string.h>

ModelData::ModelData(size_t document_count, size_t vocab_size) {
    this -> document_count = document_count;
    this -> vocab_size = vocab_size;

    document_counts = new size_t[document_count * vocab_size];
    background_lm = new double[vocab_size];
}

ModelData::ModelData(const ModelData &other) {
    document_count = other.document_count;
    vocab_size = other.vocab_size;

    document_counts = new size_t[document_count * vocab_size];
    background_lm = new double[vocab_size];

    memcpy(document_counts, other.document_counts, sizeof(size_t) * document_count * vocab_size);
    memcpy(background_lm, other.background_lm, sizeof(double) * vocab_size);
}

cl_int ModelData::mirrorGpu() {
    cl_int to_return = CL_SUCCESS;

    is_gpu_mirrored = true;

    document_counts_d = gpu::hostToDeviceCopy<size_t>(document_counts, document_count * vocab_size, &to_return);
    background_lm_d = gpu::hostToDeviceCopy<double>(background_lm, vocab_size, &to_return);

    return to_return;
}

ModelData::~ModelData() {
    if (is_gpu_mirrored) {
        clReleaseMemObject(document_counts_d);
        clReleaseMemObject(background_lm_d);
    }

    delete[] document_counts;
    delete[] background_lm;
}