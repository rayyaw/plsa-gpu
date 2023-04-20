#include "modelData.h"

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

ModelData::~ModelData() {
    delete[] document_counts;
    delete[] background_lm;
}