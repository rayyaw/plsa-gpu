// FIXME - Use a reduction to get more speedup
// This is still faster without it, as it eliminates several HtoD and DtoH copies
// by allowing the entire EM step calculation to run on the GPU

__kernel void normalizeDocumentCoverage(
    __global double* document_coverage,
    const ulong num_topics,
    const ulong num_documents) {
    
    const ulong document = get_global_id(0);

    if (document >= num_documents) return;

    double denom = 0.0;

    for (ulong topic = 0; topic < num_topics; topic++) {
        denom += document_coverage[document * num_topics + topic];
    }

    for (ulong topic = 0; topic < num_topics; topic++) {
        document_coverage[document * num_topics + topic] /= denom;
    }
}

__kernel void normalizeTopicModels(
    __global double *topic_models,
    const ulong num_topics, 
    const ulong vocab_size) {

    const ulong topic = get_global_id(0);

    if (topic >= num_topics) return;

    double denom = 0.0;

    for (ulong word = 0; word < vocab_size; word++) {
        denom += topic_models[topic * vocab_size + word];
    }

    for (ulong word = 0; word < vocab_size; word++) {
        topic_models[topic * vocab_size + word] /= denom;
    }
}