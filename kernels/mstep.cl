// FIXME - Optimize these so they run faster (if necessary)
__kernel void computeTopicUpdate(
    __global const double* P_zdw_B,
    __global const double* P_zdw_j,
    __global const ulong* document_counts,
    __global double* topic_models,
    const ulong num_documents,
    const ulong vocab_size,
    const ulong num_topics) {

    const ulong topic = get_global_id(0);
    const ulong word = get_global_id(1);
    if (topic >= num_topics || word >= vocab_size) return;

    double num = 0.0;
    for (ulong document = 0; document < num_documents; document++) {
        ulong idx = (document * vocab_size) + word;
        double smooth_ct = (double) document_counts[idx];
        num += smooth_ct * (1.0 - P_zdw_B[idx]) * P_zdw_j[(topic * num_documents + document) * vocab_size + word];
    }

    topic_models[topic * vocab_size + word] = num;
}

__kernel void computeDocumentUpdate(
    __global const double* P_zdw_B,
    __global const double* P_zdw_j,
    __global const ulong* document_counts,
    __global double* document_coverage,
    const ulong num_documents,
    const ulong vocab_size,
    const ulong num_topics) {

    const ulong topic = get_global_id(0);
    const ulong document = get_global_id(1);

    if (document >= num_documents || topic >= num_topics) return;

    // For each topic/word pair
    double num = 0;
    for (ulong word = 0; word < vocab_size; word++) {
        ulong idx = document * vocab_size + word;
        double smooth_ct = (double) document_counts[idx];
        num += smooth_ct * (1 - P_zdw_B[idx]) * P_zdw_j[(topic * num_documents + document) * vocab_size + word];
    }

    document_coverage[document * num_topics + topic] = num;
}
