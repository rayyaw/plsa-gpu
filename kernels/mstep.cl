// FIXME - Parallelize over words as well
__kernel void computeTopicUpdate(
    __global const double* P_zdw_B,
    __global const double* P_zdw_j,
    __global const ulong* document_counts,
    __global double* topic_models,
    const ulong num_documents,
    const ulong vocab_size,
    const ulong num_topics)
{
    const ulong gid = get_global_id(0);
    if (gid >= num_topics) {
        return;
    }

    double denom = 0.0;

    // Sum over all words in the collection
    for (ulong document = 0; document < num_documents; document++) {
        for (ulong word = 0; word < vocab_size; word++) {
            ulong idx = document * vocab_size + word;
            double smooth_ct = (double) document_counts[idx];
            denom += smooth_ct * (1.0 - P_zdw_B[idx]) * P_zdw_j[(gid * num_documents + document) * vocab_size + word];
        }
    }

    for (ulong word = 0; word < vocab_size; word++) {
        double num = 0.0;
        for (ulong document = 0; document < num_documents; document++) {
            ulong idx = document * vocab_size + word;
            double smooth_ct = (double) document_counts[idx];
            num += smooth_ct * (1.0 - P_zdw_B[idx]) * P_zdw_j[(gid * num_documents + document) * vocab_size + word];
        }
        // FIXME - This access hangs (illegal).
        // it's a read issue (even in num)
        // literally any input access hangs (even index 0)
        // but the memory is copied over correctly???
        topic_models[gid * vocab_size + word] = num / denom;
    }
}
