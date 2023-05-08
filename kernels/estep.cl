// FIXME - This gives the wrong value
// Correct values
// Model Error: 3.38041
// Coverage Error: 132.969
__kernel void computeBackgroundPrior(
    __global const double *background_lm,
    __global const double *denoms_common,

    __global double *P_zdw_B,
    
    const double backgroundLmProb,

    const ulong num_documents,
    const ulong vocab_size) {

    const ulong document = get_global_id(0);
    const ulong word = get_global_id(1);

    if (document >= num_documents || word >= vocab_size) return;

    double topicLmProb = 1 - backgroundLmProb;

    double P_zdw_B_num = backgroundLmProb * background_lm[word];
    double P_zdw_B_denom = (backgroundLmProb * background_lm[word]) + 
        (denoms_common[document * vocab_size + word] * topicLmProb);

    P_zdw_B[document * vocab_size] = P_zdw_B_num / P_zdw_B_denom;
}

__kernel void computeTopicPrior(
    __global const double *document_coverage,
    __global const double *topic_models,
    __global const double *denoms_common,
    __global const double *background_lm,

    __global double *P_zdw_j,
    
    const double backgroundLmProb,

    const ulong num_documents,
    const ulong vocab_size,
    const ulong num_topics) {

    const ulong topic = get_global_id(0);
    const ulong document = get_global_id(1);
    const ulong word = get_global_id(2);

    if (topic >= num_topics || document >= num_documents || word >= vocab_size) return;

    double P_zdw_j_num = document_coverage[topic * num_documents + document] * topic_models[topic * vocab_size + word];
    double P_zdw_j_denom = denoms_common[document * vocab_size + word] + (backgroundLmProb * background_lm[word]);

    P_zdw_j[((topic * num_documents) + document) * vocab_size + word] = P_zdw_j_num / P_zdw_j_denom;

}