__kernel void reduce_word_encodings(__global double *output_lm, 
                                    __global const size_t *counts) {
    // Non-Coarsened version
    size_t word_count = 0;
    int per_thread = ceil((vocab_size * 1.0f) / get_global_size(0));

    for (unsigned long long i = 0; i < num_documents; i++) {
        word_count += counts[i * vocab_size + get_global_id(0)];
    }
    
    double output = (word_count * 1.0f) / total_word_count;
    output_lm[current] = output;

}