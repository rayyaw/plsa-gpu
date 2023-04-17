__kernel void reduce_word_encodings(__global double *output_lm, 
    const unsigned long long total_word_count, const unsigned long long num_documents, const unsigned long long vocab_size,
    __global const size_t *counts) {
    // Coarsened version
    // FIXME - Access coalescing
    size_t word_count = 0;
    int per_thread = ceil((vocab_size * 1.0f) / get_global_size(0));

    for (unsigned long long current = get_global_id(0) * per_thread; current < (get_global_id(0) + 1) * per_thread; current++) {
        if (current >= vocab_size) return;

        for (unsigned long long i = 0; i < num_documents; i++) {
            word_count += counts[i * vocab_size + current];
        }

        double output = (word_count * 1.0f) / total_word_count;

        // FIXME - This line is LLVM unsupported libcall? Why?
        // Even output_lm[0] = 0 breaks (?)
        // Writing to output_lm at ANY position is the issue
        //output = output_lm[current];
        output_lm[current] = output;
    }

}