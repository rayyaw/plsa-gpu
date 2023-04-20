__kernel void batchReduce(__global const double *input,
                          __global double *output,
                          __global const long *data //offset_per_input, num_threads
                          ) {

    // Non-Coarsened version
    double accumulator = 0;
    long offset_per_input = data[0];

    if (get_global_id(0) < data[1]) {
        for (long i = offset_per_input * get_global_id(0); i < offset_per_input * (get_global_id(0) + 1); i++) {
            // FIXME - Transpose the input and use shared memory tiling
            accumulator += input[i];
        }

        output[get_global_id(0)] = accumulator;
    }

}