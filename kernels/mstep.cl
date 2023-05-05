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
    if (topic >= num_topics || word >= vocab_size) {
        return;
    }

    double num = 0.0;
    for (ulong document = 0; document < num_documents; document++) {
        ulong idx = (document * vocab_size) + word;
        double smooth_ct = (double) document_counts[idx];
        num += smooth_ct * (1.0 - P_zdw_B[idx]) * P_zdw_j[(topic * num_documents + document) * vocab_size + word];
    }

    topic_models[topic * vocab_size + word] = num;
}

__kernel void reduceSum(__global const double *input, __global double *output, const ulong n) {
    // Determine the global and local work item IDs.
    const ulong gid = get_global_id(0);
    const ulong lid = get_local_id(0);
    const ulong lsize = get_local_size(0);

    // Allocate a shared memory array for the local sums.
    __local double local_sums[256];

    // Initialize the local sum for this work item to its input value.
    double local_sum = (gid < n) ? input[gid] : 0.0;

    // Perform the reduction operation across work items in the work group.
    for (ulong s = lsize / 2; s > 0; s >>= 1) {
        // Store this work item's local sum in shared memory.
        local_sums[lid] = local_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Update the local sum for this work item with the sum of its neighbor's local sum.
        if (lid < s) {
            local_sum += local_sums[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result of the reduction for this work group to global memory.
    if (lid == 0) {
        output[get_group_id(0)] = local_sum;
    }
}
