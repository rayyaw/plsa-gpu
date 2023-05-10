// FIXME - Suboptimal version
__kernel void reduceAlongMajorAxisWide(
    __global const double *arr, 
    __global double *outputs, 
    const ulong num_major,
    const ulong num_minor) {

    const ulong majorAxis = get_global_id(0);

    if (majorAxis >= num_major) return;

    double sum = 0;
    for (ulong minorAxis = 0; minorAxis < num_minor; minorAxis++) {
        sum += arr[majorAxis * num_minor + minorAxis];
    }

    outputs[majorAxis] = sum;
}

// Modified from a ChatGPT version
__kernel void reduceAlongMajorAxisTall(
    __global const double* input, 
    __global double* output, 
    const ulong num_major,
    const ulong num_minor) {
    // Allocate local memory for each workgroup
    __local double localData[256];
    
    const ulong majorAxis = get_global_id(0);

    // Get the global ID and local ID
    const ulong globalID = get_global_id(1);
    const ulong localID = get_local_id(1);
    const ulong groupID = get_group_id(1);
    const ulong groupSize = get_local_size(1);
    const ulong numGroups = get_global_size(1) / get_local_size(1);
 
    // Load data from global memory to local memory
    if (globalID < num_minor) {
        localData[localID] = input[majorAxis * num_minor + globalID];
    } else {
        localData[localID] = 0.0f; // Pad with identity element for elements beyond array size
    }

    // Perform reduction within each workgroup
    for (ulong stride = groupSize / 2; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE); // Synchronize within workgroup

        if (localID < stride) {
            localData[localID] += localData[localID + stride];
        }
    }

    // Store the final result in global memory
    // FIXME - Don't use global atomics as this is slow!
    if (localID == 0) {
        // FIXME - we need atomics
        output[majorAxis * numGroups + groupID] = localData[0];
    }
}

__kernel void normalizeAlongMajorAxis(
    __global double *values,
    __global const double *sums,
    const ulong num_major,
    const ulong num_minor) {

    const ulong majorAxis = get_global_id(0);
    const ulong minorAxis = get_global_id(1);
    
    if (majorAxis >= num_major || minorAxis >= num_minor) return; 

    values[majorAxis * num_minor + minorAxis] /= sums[majorAxis];
}