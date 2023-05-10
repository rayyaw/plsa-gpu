// FIXME - Suboptimal version
__kernel void reduceAlongMajorAxis(
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