__kernel void vectorAbsDiff(__global const double* a, __global const double* b, __global double* c, const ulong vectorSize)
{
    const ulong globalID = get_global_id(0);

    if (globalID < vectorSize) {
        double diff = fabs(a[globalID] - b[globalID]);
        c[globalID] = diff;
    }
}
