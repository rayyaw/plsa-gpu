#pragma once

#include <CL/cl.h>

namespace linalg {
    cl_mem reduceTall(cl_mem &input, size_t num_major, size_t num_minor, size_t blockSize, cl_kernel kernel, cl_int *err);
}