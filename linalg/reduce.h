#pragma once

#include <CL/cl.h>

namespace linalg {
    cl_int reduceWide(cl_mem &input, cl_mem &output, size_t num_major, size_t num_minor, size_t block_size);
    cl_mem reduceTall(cl_mem &input, size_t num_major, size_t num_minor, size_t block_size, cl_int *err);
}