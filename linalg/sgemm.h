#pragma once

#include <CL/cl.h>
#include <cstddef>

namespace linalg {
    /**
     * @brief Performs matrix multiplication of matrices A and B and stores the result in C.
     * All arrays should be stored in pre-allocated host memory.
     *
     * @param A An input float buffer containing the first matrix to multiply.
     * @param B An input float buffer containing the second matrix to multiply.
     * @param C An output float buffer to store the result of the matrix multiplication.
     * @param M The number of rows in matrix A and C.
     * @param N The number of columns in matrix B and C.
     * @param K The number of columns in matrix A and rows in matrix B.
     * 
     * @return An error code, or CL_SUCCESS if no error occurred
     */
    cl_int sgemm(double *A, double *B, double *C, unsigned int M, unsigned int N, unsigned int K);

    // Same as normal sgemm, but i/o are on device memory
    cl_int sgemmDevice(cl_mem A, cl_mem B, cl_mem C, unsigned int M, unsigned int N, unsigned int K);
}