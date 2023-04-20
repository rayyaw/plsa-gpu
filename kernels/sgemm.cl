#define BLOCK_SIZE 16

// Known correct version
__kernel void sgemm(__global const double *A,
                    __global const double *B,
                    __global double *C,
                    const unsigned int M,
                    const unsigned int N,
                    const unsigned int K) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    double sum = 0.0f;

    int it = get_local_id(0);
    int jt = get_local_id(1);

    __local double A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local double B_tile[BLOCK_SIZE][BLOCK_SIZE]; 

    for (int stride = 0; stride < K; stride += BLOCK_SIZE) {
        // Shared memory loads
        if (i < M && stride + jt < K) {
            A_tile[it][jt] = A[i*K + (stride+jt)];
        } else {
            A_tile[it][jt] = 0;
        }

        if (j < N && stride + it < K) {
            B_tile[it][jt] = B[(stride+it)*N + j];
        } else {
            B_tile[it][jt] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform computation on tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += A_tile[it][k] * B_tile[k][jt];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (i < M && j < N) {
        C[i*N + j] = sum;
    }

}
