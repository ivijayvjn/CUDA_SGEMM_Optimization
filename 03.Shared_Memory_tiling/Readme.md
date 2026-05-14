``` cuda

__global__ void shmem_sgemm(int m, int n, int k,float alpha,const float *A,const float *B,float beta,float *C) 
{
//which tile row of C this cuda block compute     
    const uint cRow = blockIdx.x;
//which tile column of C this cuda block compute
    const uint cCol = blockIdx.y;
//thread column position inside the tile
    const uint threadCol = threadIdx.x % BLOCKSIZE;
//thread row position inside the tile
    const uint threadRow = threadIdx.x / BLOCKSIZE;
//allocate shared memory tile cache for A     
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
//allocate shared memory tile cache for B
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
//Moving A pointer to the starting row tile needed by this block
    A += cRow * BLOCKSIZE * k;
//moving B pointer to the starting column tile needed by this block    
    B += cCol * BLOCKSIZE;
//moving c pointer to the output tile this block will compute 
    C += cRow * BLOCKSIZE * n + cCol * BLOCKSIZE;
//created accumulator
    float tmp = 0.0f;
//loop through A and B tiles along the K dimension
    for (int bkIdx = 0; bkIdx < k; bkIdx += BLOCKSIZE) {
//load one A element from global memory into shared memory as As
        As[threadRow * BLOCKSIZE + threadCol] =
            A[threadRow * k + threadCol];
//load one B element from global memory into shared memory as Bs

        Bs[threadRow * BLOCKSIZE + threadCol] =
            B[threadRow * n + threadCol];
//synchronize threads so all load happens

        __syncthreads();
//advance glonal memory pointers to the next A and B tiles

        A += BLOCKSIZE;
        B += BLOCKSIZE * n;

// Compute partial dot product using values cached in shared memory
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
// Ensure all threads finish using current shared-memory tiles before overwriting them
        __syncthreads();
    }
//write into final matrix
    C[threadRow * n + threadCol] =
        alpha * tmp + beta * C[threadRow * n + threadCol];
}

```

![GPU_Execution](/03.Shared_Memory_tiling/output.png)
