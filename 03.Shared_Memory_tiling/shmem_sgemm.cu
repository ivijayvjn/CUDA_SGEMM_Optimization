#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define M 1024
#define K 1024
#define N 1024
#define BLOCKSIZE 32

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

int main() {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f;
    }

    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCKSIZE * BLOCKSIZE);

    dim3 blocksPerGrid(
        (M + BLOCKSIZE - 1) / BLOCKSIZE,
        (N + BLOCKSIZE - 1) / BLOCKSIZE
    );

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up run
    shmem_sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    shmem_sgemm<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (milliseconds / 1000.0) / 1e9;
// Each A/B tile value is reused roughly by BLOCSIZE threads after being loaded into shared memory
    double estimated_bytes =
        (2.0 * M * N * K * sizeof(float) / BLOCKSIZE ) +
        (1.0 * M * N * sizeof(float));

    double arithmetic_intensity = flops / estimated_bytes;

    printf("Shared Memory SGEMM output:\n");
    printf("C[0] = %f\n", h_C[0]);
    printf("C[last] = %f\n", h_C[M * N - 1]);

    printf("\nPerformance Metrics:\n");
    printf("Time: %f ms\n", milliseconds);
    printf("GFLOP/s: %f\n", gflops);
    printf("Estimated Memory Traffic: %f GB\n", estimated_bytes / 1e9);
    printf("Arithmetic Intensity: %f FLOP/byte\n", arithmetic_intensity);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
