#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define M 1024
#define K 1024
#define N 1024
#define BLOCK_SIZE 32

__global__ void coalesced_sgemm(int m, int n, int k , float alpha, const float *A, const float *B,float beta,float *C) {
    int threadid = threadIdx.x;
    int row = blockIdx.x * BLOCK_SIZE + threadid / BLOCK_SIZE;
    int col = blockIdx.y * BLOCK_SIZE + threadid % BLOCK_SIZE;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
    
}

int main() {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N ; i++)
    {
        h_B[i] = 1.0f;
    }
    for (int i = 0; i < M * N; i++)
    {
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A , size_A);
    cudaMalloc(&d_B , size_B);
    cudaMalloc(&d_C , size_C);

    cudaMemcpy(d_A , h_A , size_A , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B , size_B , cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C , size_C , cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE * BLOCK_SIZE);

    dim3 blocksPerGrid(
        (M + BLOCK_SIZE - 1 )/BLOCK_SIZE,
        (N + BLOCK_SIZE - 1 )/BLOCK_SIZE
    );

    float alpha = 1.0f;
    float beta = 0.0f;

//warm run

    coalesced_sgemm<<<blocksPerGrid,threadsPerBlock>>>(M , N , K , alpha , d_A , d_B , beta , d_C);

    cudaDeviceSynchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    coalesced_sgemm<<<blocksPerGrid,threadsPerBlock>>>(M , N , K , alpha , d_A , d_B , beta , d_C);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds , start , stop);

    cudaMemcpy(h_C , d_C , size_C , cudaMemcpyDeviceToHost);

    double flops = 2.0 * M * N * K;
    double gflops = flops / (milliseconds / 1000.0) / 1e9;

    double estimated_bytes = (2.0 * M * N * K * sizeof(float)) + (1.0 * M * N * sizeof(float));

    double arithmetic_intensity = flops / estimated_bytes;

    printf("Coalesced SGEMM output :\n");
    printf("C[0] : %f\n" , h_C[0]);
    printf("C[last] = %f\n" , h_C[M * N - 1]);


    printf("Performance Metrics:\n");
    printf("Time : %f ms\n" , milliseconds);
    printf("Gflops : %f\n" , gflops);
    printf("Estimated Memory traffic : %f GB\n" , estimated_bytes / 1e9);
    printf("Arithmetic Intensity : %f FLOP/byte\n" , arithmetic_intensity);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

   return 0; 
}
