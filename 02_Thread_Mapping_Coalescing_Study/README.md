```
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
```

![GPU_Execution](/02_Thread_Mapping_Coalescing_Study/Thread_mapping_study.png)
