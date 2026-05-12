```
__global__ void naive_sgemm(int m, int n, int k , float alpha, const float *A, const float *B,float beta,float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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

![GPU_Execution](/01.Naive_implementation/Naive_SGEMM_output.png)
