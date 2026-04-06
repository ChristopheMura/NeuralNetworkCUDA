#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matMul(double* A, double* B, double* C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        double sum = 0.0f;
        for (int k = 0; k < colsA; k++)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}


int main(int argc, char** argv)
{
    // Matrice 3x2 * 2x3 = 3x3 sur GPU
    int rowsA = 3, colsA = 2, colsB = 3;

    double h_A[] = {1,2, 3,4, 5,6};
    double h_B[] = {7,8,9, 10,11,12};
    double h_C[9] = {0};

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rowsA * colsA * sizeof(double));
    cudaMalloc(&d_B, colsA * colsB * sizeof(double));
    cudaMalloc(&d_C, rowsA * colsB * sizeof(double));

    cudaMemcpy(d_A, h_A, rowsA * colsA * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, colsA * colsB * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((colsB + 15) / 16, (rowsA + 15) / 16);
    matMul<<<grid, block>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    cudaMemcpy(h_C, d_C, rowsA * colsB * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Résultat C:\n");
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++)
            printf("%.0f ", h_C[i * colsB + j]);
        printf("\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}