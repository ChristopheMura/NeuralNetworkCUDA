#include <stdio.h>

__global__ void hello()
{
    printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv)
{
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    return 0;
}