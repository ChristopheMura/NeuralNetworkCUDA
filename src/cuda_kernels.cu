#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <cstdio>

// ============================================================
//  MACRO utilitaire : vérifie les erreurs CUDA
//  Usage : CUDA_CHECK(cudaMalloc(...))
// ============================================================

#define CUDA_CHECK(call)                                                                                \
    do {                                                                                                \
        cudaError_t err = (call);                                                                       \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            throw std::runtime_error(cudaGetErrorString(err));                                          \
        }                                                                                               \
    } while (0)


// ============================================================
//  ÉTAPE 1 — Fonctions utilitaires CPU
//  Aplatissement : vector<vector<double>> → vector<double>
//
//  Exemple pour une matrice M de taille (rows x cols) :
//    M[i][j]  →  flat[i * cols + j]
//
//  IMPORTANT : X est stockée (m x n0) côté CPU
//              mais on la transpose en (n0 x m) pour le GPU
//              afin que la matmul W·X soit cohérente
// ============================================================

// Aplatit une matrice row-major
static std::vector<double> flatten(const std::vector<std::vector<double>>& M)
{
    int rows = M.size();
    int cols = M[0].size();
    std::vector<double> flat(rows * cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            flat[i * cols + j] = M[i][j];
        }
    }

    return flat;
}

// Aplatit X et la transpose : X CPU est (m x n0), on veut (n0 x m) sur GPU
// Ainsi W1(n1 x n0) · X_T(n0 x m) = Z1(n1 x m)
static std::vector<double> flattenTranspose(const std::vector<std::vector<double>>& M)
{
    int rows = M.size();
    int cols = M[0].size();
    std::vector<double> flat(rows * cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            flat[j * rows + i] = M[i][j];
        }
    }

    return flat;
}

// Reconstruit un vector<vector<double>> depuis un tableau plat (rows x cols)
static std::vector<std::vector<double>> unflatten(const std::vector<double>& flat, int rows, int cols)
{
    std::vector<std::vector<double>> M(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            M[i][j] = flat[i * cols + j];
        }
    }

    return M;
}


// ----------------------------------------------------------
//  Kernel 1 : Multiplication matricielle + biais
//  Calcule C = A · B + bias
//
//  A  : (rowsA x colsA)
//  B  : (colsA x colsB)
//  C  : (rowsA x colsB)
//  bias : (rowsA) — ajouté à chaque colonne de C
//
//  Chaque thread calcule UN élément C[row][col]
// ----------------------------------------------------------
__global__ void matMulBiasKernel(const double* A, const double* B, const double* bias, double* C, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        double sum = bias[row];     // Biais de la ligne
        for (int k = 0; k < colsA; k++)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// ----------------------------------------------------------
//  Kernel 2 : Sigmoid élément par élément
//  A[i] = 1 / (1 + exp(-Z[i]))
//
//  Chaque thread traite UN élément
// ----------------------------------------------------------
__global__ void sigmoidKernel(const double* Z, double* A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        A[idx] = 1.0f / (1.0f + exp(-Z[idx]));
    }
}

// ============================================================
//  Fonction helper : lance matMulBias + sigmoid pour une couche
//
//  W     : poids sur GPU  (n_out x n_in)
//  Input : entrée sur GPU (n_in x m)
//  b     : biais sur GPU  (n_out)
//  Z     : sortie Z       (n_out x m)  [sortie]
//  A     : sortie A       (n_out x m)  [sortie]
// ============================================================
static void forwardLayer(const double* W, const double* Input, const double* b, double* Z, double* A, int n_out, int n_in, int m)
{
    /* *** matMulBias : Z = W . Input + b *** */
    dim3 block(32, 8);
    dim3 grid((m        + block.x - 1) / block.x,
              (n_out    + block.y - 1) / block.y);
    
    matMulBiasKernel<<<grid, block>>>(W, Input, b, Z, n_out, n_in, m);
    CUDA_CHECK(cudaGetLastError());

    /* *** Sigmoid : A = sigmoid(Z) *** */
    int total = n_out * m;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    sigmoidKernel<<<blocks, threads>>>(Z, A, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
//  AllocateGPUMemory — appel unique avant l'entrainement
// ============================================================
GPUMemory AllocateGPUMemory(int n0, int n1, int n2, int n3, int m)
{
    GPUMemory mem;

    mem.n0 = n0; 
    mem.n1 = n1; 
    mem.n2 = n2; 
    mem.n3 = n3; 
    mem.m = m;

    CUDA_CHECK(cudaMalloc(&mem.d_X, n0 * m * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&mem.d_W1, n1 * n0 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_b1, n1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_Z1, n1 * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_A1, n1 * m * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&mem.d_W2, n2 * n1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_b2, n2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_Z2, n2 * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_A2, n2 * m * sizeof(double)));

    CUDA_CHECK(cudaMalloc(&mem.d_W3, n3 * n2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_b3, n3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_Z3, n3 * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mem.d_A3, n3 * m * sizeof(double)));

    return mem;
}

// ============================================================
//  FreeGPUMemory
// ============================================================
void FreeGPUMemory(GPUMemory& mem)
{
    cudaFree(mem.d_X);

    cudaFree(mem.d_W1); 
    cudaFree(mem.d_b1); 
    cudaFree(mem.d_Z1); 
    cudaFree(mem.d_A1);

    cudaFree(mem.d_W2); 
    cudaFree(mem.d_b2); 
    cudaFree(mem.d_Z2); 
    cudaFree(mem.d_A2);

    cudaFree(mem.d_W3); 
    cudaFree(mem.d_b3); 
    cudaFree(mem.d_Z3); 
    cudaFree(mem.d_A3);
}

// ============================================================
//  ÉTAPE 2 + 4 — ForwardPropagation_GPU
//
//  1. Aplatir et transposer X (CPU → format GPU)
//  2. Allouer la mémoire GPU (cudaMalloc)
//  3. Copier CPU → GPU (cudaMemcpy HostToDevice)
//  4. Lancer les kernels pour chaque couche
//  5. Copier GPU → CPU (cudaMemcpy DeviceToHost)
//  6. Libérer la mémoire GPU (cudaFree)
//  7. Reconstruire la structure Activations
// ============================================================

// ============================================================
//  ForwardPropagation_GPU — version avec memoire pre-allouee
//  C'est cette version qu'on utilise pour l'entrainement
// ============================================================
Activations ForwardPropagation_GPU(std::vector<std::vector<double>>& X, NetworkParams& params)
{
    int m  = (int)X.size();
    int n0 = (int)X[0].size();
    int n1 = (int)params.W1.size();
    int n2 = (int)params.W2.size();
    int n3 = (int)params.W3.size();

    GPUMemory mem = AllocateGPUMemory(n0, n1, n2, n3, m);
    Activations activations = ForwardPropagation_GPU(X, params, mem);
    FreeGPUMemory(mem);

    return activations;
}

Activations ForwardPropagation_GPU(std::vector<std::vector<double>>& X, NetworkParams& params, GPUMemory& mem)
{
    int m  = mem.m;
    int n0 = mem.n0;
    int n1 = mem.n1;
    int n2 = mem.n2;
    int n3 = mem.n3;

    // --- Aplatissement CPU ---
    std::vector<double> h_X  = flattenTranspose(X);
    std::vector<double> h_W1 = flatten(params.W1);
    std::vector<double> h_W2 = flatten(params.W2);
    std::vector<double> h_W3 = flatten(params.W3);

    // --- Copie CPU -> GPU (pas de cudaMalloc ici) ---
    CUDA_CHECK(cudaMemcpy(mem.d_X,  h_X.data(),          n0 * m  * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_W1, h_W1.data(),         n1 * n0 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_b1, params.b1.data(),    n1       * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_W2, h_W2.data(),         n2 * n1 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_b2, params.b2.data(),    n2       * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_W3, h_W3.data(),         n3 * n2 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mem.d_b3, params.b3.data(),    n3       * sizeof(double), cudaMemcpyHostToDevice));

    // --- Kernels ---
    forwardLayer(mem.d_W1, mem.d_X,  mem.d_b1, mem.d_Z1, mem.d_A1, n1, n0, m);
    forwardLayer(mem.d_W2, mem.d_A1, mem.d_b2, mem.d_Z2, mem.d_A2, n2, n1, m);
    forwardLayer(mem.d_W3, mem.d_A2, mem.d_b3, mem.d_Z3, mem.d_A3, n3, n2, m);

    // Un seul sync a la fin de tout le forward pass
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copie GPU -> CPU ---
    std::vector<double> h_Z1(n1 * m), h_A1(n1 * m);
    std::vector<double> h_Z2(n2 * m), h_A2(n2 * m);
    std::vector<double> h_Z3(n3 * m), h_A3(n3 * m);

    CUDA_CHECK(cudaMemcpy(h_Z1.data(), mem.d_Z1, n1 * m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_A1.data(), mem.d_A1, n1 * m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z2.data(), mem.d_Z2, n2 * m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_A2.data(), mem.d_A2, n2 * m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Z3.data(), mem.d_Z3, n3 * m * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_A3.data(), mem.d_A3, n3 * m * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Reconstruction Activations ---
    Activations activations;
    activations.Z1 = unflatten(h_Z1, n1, m);
    activations.A1 = unflatten(h_A1, n1, m);
    activations.Z2 = unflatten(h_Z2, n2, m);
    activations.A2 = unflatten(h_A2, n2, m);
    activations.Z3 = unflatten(h_Z3, n3, m);
    activations.A3 = unflatten(h_A3, n3, m);

    return activations;
}

// Activations ForwardPropagation_GPU(std::vector<std::vector<double>>& X, NetworkParams& params)
// {
//     // --- Dimensions ---
//     int m = X.size();           // nombre d'échantillons
//     int n0 = X[0].size();       // taille entrée
//     int n1 = params.W1.size();  // neuronnes couche 1
//     int n2 = params.W2.size();  // neuronnes couche 2
//     int n3 = params.W3.size();  // neuronnes couche 3 (couche finale = 1)

//     // --------------------------------------------------------
//     //  ÉTAPE 1 - Aplatissement CPU
//     // --------------------------------------------------------
//     std::vector<double> h_X = flattenTranspose(X);  // (n0 * m)
//     std::vector<double> h_W1 = flatten(params.W1);  // (n1 * n0)
//     std::vector<double> h_W2 = flatten(params.W2);  // (n2 * n1)
//     std::vector<double> h_W3 = flatten(params.W3);  // (n3 * n2)

//     // Les biais sont déjà des vector<double>, donc pas besoin de flatten
//     const std::vector<double>& h_b1 = params.b1;    // (n1)
//     const std::vector<double>& h_b2 = params.b2;    // (n2)
//     const std::vector<double>& h_b3 = params.b3;    // (n3)

//     // --------------------------------------------------------
//     //  ÉTAPE 2 - Allocation GPU
//     // --------------------------------------------------------
//     double *d_X,    *d_W1,  *d_b1,  *d_Z1,  *d_A1;
//     double *d_W2,   *d_b2,  *d_Z2,  *d_A2;
//     double *d_W3,   *d_b3,  *d_Z3,  *d_A3;

//     // --- Entrée ---
//     CUDA_CHECK( cudaMalloc(&d_X, n0 * m * sizeof(double)) );

//     // --- Couche 1 ---
//     CUDA_CHECK( cudaMalloc(&d_W1, n1 * n0 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_b1, n1 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_Z1, n1 * m * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_A1, n1 * m * sizeof(double)) );

//     // --- Couche 2 ---
//     CUDA_CHECK( cudaMalloc(&d_W2, n2 * n1 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_b2, n2 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_Z2, n2 * m * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_A2, n2 * m * sizeof(double)) );

//     // --- Couche 3 ---
//     CUDA_CHECK( cudaMalloc(&d_W3, n3 * n2 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_b3, n3 * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_Z3, n3 * m * sizeof(double)) );
//     CUDA_CHECK( cudaMalloc(&d_A3, n3 * m * sizeof(double)) );

//     // --------------------------------------------------------
//     //  ÉTAPE 3 - Copie CPU to GPU (cudaMemcpyHostToDevice)
//     // --------------------------------------------------------
//     CUDA_CHECK( cudaMemcpy(d_X, h_X.data(), n0 * m * sizeof(double), cudaMemcpyHostToDevice) );

//     CUDA_CHECK( cudaMemcpy(d_W1, h_W1.data(), n1 * n0 * sizeof(double), cudaMemcpyHostToDevice) );
//     CUDA_CHECK( cudaMemcpy(d_b1, h_b1.data(), n1 * sizeof(double), cudaMemcpyHostToDevice) );

//     CUDA_CHECK( cudaMemcpy(d_W2, h_W2.data(), n2 * n1 * sizeof(double), cudaMemcpyHostToDevice) );
//     CUDA_CHECK( cudaMemcpy(d_b2, h_b2.data(), n2 * sizeof(double), cudaMemcpyHostToDevice) );

//     CUDA_CHECK( cudaMemcpy(d_W3, h_W3.data(), n3 * n2 * sizeof(double), cudaMemcpyHostToDevice) );
//     CUDA_CHECK( cudaMemcpy(d_b3, h_b3.data(), n3 * sizeof(double), cudaMemcpyHostToDevice) );

//     // --------------------------------------------------------
//     //  ÉTAPE 4 - Kernels (une couche à la fois)
//     // --------------------------------------------------------
//     // Couche 1 : Z1 = W1 . X + b1, A1 = sigmoid(Z1)
//     forwardLayer(d_W1, d_X, d_b1, d_Z1, d_A1, n1, n0, m);
    
//     // Couche 2 : Z2 = W2 . X + b2, A2 = sigmoid(Z2)
//     forwardLayer(d_W2, d_A1, d_b2, d_Z2, d_A2, n2, n1, m);

//     // Couche 3 : Z3 = W3 . X + b3, A3 = sigmoid(Z3)
//     forwardLayer(d_W3, d_A2, d_b3, d_Z3, d_A3, n3, n2, m);

//     // --------------------------------------------------------
//     //  ÉTAPE 5 - Copie GPU to CPU
//     // --------------------------------------------------------
//     std::vector<double> h_Z1(n1 * m), h_A1(n1 * m);
//     std::vector<double> h_Z2(n2 * m), h_A2(n2 * m);
//     std::vector<double> h_Z3(n3 * m), h_A3(n3 * m);

//     CUDA_CHECK( cudaMemcpy(h_Z1.data(), d_Z1, n1 * m * sizeof(double), cudaMemcpyDeviceToHost) );
//     CUDA_CHECK( cudaMemcpy(h_A1.data(), d_A1, n1 * m * sizeof(double), cudaMemcpyDeviceToHost) );

//     CUDA_CHECK( cudaMemcpy(h_Z2.data(), d_Z2, n2 * m * sizeof(double), cudaMemcpyDeviceToHost) );
//     CUDA_CHECK( cudaMemcpy(h_A2.data(), d_A2, n2 * m * sizeof(double), cudaMemcpyDeviceToHost) );

//     CUDA_CHECK( cudaMemcpy(h_Z3.data(), d_Z3, n3 * m * sizeof(double), cudaMemcpyDeviceToHost) );
//     CUDA_CHECK( cudaMemcpy(h_A3.data(), d_A3, n3 * m * sizeof(double), cudaMemcpyDeviceToHost) );

//     // --------------------------------------------------------
//     //  ÉTAPE 6 - Libération mémoire GPU
//     // --------------------------------------------------------
//     cudaFree(d_X);

//     cudaFree(d_W1);
//     cudaFree(d_b1);
//     cudaFree(d_Z1);
//     cudaFree(d_A1);

//     cudaFree(d_W2);
//     cudaFree(d_b2);
//     cudaFree(d_Z2);
//     cudaFree(d_A2);

//     cudaFree(d_W3);
//     cudaFree(d_b3);
//     cudaFree(d_Z3);
//     cudaFree(d_A3);

//     // --------------------------------------------------------
//     //  ÉTAPE 7 - Reconstruction de la structure Activations
//     //  On remet les tableaux plats dans des vector<vector<double>>
//     //  au format (n x m) attendu par BackPropagation
//     // --------------------------------------------------------
//     Activations activattions;

//     activattions.Z1 = unflatten(h_Z1, n1, m);
//     activattions.A1 = unflatten(h_A1, n1, m);

//     activattions.Z2 = unflatten(h_Z2, n2, m);
//     activattions.A2 = unflatten(h_A2, n2, m);

//     activattions.Z3 = unflatten(h_Z3, n3, m);
//     activattions.A3 = unflatten(h_A3, n3, m);

//     return activattions;
// }
