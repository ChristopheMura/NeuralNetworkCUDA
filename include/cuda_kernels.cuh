#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <vector>
#include "types.h"


// ============================================================
//  Structure de memoire GPU pre-allouee
//  Allouer une seule fois avec AllocateGPUMemory,
//  reutiliser a chaque ForwardPropagation_GPU,
//  liberer a la fin avec FreeGPUMemory
// ============================================================
struct GPUMemory
{
    double *d_X;
    double *d_W1, *d_b1, *d_Z1, *d_A1;
    double *d_W2, *d_b2, *d_Z2, *d_A2;
    double *d_W3, *d_b3, *d_Z3, *d_A3;

    // Dimensions stockees pour verification
    int n0, n1, n2, n3, m;
};

// Alloue la memoire GPU une seule fois
GPUMemory AllocateGPUMemory(int n0, int n1, int n2, int n3, int m);

// Libere la memoire GPU
void FreeGPUMemory(GPUMemory& mem);

// Version avec memoire pre-allouee (rapide — pour benchmark et entrainement)
Activations ForwardPropagation_GPU(std::vector<std::vector<double>>& X, NetworkParams& params, GPUMemory& mem);

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
Activations ForwardPropagation_GPU(std::vector<std::vector<double>>& X, NetworkParams& params);

#endif