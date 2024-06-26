#include <stdio.h>
#include "hip/hip_runtime.h"
#include "include/functions.hpp"

uint64_t* moveArrayToGPU(uint64_t* array, int n);
uint64_t* moveArrayToHost(uint64_t* GPUArray, int n);

void gpuAdd(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
    int N, int L, uint64_t* moduli);

void gpuMult(uint64_t *CPUArrayA, uint64_t *CPUArrayB, uint64_t *CPUArrayC,
   int N, int L, uint64_t* moduli);

void gpuNtt(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p);

void hipSync();

void gpuEmptyKernel();

void gpuNTT(uint64_t *rns_arrays, uint64_t *psi_arrays, uint64_t logN, int N, int L, uint64_t* moduli, uint64_t* mus, int qbit);

void gpuINTT(uint64_t *rns_arrays, uint64_t *inv_psi_arrays, uint64_t logN, int N, int L, uint64_t* moduli, uint64_t* mus, int qbit);