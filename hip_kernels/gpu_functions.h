#include <stdio.h>
#include "hip/hip_runtime.h"

uint64_t* moveArrayToGPU(uint64_t* array, int n);
uint64_t* moveArrayToHost(uint64_t* GPUArray, int n);

void gpuAdd(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
    int n, uint64_t modulus);

void gpuMult(const uint64_t *CPUArrayA, const uint64_t *CPUArrayB, uint64_t *CPUArrayC,
    size_t n, uint64_t modulus);

void gpuNtt(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p);