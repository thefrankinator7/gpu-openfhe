#include <stdio.h>
#include <cstdint>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using std::uint64_t;

__global__ void vecAdd(
    const uint64_t *a, const uint64_t *b, uint64_t *c, size_t n, uint64_t modulus)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
    c[id] = a[id] + b[id];
    if (c[id]>modulus) {
        c[id]=c[id]-modulus;
    }
    if (c[id]<0) {
        c[id]=c[id]+modulus;
    }
}

__global__ void vecMult(
    const uint64_t *a, const uint64_t *b, uint64_t *c, size_t n, uint64_t modulus)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if (id < n)
    c[id] = a[id] * b[id] % modulus;
    // while (c[id]>modulus) {
    //     c[id]=c[id]-modulus;
    // }
    // while (c[id]<0) {
    //     c[id]=c[id]+modulus;
    // }
}

__global__ void nttKernel(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        for (int len = 2; len <= n; len <<= 1) {
            int halfLen = len >> 1;
            int baseTwiddle = n / len;
            for (int i = id; i < n; i += len) {
                int j = i + halfLen;
                int twiddleIdx = (i % halfLen) * baseTwiddle;
                uint64_t t = (twiddles[twiddleIdx] * data[j]) % p;
                data[j] = (data[i] - t + p) % p;
                data[i] = (data[i] + t) % p;
            }
        }
        __syncthreads();
    }
}

uint64_t* moveArrayToGPU(uint64_t* array, int n) {
    uint64_t *GPUArray;
    size_t bytes = n*sizeof(uint64_t);
    HIP_ASSERT(hipMalloc(&GPUArray, bytes));
    HIP_ASSERT(hipMemcpy(GPUArray, array, bytes, hipMemcpyHostToDevice));
    return GPUArray;
}

uint64_t* moveArrayToHost(uint64_t* GPUArray, int n) {
    uint64_t* HostArray = new uint64_t[n]();
    size_t bytes = n*sizeof(uint64_t);
    HIP_ASSERT(hipMemcpy(HostArray, GPUArray, bytes, hipMemcpyDeviceToHost));
    return HostArray;
}


void gpuAdd(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
    int n, uint64_t modulus)
    {
    //Number of threads in each thread block
    int blockSize = 1024;
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
    // add start timer
    
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, n, modulus);
    hipDeviceSynchronize();
    // add end timer, print
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout.precision(8);

    //std::cout << "add time " << duration << "ns" << std::endl;
    }


void gpuMult(const uint64_t *CPUArrayA, const uint64_t *CPUArrayB, uint64_t *CPUArrayC,
    size_t n, uint64_t modulus)
    {
        // Declare all CPU arrays here
    // Size, in bytes, of each vector
    
    uint64_t *GPUArrayA, *GPUArrayB, *GPUArrayC;
    size_t bytes = n*sizeof(uint64_t);

    HIP_ASSERT(hipMalloc(&GPUArrayA, bytes));
    HIP_ASSERT(hipMalloc(&GPUArrayB, bytes));
    HIP_ASSERT(hipMalloc(&GPUArrayC, bytes));
    HIP_ASSERT(hipMemcpy(GPUArrayA, CPUArrayA, bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(GPUArrayB, CPUArrayB, bytes, hipMemcpyHostToDevice));

    //Number of threads in each thread block
    int blockSize = 1024;
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
    // Execute the kernel
    vecMult<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, n, modulus);
    hipDeviceSynchronize();
    
    HIP_ASSERT(hipMemcpy(CPUArrayC, GPUArrayC, bytes, hipMemcpyDeviceToHost));
    }

void gpuNtt(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p) {
    uint64_t *d_data, *d_twiddles;
    hipMalloc(&d_data, n * sizeof(int));
    hipMalloc(&d_twiddles, n * sizeof(int));
    hipMemcpy(d_data, data, n * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_twiddles, twiddles, n * sizeof(int), hipMemcpyHostToDevice);

    //Number of threads in each thread block
    int blockSize = 1024;
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
    // add start timer
    
    auto start = std::chrono::high_resolution_clock::now();
    nttKernel<<<gridSize, blockSize>>>(d_data, d_twiddles, n, p);

    hipMemcpy(data, d_data, n * sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_data);
    hipFree(d_twiddles);
}