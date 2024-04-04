#include <stdio.h>
#include <cstdint>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>

// #include <cassert>
// #include <cmath>
// #include <cstdlib>
// #include <ctime>


#define uint128_t          unsigned __int128
#define HIP_ASSERT(x) (assert((x)==hipSuccess))

using std::uint64_t;

__global__ void vecAdd(
    const uint64_t *a, const uint64_t *b, uint64_t *c, int total, uint64_t* moduli)
{
    int id=blockIdx.y * gridDim.x * blockDim.x +
                blockIdx.x * blockDim.x +
                threadIdx.x;
    
    uint64_t modulus = moduli[blockIdx.y];
    if (id < total) {
        c[id]=(a[id] + b[id]) % modulus;
    }
}

__global__ void vecMult(
    const uint64_t *a, const uint64_t *b, uint64_t *c, int total, uint64_t* moduli)
{
    int id=blockIdx.y * gridDim.x * blockDim.x +
                blockIdx.x * blockDim.x +
                threadIdx.x;

    uint64_t modulus = moduli[blockIdx.y];

    uint128_t result=a[id] * b[id] % modulus;

    if (id < total) {
        c[id]= (uint64_t) result;
    }
}

uint64_t* moveArrayToGPU(uint64_t* array, int n) {
    uint64_t *GPUArray;
    size_t bytes = n*sizeof(uint64_t);
    HIP_ASSERT(hipMalloc(&GPUArray, bytes));
    HIP_ASSERT(hipMemcpy(GPUArray, array, bytes, hipMemcpyHostToDevice));
    hipDeviceSynchronize();
    return GPUArray;
}

uint64_t* moveArrayToHost(uint64_t* GPUArray, int n) {
    size_t bytes = n*sizeof(uint64_t);
    uint64_t* HostArray;
 
    HIP_ASSERT(hipHostMalloc(&HostArray, bytes, hipHostMallocDefault));
    HIP_ASSERT(hipMemcpy(HostArray, GPUArray, bytes, hipMemcpyDeviceToHost));
    return HostArray;
}


void gpuAdd(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
    int N, int L, uint64_t* moduli)
    {
    int total=N*L;
    
    dim3 blockSize(1024, 1); // 1024 threads per block

    // Calculate the number of blocks needed in each dimension
    int gridX = (int)ceil((float) N / blockSize.x);
    int gridY = L;//(int)ceil((float) L / blockSize.y);
    dim3 gridSize(gridX, gridY); // Grid size

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, total, moduli);
    //hipDeviceSynchronize();

    }

void gpuMult(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
    int N, int L, uint64_t* moduli)
    {
    int total=N*L;
    
    dim3 blockSize(1024, 1); // 1024 threads per block

    // Calculate the number of blocks needed in each dimension
    int gridX = (int)ceil((float) N / blockSize.x);
    int gridY = L;//(int)ceil((float) L / blockSize.y);
    dim3 gridSize(gridX, gridY); // Grid size

    // Execute the kernel
    vecMult<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, total, moduli);
    //hipDeviceSynchronize();

    }


void hipSync(){
    hipDeviceSynchronize();
}

// void gpuRelin(uint64_t *GPUArrayA, uint64_t *GPUArrayB, uint64_t *GPUArrayC,
//     int N, int L, uint64_t* moduli, uint64_t* evk)

// void gpuMult(const uint64_t *CPUArrayA, const uint64_t *CPUArrayB, uint64_t *CPUArrayC,
//     size_t n, uint64_t modulus)
//     {
//         // Declare all CPU arrays here
//     // Size, in bytes, of each vector
    
//     uint64_t *GPUArrayA, *GPUArrayB, *GPUArrayC;
//     size_t bytes = n*sizeof(uint64_t);

//     HIP_ASSERT(hipMalloc(&GPUArrayA, bytes));
//     HIP_ASSERT(hipMalloc(&GPUArrayB, bytes));
//     HIP_ASSERT(hipMalloc(&GPUArrayC, bytes));
//     HIP_ASSERT(hipMemcpy(GPUArrayA, CPUArrayA, bytes, hipMemcpyHostToDevice));
//     HIP_ASSERT(hipMemcpy(GPUArrayB, CPUArrayB, bytes, hipMemcpyHostToDevice));

//     //Number of threads in each thread block
//     int blockSize = 1024;
//     // Number of thread blocks in grid
//     int gridSize = (int)ceil((float)n/blockSize);
//     // Execute the kernel
//     vecMult<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, n, modulus);
//     hipDeviceSynchronize();
    
//     HIP_ASSERT(hipMemcpy(CPUArrayC, GPUArrayC, bytes, hipMemcpyDeviceToHost));
//     }


// __global__ void nttKernel(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p) {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (id < n) {
//         for (int len = 2; len <= n; len <<= 1) {
//             int halfLen = len >> 1;
//             int baseTwiddle = n / len;
//             for (int i = id; i < n; i += len) {
//                 int j = i + halfLen;
//                 int twiddleIdx = (i % halfLen) * baseTwiddle;
//                 uint64_t t = (twiddles[twiddleIdx] * data[j]) % p;
//                 data[j] = (data[i] - t + p) % p;
//                 data[i] = (data[i] + t) % p;
//             }
//         }
//         __syncthreads();
//     }
// }

// void gpuNtt(uint64_t *data, const uint64_t *twiddles, size_t n, size_t p) {
//     uint64_t *d_data, *d_twiddles;
//     hipMalloc(&d_data, n * sizeof(int));
//     hipMalloc(&d_twiddles, n * sizeof(int));
//     hipMemcpy(d_data, data, n * sizeof(int), hipMemcpyHostToDevice);
//     hipMemcpy(d_twiddles, twiddles, n * sizeof(int), hipMemcpyHostToDevice);

//     //Number of threads in each thread block
//     int blockSize = 1024;
//     // Number of thread blocks in grid
//     int gridSize = (int)ceil((float)n/blockSize);
//     // add start timer
    
//     auto start = std::chrono::high_resolution_clock::now();
//     nttKernel<<<gridSize, blockSize>>>(d_data, d_twiddles, n, p);

//     hipMemcpy(data, d_data, n * sizeof(int), hipMemcpyDeviceToHost);
//     hipFree(d_data);
//     hipFree(d_twiddles);
// }