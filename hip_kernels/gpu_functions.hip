#include <stdio.h>
#include <cstdint>
#include "hip/hip_runtime.h"
#include <chrono>
#include <iostream>
#include "functions.hpp"

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

// Modular multipliers //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void neals_barrett(uint128_t& c, uint64_t& q, uint64_t& mu, int& qbit)  // Ozerk
{  
    uint128_t rx;
    rx = c >> (qbit - 2);
    rx *= mu;
    rx >>= qbit + 3;
    rx *= q;
    c -= rx;
    c -= q * (c >= (uint128_t)q);
}

// Butterfly operations //////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void barrett_CT_butterfly(uint64_t& a, uint64_t& b, uint64_t& twiddle, uint64_t &q, uint64_t& mu, int& qbit){
    uint128_t temp_mult = (uint128_t)b * twiddle;
    neals_barrett(temp_mult, q, mu, qbit);
    b = (uint64_t)temp_mult;
    uint64_t temp = a + b;
    temp -= q * (temp>=q);
    b = q*(a<b) + (a-b);
    a = temp;
};

__device__ __forceinline__ void barrett_GS_butterfly(uint64_t& a, uint64_t& b, uint64_t& twiddle, uint64_t &q, uint64_t& q2, uint64_t& mu, int& qbit){
    uint64_t temp = a + b;
    temp -= q * (temp>=q); // addition
    a += q*(a < b);
    uint128_t temp_mult = (uint128_t)(a - b); 
    temp_mult *= twiddle;
    neals_barrett(temp_mult, q, mu, qbit);
    a = (temp >> 1) + q2*(temp & 1);   // store addition result
    temp = (uint64_t)temp_mult;                   // cast to uint64_t
    b = (temp >> 1) + q2*(temp & 1);   // store subtraction result
}

// 2D (i)NTT kernels ///////////////////////////////////////////////////////////////////////////////////////
__global__ void NTT2D_1st(  uint64_t *a,
                            uint64_t *psi,
                            uint64_t q,
                            uint64_t mu,
                            int qbit){
    // -----variable init-----
    int index = threadIdx.x * blockDim.x * 2 + blockIdx.x; // global index of the butterfly operation, also the global memory index
    // 2D NTT consists of 2 kernels:
    // 1st kernel : global m != local m, global length = local length
    // 2nd kernel : global m = local m, global length != local length
    // int glo_m = Nper2; // global step size
    int log_g_m = logNper2; // log global step size
    int l_m = 128; // local step size
    int log_l_m = 7; // log local step size

    // -----bring coefficients to the shared memory-----
    __shared__ uint64_t shared_coeff[BLOCK_SIZE2D*2]; // block_size = 128 * 2 = 256, 256-point NTT

    // 1 thread brings 2 coeff
    shared_coeff[threadIdx.x] = a[index]; // take the first coeff for the bfly
    shared_coeff[threadIdx.x+128] = a[index + Nper2]; // take the second coeff for the bfly, the paired coeff

    // wait until all threads finish
    __syncthreads();

    // -----perform 255 point NTT-----
    #pragma unroll
    for (int length = 1 ; length < 256 ; length<<=1, l_m>>=1, log_g_m--, log_l_m--){
        // compute the local butterfly indices
        int i = threadIdx.x >> log_l_m;
        int bfly_i1 = (i*l_m*2) + (threadIdx.x % l_m);
        int bfly_i2 = bfly_i1 + l_m; 

        // generate global twiddle factor
        int psi_idx = length + (index >> log_g_m);
        uint64_t twiddle = psi[psi_idx];

        // butterfly operation
        barrett_CT_butterfly(shared_coeff[bfly_i1], shared_coeff[bfly_i2], twiddle, q, mu, qbit);

        // wait for all threads to complete their butterfly operation
        __syncthreads();
    }
    // -----return the coefficients to the global memory-----
    a[index] = shared_coeff[threadIdx.x];
    a[index + Nper2] = shared_coeff[threadIdx.x + 128];
}

__global__ void NTT2D_2nd(  uint64_t *a,
                            uint64_t *psi,
                            uint64_t q,
                            uint64_t mu,
                            int qbit){
    // -----variable init-----
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index of the butterfly operation
    int g_addr = blockIdx.x * blockDim.x * 2 + threadIdx.x; // global memory index
    // 2D NTT consists of 2 kernels:
    // 1st kernel : global m != local m, global length = local length
    // 2nd kernel : global m = local m, global length != local length
    int m = 128; // local step size
    int logm = 7; // log local step size
    int g_length = 256; // global length, for twiddle factor memory read

    // -----bring coefficients to the shared memory-----
    __shared__ uint64_t shared_coeff[BLOCK_SIZE2D*2]; // block_size = 128 * 2 = 256, 256-point NTT

    // 1 thread brings 2 coeff
    shared_coeff[threadIdx.x] = a[g_addr]; // take the first coeff for the bfly
    shared_coeff[threadIdx.x+128] = a[g_addr + 128]; // take the second coeff for the bfly, the paired coeff

    // wait until all threads finish
    __syncthreads();

    // -----perform 255 point NTT-----
    #pragma unroll
    for (int length = 1 ; length < 256 ; length<<=1, g_length<<=1, m>>=1, logm--){
        // compute the local butterfly indices
        int i = threadIdx.x >> logm;
        int bfly_i1 = (i * m * 2) + (threadIdx.x % m);
        int bfly_i2 = bfly_i1 + m; 

        // generate global twiddle factor
        int psi_idx = g_length + (index >> logm);
        uint64_t twiddle = psi[psi_idx];

        // butterfly operation
        barrett_CT_butterfly(shared_coeff[bfly_i1], shared_coeff[bfly_i2], twiddle, q, mu, qbit);

        // wait for all threads to complete their butterfly operation
        __syncthreads();
    }
    // -----return the coefficients to the global memory-----
    a[g_addr] = shared_coeff[threadIdx.x];
    a[g_addr + 128] = shared_coeff[threadIdx.x + 128];
}

__global__ void iNTT2D_1st( uint64_t *a,
                            uint64_t *psi,
                            uint64_t q,
                            uint64_t mu,
                            int qbit){
    // -----variable init-----
    int index = blockIdx.x * blockDim.x + threadIdx.x; // global index of the butterfly operation
    int g_addr = blockIdx.x * blockDim.x * 2 + threadIdx.x; // global memory index
    // 2D NTT consists of 2 kernels:
    // 1st kernel : global m != local m, global length = local length
    // 2nd kernel : global m = local m, global length != local length
    int m = 1; // local step size
    int logm = 0; // log local step size
    int g_length = Nper2; // global length, for twiddle factor memory read
    uint64_t q2 = (q + 1) >> 1; // for modular reduction

    // -----bring coefficients to the shared memory-----
    __shared__ uint64_t shared_coeff[BLOCK_SIZE2D*2]; // block_size = 128 * 2 = 256, 256-point NTT

    // 1 thread brings 2 coeff
    shared_coeff[threadIdx.x] = a[g_addr]; // take the first coeff for the bfly
    shared_coeff[threadIdx.x+128] = a[g_addr + 128]; // take the second coeff for the bfly, the paired coeff

    // wait until all threads finish
    __syncthreads();

    // -----perform 255 point NTT-----
    #pragma unroll
    for (int length = 128 ; length >= 1 ; length>>=1, g_length>>=1, m<<=1, logm++){
        // compute the local butterfly indices
        int i = threadIdx.x >> logm;
        int bfly_i1 = (i * m * 2) + (threadIdx.x % m);
        int bfly_i2 = bfly_i1 + m; 

        // generate global twiddle factor
        int psi_idx = g_length + (index >> logm);
        uint64_t twiddle = psi[psi_idx];

        // butterfly operation
        barrett_GS_butterfly(shared_coeff[bfly_i1], shared_coeff[bfly_i2], twiddle, q, q2, mu, qbit);

        // wait for all threads to complete their butterfly operation
        __syncthreads();
    }
    // -----return the coefficients to the global memory-----
    a[g_addr] = shared_coeff[threadIdx.x];
    a[g_addr + 128] = shared_coeff[threadIdx.x + 128];
}

__global__ void iNTT2D_2nd( uint64_t *a,
                            uint64_t *psi,
                            uint64_t q,
                            uint64_t mu,
                            int qbit){
    // -----variable init-----
    int index = threadIdx.x * blockDim.x * 2 + blockIdx.x; // global index of the butterfly operation
    // 2D NTT consists of 2 kernels:
    // 1st kernel : global m != local m, global length = local length
    // 2nd kernel : global m = local m, global length != local length
    // int glo_m = Nper2; // global step size
    int log_g_m = 8; // log global step size
    int l_m = 1; // local step size
    int log_l_m = 0; // log local step size
    uint64_t q2 = (q + 1) >> 1; // for modular reduction

    // -----bring coefficients to the shared memory-----
    __shared__ uint64_t shared_coeff[BLOCK_SIZE2D*2]; // block_size = 128 * 2 = 256, 256-point NTT

    // 1 thread brings 2 coeff 
    shared_coeff[threadIdx.x] = a[index]; // take the first coeff for the bfly
    shared_coeff[threadIdx.x+128] = a[index+ Nper2]; // take the second coeff for the bfly, the paired coeff

    // wait until all threads finish
    __syncthreads();

    // -----perform 255 point NTT-----
    #pragma unroll
    for (int length = 128 ; length >= 1 ; length>>=1, l_m<<=1, log_g_m++, log_l_m++){
        // compute the local butterfly indices
        int i = threadIdx.x >> log_l_m;
        int bfly_i1 = (i*l_m*2) + (threadIdx.x % l_m);
        int bfly_i2 = bfly_i1 + l_m; 

        // generate global twiddle factor
        int psi_idx = length + (index >> log_g_m);
        uint64_t twiddle = psi[psi_idx];

        // butterfly operation
        barrett_GS_butterfly(shared_coeff[bfly_i1], shared_coeff[bfly_i2], twiddle, q, q2, mu, qbit);

        // wait for all threads to complete their butterfly operation
        __syncthreads();
    }
    // -----return the coefficients to the global memory-----
    a[index] = shared_coeff[threadIdx.x];
    a[index + Nper2] = shared_coeff[threadIdx.x + 128];
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
    HIP_ASSERT(hipFree(GPUArray));
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

void gpuNTT(uint64_t *h_input, uint64_t logN, int N, int L, uint64_t* moduli, int qbit) {
    
    //myNTT CPU_in(logN, 62);

    // need to do ntt for each limb in rns;
    
    // do ntt 
}