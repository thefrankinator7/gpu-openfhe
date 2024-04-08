#include "hip/hip_runtime.h"
#include <stdio.h>
#include "functions.hpp"
#include <hip/hip_runtime.h>
#include "debugger.hpp"

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

__device__ __forceinline__ void classic_barrett(uint128_t& a, uint64_t& q, uint64_t& mu, int& qbit)  // Lee
{  
    uint128_t rx;
    rx = a >> (qbit - 1);
    rx *= mu;
    rx >>= qbit + 1;
    rx *= q;
    a -= rx;
    // first conditional subtraction
    a -= q * (a >= (uint128_t)q);
    // second conditional subtraction
    a -= q * (a >=(uint128_t)q);
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

// Neals barrett kernels ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void NTT_barrett_single(uint64_t *a, uint64_t *psi, uint64_t q, uint64_t mu, int qbit, int logn)
{
    // butterfly index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // memory index
    int global_id = blockIdx.x * 2048 + threadIdx.x;
    int stage_offset = logn - 11;

    // variables
    int bfly_index1, bfly_index2, i;
    uint64_t psi_index, twiddle, u, v, temp;
    uint128_t temp_mult;
    
    int m = 1024; // step size
    int logm = 10;
    
    // shared memory
    __shared__ uint64_t shared_coeff[BLOCK_SIZE*2];

    // fill the shared memory with 2048 elements
    shared_coeff[threadIdx.x] = a[global_id];
    shared_coeff[threadIdx.x + 1024] = a[global_id + 1024];

    // syncthreads
    __syncthreads();

    int start_length = 1 << stage_offset;
    int stop_length = 1 << logn;
 
    #pragma unroll
    for (int length = start_length; length < stop_length; length<<=1, m>>=1, logm--)
    {   // iterations for ntt
        i = threadIdx.x >> logm; // box index
        bfly_index1 = (i * m * 2) + (threadIdx.x % m);
        bfly_index2 = bfly_index1 + m;

        // generate twiddle factor
        psi_index = length + (index >> logm);
        twiddle = psi[psi_index];
        
        // temporary variables
        u = shared_coeff[bfly_index1];
        temp_mult = (uint128_t)shared_coeff[bfly_index2] * twiddle;

        // Barrett reduction
        neals_barrett(temp_mult, q, mu, qbit);

        // cast to uint64_t
        v = (uint64_t)temp_mult;  

        // butterfly results
        temp = u + v;
        shared_coeff[bfly_index1] = temp - q * (temp >= q);
        shared_coeff[bfly_index2] = q*(u<v) + (u-v);

        __syncthreads();
    }

    // return data to the global memory
    a[global_id] = shared_coeff[threadIdx.x] ;
    a[global_id + 1024] = shared_coeff[threadIdx.x + 1024];

}

__global__ void INTT_barrett_single(uint64_t *a, uint64_t *inv_psi, uint64_t q, uint64_t mu, int qbit, int logn_min1)
{
    // butterfly index
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // memory index
    int global_id = blockIdx.x*2048 + threadIdx.x;
    int stage_offset = logn_min1-10;

    int bfly_index1, bfly_index2, i;
    uint64_t psi_index, twiddle, u, v, temp;
    uint128_t temp_mult;

    int m = 1; // step size
    int logm = 0;

    uint64_t q2 = (q + 1) >> 1;
    
    // share memory
    __shared__ uint64_t shared_coeff[BLOCK_SIZE*2];

    // fill the shared memory with 2048 elements
    shared_coeff[threadIdx.x] = a[global_id];
    shared_coeff[threadIdx.x + 1024] = a[global_id + 1024];

    // syncthreads
    __syncthreads();

    int start_length = 1<<logn_min1;
    int stop_length = 1<<stage_offset;

    #pragma unroll
    for (int length = start_length; length >= stop_length ; length>>=1, m<<=1, logm++)
    {   // iterations for intt
        i = threadIdx.x >> logm;
        bfly_index1 = (i * m * 2) + (threadIdx.x % m);
        bfly_index2 = bfly_index1 + m;

        // generate twiddle factor
        psi_index = length + (index >> logm);
        twiddle = inv_psi[psi_index];
        
        u = shared_coeff[bfly_index1];
        v = shared_coeff[bfly_index2];

        // butterfly top output
        temp = u + v;
        temp -= q * (temp >= q);
        shared_coeff[bfly_index1] = (temp >> 1) + q2*(temp & 1);
        
        // butterfly lower output
        u += q*(u<v);
        temp_mult = (uint128_t)(u - v);
        temp_mult *= twiddle;
        neals_barrett(temp_mult, q, mu, qbit); // Barrett reduction
        temp = (uint64_t)temp_mult;
        shared_coeff[bfly_index2] = (temp >> 1) + q2*(temp & 1);

        __syncthreads();
    }

    // return data to the global memory
    a[global_id] = shared_coeff[threadIdx.x] ;
    a[global_id + 1024] = shared_coeff[threadIdx.x + 1024];
}

__global__ void NTT_barrett_multi(uint64_t *a, uint64_t *psi, uint64_t q, uint64_t mu, int qbit, int m, int logm, int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index >> logm;
    int bfly_index1 = (i * m * 2) + (index % m);
    int bfly_index2 = bfly_index1 + m;

    uint64_t psi_index = length + i; // width = stage
    uint64_t twiddle = psi[psi_index];
    
    // temporary variables
    uint64_t u = a[bfly_index1];
    uint128_t temp_mult = (uint128_t)a[bfly_index2] * twiddle;

    // Barrett reduction
    neals_barrett(temp_mult, q, mu, qbit);

    // cast to uint64_t
    uint64_t v = (uint64_t)temp_mult;  

    // butterfly results
    uint64_t temp = u + v;
    a[bfly_index1] = temp - q * (temp >= q);
    a[bfly_index2] = q*(u<v) + (u-v);
}

__global__ void INTT_barrett_multi(uint64_t *a, uint64_t *inv_psi, uint64_t q, uint64_t mu, int qbit, int m, int logm, int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index >> logm;
    int bfly_index1 = (i * m * 2) + (index % m);
    int bfly_index2 = bfly_index1 + m;

    uint64_t q2 = (q + 1) >> 1;

    uint64_t psi_index = length + i; // width = logn - stage - 1
    uint64_t twiddle = inv_psi[psi_index];
    
    uint64_t u = a[bfly_index1];
    uint64_t v = a[bfly_index2];

    // butterfly top output
    uint64_t temp = u + v;
    temp -= q * (temp >= q);
    a[bfly_index1] = (temp >> 1) + q2*(temp & 1);

    // butterfly lower output
    u += q*(u<v);
    uint128_t temp_mult = (uint128_t)(u - v);
    temp_mult *= (uint128_t)twiddle;
    neals_barrett(temp_mult, q, mu, qbit); // Barrett reduction
    temp = (uint64_t)temp_mult;
    a[bfly_index2] = (temp >> 1) + q2*(temp & 1);
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


// GPU function call ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void barrett_test(uint64_t *h_input, uint64_t *h_ntt, uint64_t *h_intt, uint64_t logN, uint64_t psi, uint64_t q, uint64_t mu, int qbit) 
{   
    // calculate number of coefficient
    const int N = 1<<logN;
    // calculate number of butterfly
    const int num_of_btfly = N>>1;
    // number of thread per block = 1024
    const int threadsPerBlock = BLOCK_SIZE;
    // calculate number of block
    const int blocks = (num_of_btfly + threadsPerBlock - 1) / threadsPerBlock;
    // cuda runtime
    hipEvent_t start, stop;
    gpuErrchk(hipEventCreate(&start));
    gpuErrchk(hipEventCreate(&stop));
    // memory size
    const int mem_size_N = sizeof(uint64_t)*N;

    // CPU pre-compute psi array
    uint64_t *h_psi = (uint64_t*)malloc(mem_size_N);
    uint64_t *h_inv_psi = (uint64_t*)malloc(mem_size_N);
    
    std::chrono::steady_clock::time_point psiGen_begin = std::chrono::steady_clock::now();
    generate_psi_array(h_psi, psi, q, logN);
    generate_invpsi_array(h_inv_psi, psi, q, logN);
    std::chrono::steady_clock::time_point psiGen_end = std::chrono::steady_clock::now();
    std::cout << "Psi array generation = " << std::chrono::duration_cast<std::chrono::microseconds>(psiGen_end - psiGen_begin).count() << "[µs]" << std::endl;
    std::cout << "                     = " << std::chrono::duration_cast<std::chrono::nanoseconds> (psiGen_end - psiGen_begin).count() << "[ns]" << std::endl;
    
    uint64_t *d_input = nullptr;
    uint64_t *d_psi = nullptr;
    uint64_t *d_inv_psi = nullptr;

    // Allocate memory 
    printf("Allocating memory...\n");
    // memory for coefficients
    gpuErrchk(hipMalloc((void **)&d_input, mem_size_N));
    // memory for psi and inverse psi
    gpuErrchk(hipMalloc((void **)&d_psi, mem_size_N));
    gpuErrchk(hipMalloc((void **)&d_inv_psi, mem_size_N));

    // copy input and psi array to device memory
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)d_input, (void *)h_input, mem_size_N, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void *)d_psi, (void *)h_psi, mem_size_N, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void *)d_inv_psi, (void *)h_inv_psi, mem_size_N, hipMemcpyHostToDevice));

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("Mem copy input      = %.3f [µs]\n", msecTotal*1000);
    
    int switch_point = 1<<(logN - 11);
    int m = N>>1;
    int logm = logN-1;

    // launch multi kernel approach
    printf("Performing NTT:\n");

    gpuErrchk(hipEventRecord(start, NULL));

    for(int length = 1 ; length < switch_point; length<<=1, m>>=1, logm--){
        NTT_barrett_multi<<<blocks, threadsPerBlock>>>(d_input, d_psi, q, mu, qbit, m, logm, length); // width = stage
        gpuErrchk(hipDeviceSynchronize());
    }
    // launch single kernel to finish NTT comp
    NTT_barrett_single<<<blocks, threadsPerBlock>>>(d_input, d_psi, q, mu, qbit, logN);
    gpuErrchk(hipDeviceSynchronize());

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("NTT kernel(s)       = %.3f [µs]\n", msecTotal*1000);

    // output memory copy
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)h_ntt, (void *)d_input, mem_size_N, hipMemcpyDeviceToHost));

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("NTT out memcpy      = %.3f [µs]\n", msecTotal*1000);
    
    // print result
    printf("NTT result: ");
    for(int i = 0; i<5; i++){
        std::cout << h_ntt[i] << " ";
    }
    printf("\n\n");

    // launch single kernel INTT
    printf("Performing INTT:\n");
    m = 2048; // step size for multikernel
    logm = 11;
    switch_point>>=1;

    gpuErrchk(hipEventRecord(start, NULL));

    INTT_barrett_single<<<blocks, threadsPerBlock>>>(d_input, d_inv_psi, q, mu, qbit, logN-1);
    
    gpuErrchk(hipDeviceSynchronize());
    for(int length = switch_point ; length >= 1; length>>=1, m<<=1, logm++){
        INTT_barrett_multi<<<blocks, threadsPerBlock>>>(d_input, d_inv_psi, q, mu, qbit, m, logm, length);
        gpuErrchk(hipDeviceSynchronize());
    }
    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("iNTT kernel(s)      = %.3f [µs]\n", msecTotal*1000);
    
    // output memory copy
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)h_intt, (void *)d_input, mem_size_N, hipMemcpyDeviceToHost));
    
    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("iNTT out memcpy     = %.3f [µs]\n", msecTotal*1000);
    
    // print result
    printf("iNTT result: ");
    for(int i = 0; i<5; i++){
        std::cout << h_intt[i] << " ";  
    }
    printf("\n");
    
    gpuErrchk(hipFree(d_input));
    gpuErrchk(hipFree(d_psi));
    gpuErrchk(hipFree(d_inv_psi));
}

void NTT2D_test(uint64_t *h_input, uint64_t *h_ntt, uint64_t *h_intt, uint64_t logN, uint64_t psi, uint64_t q, uint64_t mu, int qbit) 
{   
    // calculate number of coefficient
    const int N = 1<<logN;
    // calculate number of butterfly
    const int num_of_btfly = N>>1;
    // number of thread per block = 1024
    const int threadsPerBlock = BLOCK_SIZE2D;
    // calculate number of block
    const int blocks = (num_of_btfly + threadsPerBlock - 1) / threadsPerBlock;
    // cuda runtime
    hipEvent_t start, stop;
    gpuErrchk(hipEventCreate(&start));
    gpuErrchk(hipEventCreate(&stop));
    // memory size
    const int mem_size_N = sizeof(uint64_t)*N;

    // CPU pre-compute psi array
    uint64_t *h_psi = (uint64_t*)malloc(mem_size_N);
    uint64_t *h_inv_psi = (uint64_t*)malloc(mem_size_N);
    
    std::chrono::steady_clock::time_point psiGen_begin = std::chrono::steady_clock::now();
    generate_psi_array(h_psi, psi, q, logN);
    generate_invpsi_array(h_inv_psi, psi, q, logN);
    std::chrono::steady_clock::time_point psiGen_end = std::chrono::steady_clock::now();
    std::cout << "Psi array generation = " << std::chrono::duration_cast<std::chrono::microseconds>(psiGen_end - psiGen_begin).count() << "[µs]" << std::endl;
    std::cout << "                     = " << std::chrono::duration_cast<std::chrono::nanoseconds> (psiGen_end - psiGen_begin).count() << "[ns]" << std::endl;
    
    uint64_t *d_input = nullptr;
    uint64_t *d_psi = nullptr;
    uint64_t *d_inv_psi = nullptr;

    // Allocate memory 
    printf("Allocating memory...\n");
    // memory for coefficients
    gpuErrchk(hipMalloc((void **)&d_input, mem_size_N));
    // memory for psi and inverse psi
    gpuErrchk(hipMalloc((void **)&d_psi, mem_size_N));
    gpuErrchk(hipMalloc((void **)&d_inv_psi, mem_size_N));

    // copy input and psi array to device memory
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)d_input, (void *)h_input, mem_size_N, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void *)d_psi, (void *)h_psi, mem_size_N, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy((void *)d_inv_psi, (void *)h_inv_psi, mem_size_N, hipMemcpyHostToDevice));

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    float msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("Mem copy input      = %.3f [µs]\n", msecTotal*1000);

    // launch 2D NTT kernels
    printf("Performing NTT:\n");

    gpuErrchk(hipEventRecord(start, NULL));
    
    NTT2D_1st<<<blocks, threadsPerBlock>>>(d_input, d_psi, q, mu, qbit);
    gpuErrchk(hipDeviceSynchronize());
    NTT2D_2nd<<<blocks, threadsPerBlock>>>(d_input, d_psi, q, mu, qbit);

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("NTT kernel(s)       = %.3f [µs]\n", msecTotal*1000);

    // output memory copy
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)h_ntt, (void *)d_input, mem_size_N, hipMemcpyDeviceToHost));

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("NTT out memcpy      = %.3f [µs]\n", msecTotal*1000);
    
    // print result
    printf("NTT result: ");
    for(int i = 0; i<5; i++){
        std::cout << h_ntt[i] << " ";
    }
    printf("\n\n");

    // launch 2D iNTT kernels
    printf("Performing INTT:\n");

    gpuErrchk(hipEventRecord(start, NULL));

    iNTT2D_1st<<<blocks, threadsPerBlock>>>(d_input, d_inv_psi, q, mu, qbit);
    gpuErrchk(hipDeviceSynchronize());
    iNTT2D_2nd<<<blocks, threadsPerBlock>>>(d_input, d_inv_psi, q, mu, qbit);

    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("iNTT kernel(s)      = %.3f [µs]\n", msecTotal*1000);
    
    // output memory copy
    gpuErrchk(hipEventRecord(start, NULL));

    gpuErrchk(hipMemcpy((void *)h_intt, (void *)d_input, mem_size_N, hipMemcpyDeviceToHost));
    
    gpuErrchk(hipEventRecord(stop, NULL));
    gpuErrchk(hipEventSynchronize(stop));
    msecTotal = 0.0f;
    gpuErrchk(hipEventElapsedTime(&msecTotal, start, stop));
    printf("iNTT out memcpy     = %.3f [µs]\n", msecTotal*1000);
    
    // print result
    printf("iNTT result: ");
    for(int i = 0; i<5; i++){
        std::cout << h_intt[i] << " ";  
    }
    printf("\n");
    
    gpuErrchk(hipFree(d_input));
    gpuErrchk(hipFree(d_psi));
    gpuErrchk(hipFree(d_inv_psi));
}

void printGPUInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d GPU devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        hipDeviceProp_t deviceProps;
        hipGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs                      : %d\n", deviceProps.multiProcessorCount);
        printf("   Total global mem         : %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   Shared mem per block     : %.0f B\n", static_cast<float>(deviceProps.sharedMemPerBlock));
        printf("   Total const mem          : %.0f B\n", static_cast<float>(deviceProps.totalConstMem));
        printf("   GPU Cap                 : %d.%d\n", deviceProps.major, deviceProps.minor);
        printf("   Max threads per block    : %d\n", deviceProps.maxThreadsPerBlock);
        printf("   Max threads dim (x,y,z)  : (%d, %d, %d)\n", deviceProps.maxThreadsDim[0],deviceProps.maxThreadsDim[1],deviceProps.maxThreadsDim[2]);
        printf("   Max grid size (x,y,z)    : (%d, %d, %d)\n", deviceProps.maxGridSize[0],deviceProps.maxGridSize[1],deviceProps.maxGridSize[2]);

    }
    printf("---------------------------------------------------------\n");
}
