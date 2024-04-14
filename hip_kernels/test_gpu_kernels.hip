#include "gpu_functions.h"
#include <iostream>
// #include "functions.hpp"
// #include "ntt_class.hpp"
// #include "debugger.hpp" 

int main() {

    int N=1024;
    int L=2;
    int numElems=N*L;

    uint64_t* testArray=new uint64_t[numElems]();
    uint64_t* testArray2=new uint64_t[numElems]();
    uint64_t* testModuli=new uint64_t[L]();

    testModuli[0]=10;
    testModuli[1]=2050;
    for (int i=0;i<numElems;i++) {
        testArray[i]=i;
        testArray2[i]=i;
    }
    
    std::cout << testModuli[0] <<std::endl;
    std::cout << testModuli[1] << std::endl;
    // Testing moveArray, I think it works?
    testArray=moveArrayToGPU(testArray, numElems);
    testArray2=moveArrayToGPU(testArray2, numElems);
    testModuli=moveArrayToGPU(testModuli, L);
    gpuAdd(testArray, testArray2, testArray, N, L, testModuli);
    testArray=moveArrayToHost(testArray, numElems);

    for (int i=0;i<16;i++) {
        std::cout << testArray[i] << std::endl;
    }

    for (int i=0;i<16;i++) {
        std::cout << testArray[N+i] << std::endl;
    }


    // int logN = 16;

    // int N = 1<<logN;

    // uint64_t *input_data = (uint64_t*)malloc(sizeof(uint64_t)*N);
    // uint64_t *barrett_ntt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    // uint64_t *barrett_intt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    // uint64_t *barrett_2Dntt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    // uint64_t *barrett_2Dintt = (uint64_t*)malloc(sizeof(uint64_t)*N);

    // int qbit=62;
    // myNTT CPU_in(logN, qbit);
}