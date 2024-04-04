#include "gpu_functions.h"
#include <iostream>

int main() {

    int N=1024;
    int L=2;
    int numElems=N*L;

    uint64_t* testArray=new uint64_t[numElems]();
    uint64_t* testArray2=new uint64_t[numElems]();
    uint64_t* testModuli=new uint64_t[L]();

    testModuli[0]=3;
    testModuli[1]=5;
    for (int i=0;i<numElems;i++) {
        testArray[i]=i;
        testArray2[i]=2;
    }
    
    std::cout << testModuli[0] <<std::endl;
    std::cout << testModuli[1] << std::endl;
    // Testing moveArray, I think it works?
    testArray=moveArrayToGPU(testArray, numElems);
    testArray2=moveArrayToGPU(testArray2, numElems);
    testModuli=moveArrayToGPU(testModuli, L);
    gpuMult(testArray, testArray2, testArray, N, L, testModuli);
    testArray=moveArrayToHost(testArray, numElems);

    for (int i=0;i<16;i++) {
        std::cout << testArray[i] << std::endl;
    }

    for (int i=0;i<16;i++) {
        std::cout << testArray[N+i] << std::endl;
    }
}