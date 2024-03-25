#include "gpu_functions.h"
#include <iostream>

int main() {


    uint64_t* testArray=new uint64_t[16]();
    uint64_t* testArray2=new uint64_t[16]();
    uint64_t* testModuli=new uint64_t[2]();

    testModuli[0]=3;
    testModuli[1]=5;
    for (int i=0;i<16;i++) {
        testArray[i]=i;
        testArray2[i]=i;
        std::cout << testArray[i] << std::endl;
    }
    std::cout << testModuli[0] <<std::endl;
    std::cout << testModuli[1] << std::endl;
    // Testing moveArray, I think it works?
    testArray=moveArrayToGPU(testArray, 16);
    testArray2=moveArrayToGPU(testArray2, 16);
    testModuli=moveArrayToGPU(testModuli,2);
    gpuAdd(testArray, testArray2, testArray, 8, 2, testModuli);
    testArray=moveArrayToHost(testArray, 16);

    for (int i=0;i<16;i++) {
        std::cout << testArray[i] << std::endl;
    }

}