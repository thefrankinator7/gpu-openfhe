//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2014-2022, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Simple examples for CKKS
 */

#define PROFILE

#include "openfhe.h"
#include "gpu_functions.h"
#include "rawciphertext.h"
#include "ciphertext-fwd.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <thread>

using namespace lbcrypto;

void addCiphertexts(int i, int j, 
                    std::vector<RawCipherText> &allCipherTexts) {
    EvalAddGPU(&allCipherTexts[i], &allCipherTexts[j]);
}

void addOpenFHECiphertexts(int i, int j, const CryptoContext<DCRTPoly> cc, 
                    const std::vector<Ciphertext<DCRTPoly>> &allOpenFHECipherTexts) {
    auto temp = cc->EvalAdd(allOpenFHECipherTexts[i], allOpenFHECipherTexts[j]);
    temp = cc->EvalAdd(allOpenFHECipherTexts[i], temp);
    
}

void multCiphertexts(int i, int j, 
                    std::vector<RawCipherText> &allCipherTexts) {
    EvalMultGPUNoRelin(&allCipherTexts[i], &allCipherTexts[j]);

}

void multOpenFHECiphertexts(int i, int j, const CryptoContext<DCRTPoly> cc, 
                    const std::vector<Ciphertext<DCRTPoly>> &allCipherTexts) {
    auto temp = cc->EvalMult(allCipherTexts[i], allCipherTexts[j]);
}

void launchEmptyKernel() {
    gpuEmptyKernel();
}


int main() {

    for (int i=0; i<32; i++) {
        gpuEmptyKernel();
    }

    // multithreading
    int numThreads = std::thread::hardware_concurrency(); // Use as many threads as there are cores, or choose your number.
    std::cout<< "Working with " << numThreads << " Threads" << std::endl;
    std::vector<std::thread> threads;

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    long long duration;

    hipSync();
    for (int i=1; i<=16; i++) {
        int numVectors = std::pow(2, i);
        std::cout << "Initializing " << numVectors << " \"vectors\" for 2^" << i << std::endl;

        hipSync();
        start = std::chrono::high_resolution_clock::now();

        // launch kernels

        int numOps=numVectors/2;
        for (int i = 0; i < numOps; ++i) {
            // Launch thread
            gpuEmptyKernel();
        }

        hipSync();
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout.precision(8);

        std::cout << "Number of Kernels: " << numOps << std::endl;
        std::cout << "total time " << duration << "us" << std::endl;
    }
    

    // loop over powers of 2, eventually crashes around i=12
    for (int i=1; i<=16; i++) {
        int numVectors = std::pow(2, i);
        std::cout << "Initializing " << numVectors << " \"vectors\" for 2^" << i << std::endl;

        hipSync();
        start = std::chrono::high_resolution_clock::now();

        // launch kernels

        int numOps=numVectors/2;
        for (int i = 0; i < numOps; ++i) {
            // Launch thread
            threads.emplace_back(gpuEmptyKernel);
        }

        for (auto &thread : threads) {
            thread.join();
        }
        hipSync();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout.precision(8);

        std::cout << "Number of Kernels: " << numOps << std::endl;
        std::cout << "total time " << duration << "us" << std::endl;

        threads.clear();

        hipSync();

    }

    return 0;
}
