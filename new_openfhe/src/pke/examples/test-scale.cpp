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
}

void multCiphertexts(int i, int j, 
                    std::vector<RawCipherText> &allCipherTexts) {
    EvalMultGPUNoRelin(&allCipherTexts[i], &allCipherTexts[j]);

}

void multOpenFHECiphertexts(int i, int j, const CryptoContext<DCRTPoly> cc, 
                    const std::vector<Ciphertext<DCRTPoly>> &allCipherTexts) {
    auto temp = cc->EvalMultNoRelin(allCipherTexts[i], allCipherTexts[j]);
}

void launchEmptyKernel() {
    gpuEmptyKernel();
}


int main() {
    uint32_t multDepth = 40;

    uint32_t scaleModSize = 50;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetRingDim(1 << 17);
    parameters.SetSecurityLevel(HEStd_NotSet);


    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    // B. Step 2: Key Generation
    /* B1) Generate encryption keys.
   * These are used for encryption/decryption, as well as in generating
   * different kinds of keys.
   */
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // Step 3: Encoding and encryption of inputs

    // Inputs
    // vector of c1 and c2, for loop running of evalAdd over vectors
    // will need to make it multithreaded
    
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distrib(1, 100); // Define the range

    // multithreading
    int numThreads = std::thread::hardware_concurrency(); // Use as many threads as there are cores, or choose your number.
    std::cout<< "Working with " << numThreads << " Threads" << std::endl;
    std::vector<std::thread> threads;

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    long long duration;
    // loop over powers of 2, eventually crashes around i=12
    for (int i=1; i<=16; i++) {
        int numVectors = std::pow(2, i);
        std::cout << "Initializing " << numVectors << " ciphertexts for 2^" << i << std::endl;

        std::vector<std::vector<double>> allVectors(numVectors);
        std::vector<RawCipherText> allCipherTexts(numVectors);
        std::vector<decltype(c1)> allOpenFHECipherTexts(numVectors);

        // Initialize ciphertexts
        for (int j = 0; j < numVectors; ++j) {
                std::vector<double> vec(8); // Create a vector of length 10
                for (double &val : vec) {
                    val = distrib(gen); // Assign random values to each element in the vector
                }

                allVectors[j]=vec;
                auto ct=cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(vec));
                auto ct_raw=GetRawCipherText(cc, ct);
                MoveToGPU(&ct_raw);
                allCipherTexts[j]=ct_raw;
                allOpenFHECipherTexts[j]=ct;
            }

        hipSync();
        start = std::chrono::high_resolution_clock::now();

        // Add together

        int numOps=numVectors/2;
        for (int i = 0; i < numOps; ++i) {
            // Launch thread
            threads.emplace_back(multCiphertexts, i, numOps+i, std::ref(allCipherTexts));
        }

        for (auto &thread : threads) {
            thread.join();
        }
        hipSync();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout.precision(8);

        
        std::cout << "Number of Operations: " << numOps << std::endl;
        std::cout << "GPU Implementation: " << std::endl;
        std::cout << "mult time " << duration << "ms" << std::endl;

        threads.clear();

        start = std::chrono::high_resolution_clock::now();

        // Add together
        for (int i = 0; i < numOps; ++i) {
            // Launch thread
            threads.emplace_back(multOpenFHECiphertexts, i, numOps+i, cc, std::ref(allOpenFHECipherTexts));
        }

        for (auto &thread : threads) {
            thread.join();
        }
        hipSync();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "OpenFHE CPU: " << std::endl;
        std::cout << "mult time " << duration << "ms" << std::endl;


        for (int j = 0; j < numVectors; ++j) {
                MoveToHost(&(allCipherTexts[j]));
        }

        threads.clear();
        hipSync();

    }

    return 0;
}
