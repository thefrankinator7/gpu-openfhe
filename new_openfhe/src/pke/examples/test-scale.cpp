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

// void multCiphertexts(int i, int j, const CryptoContext<DCRTPoly> cc, 
//                     const std::vector<Ciphertext<DCRTPoly>> &allCipherTexts) {
//     auto temp = cc->EvalMult(allCipherTexts[i], allCipherTexts[j]);
// }



int main() {
    uint32_t multDepth = 40;

    uint32_t scaleModSize = 50;

    uint32_t batchSize = 8;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

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

    /* B2) Generate the digit size
   * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
   * we get a result with some components that are valid under key s, and
   * with an additional component that's valid under key s^2.
   *
   * In most cases, we want to perform relinearization of the multiplicaiton
   * result, i.e., we want to transform the s^2 component of the ciphertext so
   * it becomes valid under original key s. To do so, we need to create what we
   * call a relinearization key with the following line.
   */
    cc->EvalMultKeyGen(keys.secretKey);

    /* B3) Generate the rotation keys
   * CKKS supports rotating the contents of a packed ciphertext, but to do so,
   * we need to create what we call a rotation key. This is done with the
   * following call, which takes as input a vector with indices that correspond
   * to the rotation offset we want to support. Negative indices correspond to
   * right shift and positive to left shift. Look at the output of this demo for
   * an illustration of this.
   *
   * Keep in mind that rotations work over the batch size or entire ring dimension (if the batch size is not specified).
   * This means that, if ring dimension is 8 and batch
   * size is not specified, then an input (1,2,3,4,0,0,0,0) rotated by 2 will become
   * (3,4,0,0,0,0,1,2) and not (3,4,1,2,0,0,0,0).
   * If ring dimension is 8 and batch
   * size is set to 4, then the rotation of (1,2,3,4) by 2 will become (3,4,1,2).
   * Also, as someone can observe
   * in the output of this demo, since CKKS is approximate, zeros are not exact
   * - they're just very small numbers.
   */
    cc->EvalRotateKeyGen(keys.secretKey, {1, -2});

    // Step 3: Encoding and encryption of inputs

    // Inputs
    // vector of c1 and c2, for loop running of evalAdd over vectors
    // will need to make it multithreaded
    
    std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> x2 = {5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25};

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

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

                // Optionally, process or display the vector here
                // For demonstration, we'll just show the first element of each vector
                // std::cout << "Vector " << j + 1 << ": " << vec[0] << "..." << std::endl;

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
            threads.emplace_back(addCiphertexts, i, numOps+i, std::ref(allCipherTexts));
        }

        for (auto &thread : threads) {
            thread.join();
        }
        hipSync();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout.precision(8);

        
        std::cout << "Number of Operations: " << numOps << std::endl;
        std::cout << "GPU Implementation: " << numOps << std::endl;
        std::cout << "add time " << duration << "ms" << std::endl;

        threads.clear();

        start = std::chrono::high_resolution_clock::now();

        // Add together

        for (int i = 0; i < numOps; ++i) {
            // Launch thread
            threads.emplace_back(addOpenFHECiphertexts, i, numOps+i, cc, std::ref(allOpenFHECipherTexts));
        }

        for (auto &thread : threads) {
            thread.join();
        }
        hipSync();

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout.precision(8);

        
        std::cout << "OpenFHE CPU: " << numOps << std::endl;
        std::cout << "add time " << duration << "ms" << std::endl;

        threads.clear();

    }

    return 0;
}
