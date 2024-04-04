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
#include "ciphertext-fwd.h"
#include "gpu_functions.h"
#include "rawciphertext.h"

using namespace lbcrypto;

void benchOps(int logq, int L, int ringDim) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(L);
    parameters.SetScalingModSize(logq);
    parameters.SetRingDim(ringDim);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    std::cout << "Parameters:" << std::endl << std::endl;
    std::cout << "N = " << cc->GetRingDimension() << std::endl;
    std::cout << "log q = " << logq << std::endl;
    std::cout << "L = " << L << std::endl;

    auto keys = cc->KeyGen();

    std::vector<double> x1 = {1,2,3,4,5,6,7,8};
    std::vector<double> x2 = {1,2,1,2,1,2,1,2};

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto c1_raw=GetRawCipherText(cc, c1);
    auto c2_raw=GetRawCipherText(cc, c2);
    MoveToGPU(&c1_raw);
    MoveToGPU(&c2_raw);


    // GPU Addition 

    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0;i<10;i++) {
    EvalAddGPU(&c1_raw, &c2_raw);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);
    std::cout << "Avg GPU add time " << duration/10 << "us" << std::endl;
    

    // CPU Addition
    
    start = std::chrono::high_resolution_clock::now();

    for (int i=0;i<10;i++) {
    auto cAdd=cc->EvalAdd(c1,c2);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);
    std::cout << "Avg CPU add time " << duration/10 << "us" << std::endl;


    // Reset raw ciphertexts
    c1_raw=GetRawCipherText(cc, c1);
    c2_raw=GetRawCipherText(cc, c2);
    MoveToGPU(&c1_raw);
    MoveToGPU(&c2_raw);


    // GPU Multiplication

    start = std::chrono::high_resolution_clock::now();

    for (int i=0;i<10;i++) {
    EvalMultGPUNoRelin(&c1_raw, &c2_raw);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);
    std::cout << "Avg GPU mult time " << duration/10 << "us" << std::endl;


    // CPU Multiplication

    start = std::chrono::high_resolution_clock::now();

    for (int i=0;i<10;i++) {
    auto cMultCPU=cc->EvalMultNoRelin(c1,c2);
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);
    std::cout << "Avg CPU mult time " << duration/10 << "us" << std::endl;

    MoveToHost(&c1_raw);
    MoveToHost(&c2_raw);
}
int main() {
    
    benchOps(50, 40, 131072);
    
}
