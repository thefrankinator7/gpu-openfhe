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

int main() {

    uint32_t multDepth = 40;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 8;
    //uint32_t ringDim=1 << 16;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    //parameters.SetRingDim(ringDim);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    auto keys = cc->KeyGen();

    cc->EvalMultKeyGen(keys.secretKey);

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

    // CKKS breakdown - Individual Ciphertext
    // two sub-ciphertexts - ct0 and ct1, polynomials in original format mod large Q
    // each sub-ciphertext is implemented by a vector of vectors, each of which represents an rns limb of the ciphertext, a polynomial with a small q modulus
    

    // Individual polynomials of each ciphertext
    // type should be vector<PolyImpl<vector<int>>>
    // ciphertext -> get sub-ciphertexts -> access idx -> get all numRes of each sub-ciphertext
    // each limb has m_data coefficients (integers) and an m_modulus modulus

    std::cout << "Converting c1 to raw, then move to GPU and back and back" << std::endl;

    auto c1_raw=GetRawCipherText(cc, c1);
    std::cout << c1_raw.numRes << std::endl;
    auto c2_raw=GetRawCipherText(cc, c2);
    std::cout << "Moving C1 to GPU" << std::endl;

    MoveToGPU(&c1_raw);
    std::cout << "Moving C2 to GPU" << std::endl;
    MoveToGPU(&c2_raw);
  
    std::cout << "Adding on GPU" << std::endl;
    hipSync();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0;i<100;i++) {
    EvalAddGPU(&c1_raw, &c2_raw);
    }
    hipSync();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/100;
    std::cout.precision(8);

    

    std::cout << "add time " << duration << "us" << std::endl;

    MoveToHost(&c1_raw);
    MoveToHost(&c2_raw);

    auto c1_back=GetOpenFHECipherText(c1_raw);

    std::cout << "Adding" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto cAdd = cc->EvalAdd(c1_back, c2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);

    std::cout << "add time " << duration << "us" << std::endl;


    Plaintext result;
    std::cout.precision(8);

    
    // auto cAdd_raw=GetRawCipherText(cc, cAdd);
    // auto cAdd_back=GetOpenFHECipherText(cAdd_raw);

    // auto cAdd_2=EvalAddGPU(c1_raw,c2_raw);
    // auto cAdd_2_back=GetOpenFHECipherText(cAdd_raw);
    

    // // Decrypt the result of addition
    cc->Decrypt(keys.secretKey, cAdd, &result);
    result->SetLength(batchSize);
    std::cout << "x1 + x2 = " << result;
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;

    // Contains both elements of the ciphertext
    // auto c_elems=cAdd->GetElements();

    // // get one element (polynomial)
    // auto ct0=c_elems[0];

    // // rns vectors
    // auto vectors=ct0.GetAllElements();

    // // single rns vector
    // auto vector=vectors[0];

    // auto coef1=(*vector.m_values)[0];
    // std::cout << coef1;
}
