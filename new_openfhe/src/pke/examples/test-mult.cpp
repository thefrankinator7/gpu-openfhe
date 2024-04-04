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

    uint32_t multDepth = 5;
    uint32_t scaleModSize = 20;
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

    // Step 3: Encoding and encryption of inputs

    // Inputs
    
    std::vector<double> x1 = {1,2,3,4,5,6,7,8};
    std::vector<double> x2 = {1,2,1,2,1,2,1,2};

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

    std::cout << "Input x1: " << ptxt1 << std::endl;
    std::cout << "Input x2: " << ptxt2 << std::endl;

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    auto c1_raw=GetRawCipherText(cc, c1);
    std::cout << c1_raw.numRes << std::endl;
    auto c2_raw=GetRawCipherText(cc, c2);
    std::cout << "Moving C1 to GPU" << std::endl;

    MoveToGPU(&c1_raw);
    std::cout << "Moving C2 to GPU" << std::endl;
    MoveToGPU(&c2_raw);

    std::cout << "Multiplying on GPU" << std::endl;

    // auto cMult=EvalMultGPU(&c1_raw, &c2_raw);
    EvalMultGPUNoRelin(&c1_raw, &c2_raw);

    auto start = std::chrono::high_resolution_clock::now();
    auto cMultCPU=cc->EvalMultNoRelin(c1,c2);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout.precision(8);
    std::cout << "mult time " << duration << "us" << std::endl;

    // Check results
    
    auto cpuMultRaw=GetRawCipherText(cc, cMultCPU);

    for (int i=0; i<10; i++) {
      std::cout << cpuMultRaw.sub_0[i] << std::endl << std::endl;
      std::cout << c1_raw.sub_0[i] << std::endl << std::endl;
    }

    // // Decrypt the result of multiplication

    Plaintext result;
    std::cout.precision(8);
    cc->Decrypt(keys.secretKey, cMultCPU, &result);
    result->SetLength(batchSize);
    std::cout << "x1 * x2 = " << result;
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
}
