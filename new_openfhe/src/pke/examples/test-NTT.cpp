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

void testNTT(int qbit, uint32_t L, int logN) {
    uint32_t multDepth = L;
    uint32_t scaleModSize = qbit;
    uint32_t batchSize = 8;
    uint32_t ringDim=1 << logN;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    parameters.SetRingDim(ringDim);
    parameters.SetSecurityLevel(HEStd_NotSet);


    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    std::cout << "Log N = " << logN << "  Log q = "<< qbit << "  L = " << L << std::endl << std::endl;
    std::cout.precision(8);

    auto keys = cc->KeyGen();

    // Step 3: Encoding and encryption of inputs

    // Inputs
    
    std::vector<double> x1 = {1,2,3,4,5,6,7,8};
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);    

    auto start = std::chrono::high_resolution_clock::now();
    auto cv = c1->GetElements();
    for (auto& c : cv) {
        c.SetFormat(Format::COEFFICIENT);
    }
    c1->SetElements(cv);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_intt_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "iNTT time on CPU: " << duration << "us" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    
    cv = c1->GetElements();
    for (auto& c : cv) {
        c.SetFormat(Format::EVALUATION);
    }
    c1->SetElements(cv);

    end = std::chrono::high_resolution_clock::now();
    auto cpu_ntt_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "NTT time on CPU: " << duration << "us" << std::endl;
    
    cv = c1->GetElements();
    for (auto& c : cv) {
        c.SetFormat(Format::COEFFICIENT);
    }
    c1->SetElements(cv);

    auto c1_raw=GetRawCipherText(cc, c1);
    //std::cout << "Moving C1 to GPU" << std::endl;
    MoveToGPU(&c1_raw);
    

    //std::cout << "Copying NTT params and moving to GPU" << std::endl;
    NTT_params params = get_NTT_params(&c1_raw, logN, qbit);
    hipSync();
    //std::cout << "Doing NTT  on GPU" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    GPU_NTT(&c1_raw, params);
    hipSync();
    end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "NTT time " << duration << "us" << std::endl;


    start = std::chrono::high_resolution_clock::now();
    
    GPU_INTT(&c1_raw, params);
    hipSync();
    end = std::chrono::high_resolution_clock::now();
    auto gpu_intt_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "iNTT time on GPU: " << duration << "us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    
    GPU_NTT(&c1_raw, params);
    hipSync();
    end = std::chrono::high_resolution_clock::now();
    auto gpu_ntt_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << "NTT time on GPU: " << duration << "us" << std::endl;

    float avg_speedup = ( (float) (cpu_ntt_duration+cpu_intt_duration) ) / ((float) (gpu_ntt_duration+gpu_intt_duration));
    std::cout << "Average speedup: " << avg_speedup << "x" << std::endl;
}

int main() {
    std::vector<int> logNs = {15,16,17};
    std::vector<int> qbits = {30, 40, 50};
    std::vector<int> Ls= {5, 10, 20, 40};
    // log q, L, log N
    for (auto& logN : logNs) {
        for (auto& qbit : qbits) {
            for (auto& L : Ls) {
                testNTT(qbit, L, logN);
            }
        }
    }
    
}
