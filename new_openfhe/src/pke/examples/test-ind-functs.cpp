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

//#include "vec_add.h"

using namespace lbcrypto;

#define EVALUATION 0;
#define COEFFICIENT 1;

/*
* The rawCipherText Class contains the basic information needed to hold a ciphertext, with pointers to data that can be contained on the GPU or main memory
* This is stored in RNS/DCRT format
*/
struct RawCipherText {
  CryptoContext<DCRTPoly> cc; // Original CryptoContext object from OpenFHE;
  Ciphertext<DCRTPoly> originalCipherText; // Original CipherText object from OpenFHE;
  uint64_t* sub_0; // pointer to sub-ciphertext 0
  uint64_t* sub_1; // pointer to sub-ciphertext 1
  uint64_t* moduli; // moduli for each limb
  int limbs; // number of limbs of ciphertext, length of moduli array and first dimension of sub-ciphertexts
  int N; // length of each polynomial
  bool format;
};

uint64_t* GetRawArray(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    // total size is l * N
    int numLimbs=polys.size();
    int numElements=(*polys[0].m_values).GetLength();
    auto totalSize = numLimbs * numElements;
    // Allocate array
    uint64_t* flattened = new uint64_t[totalSize];

    // Fill the array
    for (int l=0;l<numLimbs;l++) {
      for (int i=0;i<numElements;i++) {
            flattened[l * numElements + i] = (*polys[l].m_values)[i].ConvertToInt();
        }
    }
    return flattened;
};

uint64_t* GetModuli(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    int numLimbs=polys.size();
    uint64_t* moduli=new uint64_t[numLimbs];
    for (int l=0;l<numLimbs;l++) {
        moduli[l]=polys[l].GetModulus().ConvertToInt();
    }
    return moduli;
};

RawCipherText GetRawCipherText(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ct) {
    RawCipherText result;
    result.cc=cc;
    result.originalCipherText=ct;
    result.limbs=ct->GetElements()[0].GetAllElements().size();
    result.N=(*(ct->GetElements()[0].GetAllElements())[0].m_values).GetLength();
    result.sub_0 = GetRawArray(ct->GetElements()[0].GetAllElements());
    result.sub_1 = GetRawArray(ct->GetElements()[1].GetAllElements());
    result.moduli = GetModuli(ct->GetElements()[0].GetAllElements());
    result.format= ct->GetElements()[0].GetFormat();

    return result;
};

Ciphertext<DCRTPoly> GetOpenFHECipherText(RawCipherText ct) {
    auto result = ct.originalCipherText;
    auto sub_0=result->GetElements()[0];
    auto sub_1=result->GetElements()[1];
    auto dcrt_0=sub_0.GetAllElements();
    auto dcrt_1=sub_1.GetAllElements();
    for (int l=0;l<ct.limbs;l++) {
        for (int i=0; i<ct.N; i++) {
            (*dcrt_0[l].m_values)[i].SetValue(ct.sub_0[l*ct.N + i]);
            (*dcrt_1[l].m_values)[i].SetValue(ct.sub_1[l*ct.N + i]);
        }
    }
    sub_0.m_vectors=dcrt_0;
    sub_1.m_vectors=dcrt_1;
    std::vector<lbcrypto::DCRTPoly> ct_new= { sub_0, sub_1};
    result->SetElements(ct_new);

    return result;
};

void MoveToGPU(RawCipherText ct) {
    ct.sub_0=moveArrayToGPU(ct.sub_0, ct.N*ct.limbs);
    ct.sub_1=moveArrayToGPU(ct.sub_1, ct.N*ct.limbs);
};

void MoveToHost(RawCipherText ct) {
    ct.sub_0=moveArrayToHost(ct.sub_0, ct.N*ct.limbs);
    ct.sub_1=moveArrayToHost(ct.sub_1, ct.N*ct.limbs);
};

RawCipherText EvalAddGPU(RawCipherText ct1, RawCipherText ct2) {
    for (int l=0;l<ct1.limbs;l++) {
        gpuAdd(&ct1.sub_0[l*ct1.N],&ct2.sub_0[l*ct1.N],&ct1.sub_0[l*ct1.N], ct1.N, ct1.moduli[l]);
        gpuAdd(&ct1.sub_1[l*ct1.N],&ct2.sub_1[l*ct1.N],&ct1.sub_1[l*ct1.N], ct1.N, ct1.moduli[l]);
    }
    return ct1;
};

int main() {

    uint32_t multDepth = 1;
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
    // ciphertext -> get sub-ciphertexts -> access idx -> get all limbs of each sub-ciphertext
    // each limb has m_data coefficients (integers) and an m_modulus modulus

    std::cout << "Converting c1 to raw, then move to GPU and back and back" << std::endl;

    auto c1_raw=GetRawCipherText(cc, c1);
    MoveToGPU(c1_raw);
    MoveToHost(c1_raw);
    auto c2_raw=GetRawCipherText(cc, c2);
    auto c1_back=GetOpenFHECipherText(c1_raw);

    std::cout << "Adding" << std::endl;
    auto cAdd = cc->EvalAdd(c1_back, c2);

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
