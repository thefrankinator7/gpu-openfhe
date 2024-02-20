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
//#include "vec_add.h"

using namespace lbcrypto;


int main() {
    // Step 1: Setup CryptoContext

    // A. Specify main parameters
    /* A1) Multiplicative depth:
   * The CKKS scheme we setup here will work for any computation
   * that has a multiplicative depth equal to 'multDepth'.
   * This is the maximum possible depth of a given multiplication,
   * but not the total number of multiplications supported by the
   * scheme.
   *
   * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
   * a multiplicative depth of 1, but requires a total of 3 multiplications.
   * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
   * either as a computation of multiplicative depth 3 as
   * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
   * as g(x_i) = (x1*x2)*(x3*x4).
   *
   * For performance reasons, it's generally preferable to perform operations
   * in the shorted multiplicative depth possible.
   */
    uint32_t multDepth = 1;

    /* A2) Bit-length of scaling factor.
   * CKKS works for real numbers, but these numbers are encoded as integers.
   * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
   * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
   * integer). Say the result of a computation based on m' is 130, then at
   * decryption, the scaling factor is removed so the user is presented with
   * the real number result of 0.13.
   *
   * Parameter 'scaleModSize' determines the bit-length of the scaling
   * factor D, but not the scaling factor itself. The latter is implementation
   * specific, and it may also vary between ciphertexts in certain versions of
   * CKKS (e.g., in FLEXIBLEAUTO).
   *
   * Choosing 'scaleModSize' depends on the desired accuracy of the
   * computation, as well as the remaining parameters like multDepth or security
   * standard. This is because the remaining parameters determine how much noise
   * will be incurred during the computation (remember CKKS is an approximate
   * scheme that incurs small amounts of noise with every operation). The
   * scaling factor should be large enough to both accommodate this noise and
   * support results that match the desired accuracy.
   */
    uint32_t scaleModSize = 50;

    /* A3) Number of plaintext slots used in the ciphertext.
   * CKKS packs multiple plaintext values in each ciphertext.
   * The maximum number of slots depends on a security parameter called ring
   * dimension. In this instance, we don't specify the ring dimension directly,
   * but let the library choose it for us, based on the security level we
   * choose, the multiplicative depth we want to support, and the scaling factor
   * size.
   *
   * Please use method GetRingDimension() to find out the exact ring dimension
   * being used for these parameters. Give ring dimension N, the maximum batch
   * size is N/2, because of the way CKKS works.
   */
    uint32_t batchSize = 8;

    /* A4) Desired security level based on FHE standards.
   * This parameter can take four values. Three of the possible values
   * correspond to 128-bit, 192-bit, and 256-bit security, and the fourth value
   * corresponds to "NotSet", which means that the user is responsible for
   * choosing security parameters. Naturally, "NotSet" should be used only in
   * non-production environments, or by experts who understand the security
   * implications of their choices.
   *
   * If a given security level is selected, the library will consult the current
   * security parameter tables defined by the FHE standards consortium
   * (https://homomorphicencryption.org/introduction/) to automatically
   * select the security parameters. Please see "TABLES of RECOMMENDED
   * PARAMETERS" in  the following reference for more details:
   * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
   */
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

    // CKKS breakdown - Individual Ciphertext
    // two sub-ciphertexts - ct0 and ct1, polynomials in original format mod large Q
    // each sub-ciphertext is implemented by a vector of vectors, each of which represents an rns limb of the ciphertext, a polynomial with a small q modulus
    

    // Individual polynomials of each ciphertext
    // type should be vector<PolyImpl<vector<int>>>
    // ciphertext -> get sub-ciphertexts -> access idx -> get all limbs of each sub-ciphertext
    // each limb has m_data coefficients (integers) and an m_modulus modulus


    auto ct10=c1->GetElements()[0];
    auto ct11=c1->GetElements()[1];
    auto ct20=c2->GetElements()[0];
    auto ct21=c2->GetElements()[1];

    auto ct10_dcrt=ct10.GetAllElements();
    auto ct11_dcrt=ct11.GetAllElements();
    auto ct20_dcrt=ct20.GetAllElements();
    auto ct21_dcrt=ct21.GetAllElements();
    
    auto ct10_mod=ct10_dcrt[0].m_params->GetModulus();
    auto ct20_mod=ct20_dcrt[0].m_params->GetModulus();
    std::cout << ct10_mod << std::endl;
    std::cout << ct20_mod << std::endl;

    int numLimbs=ct10_dcrt.size();
    int numElements=(*ct10_dcrt[0].m_values).GetLength();

    std::cout << "Got parameters of ciphertexts" << std::endl;
    
    // Each is vector of vectors for each sub-ciphertext
    std::vector<std::vector<uint64_t>> ct10_vals(numLimbs);
    std::vector<std::vector<uint64_t>> ct11_vals(numLimbs);
    std::vector<std::vector<uint64_t>> ct20_vals(numLimbs);
    std::vector<std::vector<uint64_t>> ct21_vals(numLimbs);

    // each is vector of moduli for each sub-ciphertext
    std::vector<uint64_t> ct10_mods(numLimbs);
    std::vector<uint64_t> ct11_mods(numLimbs);
    std::vector<uint64_t> ct20_mods(numLimbs);
    std::vector<uint64_t> ct21_mods(numLimbs);

    for (int l=0;l<numLimbs;l++) {
      for (int i=0;i<numElements;i++) {
        ct10_vals[l].push_back((*ct10_dcrt[l].m_values)[i].ConvertToInt());
        ct11_vals[l].push_back((*ct11_dcrt[l].m_values)[i].ConvertToInt());
        ct20_vals[l].push_back((*ct20_dcrt[l].m_values)[i].ConvertToInt());
        ct21_vals[l].push_back((*ct21_dcrt[l].m_values)[i].ConvertToInt());
      }
      ct10_mods.push_back(ct10_dcrt[l].GetModulus().ConvertToInt());
    }

    std::cout << ct10_vals[0][0] << std::endl;
    std::cout << ct10_vals[0][0] << std::endl;

    // Switch Values
    // Add values
    for (int l=0;l<numLimbs;l++) {
      for (int i=0;i<numElements;i++) {
        ct20_vals[l][i]+=ct10_vals[l][i];
        ct21_vals[l][i]+=ct11_vals[l][i];
        (*ct10_dcrt[l].m_values)[i].SetValue(ct20_vals[l][i]);
        (*ct11_dcrt[l].m_values)[i].SetValue(ct21_vals[l][i]);
      }
    }
    std::cout << "Adding outside then inside openFHE" << std::endl;

    ct10.m_vectors=ct10_dcrt;
    ct11.m_vectors=ct11_dcrt;
    std::vector<lbcrypto::DCRTPoly> c1_new= { ct10, ct11};
    c1->SetElements(c1_new);
    // Add ciphertexts together:

    for (int l=0;l<numLimbs;l++) {
      for (int i=0;i<numElements;i++) {
        (*ct10_dcrt[l].m_values)[i]+=(*ct20_dcrt[l].m_values)[i] % ct10_dcrt[l].m_params->GetModulus();
        (*ct11_dcrt[l].m_values)[i]+=(*ct21_dcrt[l].m_values)[i] % ct11_dcrt[l].m_params->GetModulus();
        }
    }
    
    auto cAdd = cc->EvalAdd(c1, c2);

    Plaintext result;
    std::cout.precision(8);


    // // Decrypt the result of addition
    cc->Decrypt(keys.secretKey, cAdd, &result);
    result->SetLength(batchSize);
    std::cout << "x1 + x2 + x2= " << result;
    std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;

}
