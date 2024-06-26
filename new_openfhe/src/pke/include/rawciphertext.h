#include "openfhe.h"
#include "gpu_functions.h"

using namespace lbcrypto;

/*
* The rawCipherText Class contains the basic information needed to hold a ciphertext, with pointers to data that can be contained on the GPU or main memory
* This is stored in RNS/DCRT format
*/
struct RawCipherText {
  CryptoContext<DCRTPoly> cc; // Original CryptoContext object from OpenFHE;
  Ciphertext<DCRTPoly> originalCipherText; // Original CipherText object from OpenFHE;
  uint64_t* sub_0; // pointer to sub-ciphertext 0
  uint64_t* sub_1; // pointer to sub-ciphertext 1
  uint64_t* sub_2; // pointer to sub-ciphertext 1
  uint64_t* moduli; // moduli for each limb
  int numRes; // number of residues of ciphertext, length of moduli array and first dimension of sub-ciphertexts
  int N; // length of each polynomial
  Format format; // current format of ciphertext, either coefficient or evaluation
};

/**
* parameter structure for doing the NTT on the GPU in RNS format
*/
struct NTT_params {
    int N;
    int L;
    int qbit;
    int logN;
    uint64_t* moduli;
    uint64_t* mus;
    uint64_t* psi_arrays;
    uint64_t* inv_psi_arrays;
};


/**
* Converts a vector of polynomial limbs to a single flattened array 
*/
uint64_t* GetRawArray(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    // total size is r * N
    int numRes=polys.size();
    int numElements=(*polys[0].m_values).GetLength();
    auto totalSize = numRes * numElements;
    // Allocate array
    uint64_t* flattened = new uint64_t[totalSize];

    // Fill the array
    for (int r=0;r < numRes;r++) {
      for (int i=0;i<numElements;i++) {
            flattened[r * numElements + i] = (*polys[r].m_values)[i].ConvertToInt();
        }
    }
    return flattened;
};

/**
* Gets the moduli from a vector of polynomial limbs and returns a single array
*/
uint64_t* GetModuli(std::vector<lbcrypto::PolyImpl<lbcrypto::NativeVector>> polys) {
    int numRes=polys.size();
    uint64_t* moduli=new uint64_t[numRes];
    for (int r=0;r < numRes;r++) {
        moduli[r]=polys[r].GetModulus().ConvertToInt();
    }
    return moduli;
};

/**
* Converts a ciphertext from openFHE into the RawCiphertext format
*/
RawCipherText GetRawCipherText(CryptoContext<DCRTPoly> cc, Ciphertext<DCRTPoly> ct) {
    RawCipherText result;
    result.cc=cc;
    result.originalCipherText=ct;
    result.numRes=ct->GetElements()[0].GetAllElements().size();
    result.N=(*(ct->GetElements()[0].GetAllElements())[0].m_values).GetLength();
    result.sub_0 = GetRawArray(ct->GetElements()[0].GetAllElements());
    result.sub_1 = GetRawArray(ct->GetElements()[1].GetAllElements());
    result.moduli = GetModuli(ct->GetElements()[0].GetAllElements());
    result.format= ct->GetElements()[0].GetFormat();

    return result;
};

/**
* Converts a ciphertext from the RawCiphertext format back to the OpenFHE ciphertext format*/
Ciphertext<DCRTPoly> GetOpenFHECipherText(RawCipherText ct) {
    auto result = ct.originalCipherText;
    auto sub_0=result->GetElements()[0];
    auto sub_1=result->GetElements()[1];
    auto dcrt_0=sub_0.GetAllElements();
    auto dcrt_1=sub_1.GetAllElements();
    for (int r=0;r<ct.numRes;r++) {
        for (int i=0; i<ct.N; i++) {
            (*dcrt_0[r].m_values)[i].SetValue(ct.sub_0[r*ct.N + i]);
            (*dcrt_1[r].m_values)[i].SetValue(ct.sub_1[r*ct.N + i]);
        }
    }
    sub_0.m_vectors=dcrt_0;
    sub_1.m_vectors=dcrt_1;
    std::vector<lbcrypto::DCRTPoly> ct_new= { sub_0, sub_1};
    result->SetElements(ct_new);

    return result;
};

/** 
* Moves a ciphertext to the GPU for computations
*/
void MoveToGPU(RawCipherText* ct) {
    int numElems=ct->N * ct->numRes;
    ct->sub_0=moveArrayToGPU(ct->sub_0, numElems);
    ct->sub_1=moveArrayToGPU(ct->sub_1, numElems);
    ct->sub_2=moveArrayToGPU(ct->sub_1, numElems);
    ct->moduli=moveArrayToGPU(ct->moduli, ct->numRes);
};

/**
* Moves a ciphertext back to main CPU memory
*/
void MoveToHost(RawCipherText* ct) {
    int numElems=ct->N * ct->numRes;
    ct->sub_0=moveArrayToHost(ct->sub_0, numElems);
    ct->sub_1=moveArrayToHost(ct->sub_1, numElems);
    ct->sub_2=moveArrayToHost(ct->sub_2, numElems);
    ct->moduli=moveArrayToHost(ct->moduli, ct->numRes);
};

/**
* Evaluates homomorphic addition on the GPU
*/
void EvalAddGPU(RawCipherText* ct1, RawCipherText* ct2) {
    gpuAdd(ct1->sub_0, ct2->sub_0, ct1->sub_0, ct1->N, ct1->numRes, ct1->moduli);
    gpuAdd(ct1->sub_1, ct2->sub_1, ct1->sub_1, ct1->N, ct1->numRes, ct1->moduli);
    //hipSync();
};


void EvalMultGPUNoRelin(RawCipherText* ct1, RawCipherText* ct2) {

    gpuMult(ct1->sub_0, ct2->sub_1, ct1->sub_2, ct1->N, ct1->numRes, ct1->moduli);
    gpuMult(ct1->sub_1, ct2->sub_0, ct2->sub_2, ct1->N, ct1->numRes, ct1->moduli);

    gpuMult(ct1->sub_0, ct2->sub_0, ct1->sub_0, ct1->N, ct1->numRes, ct1->moduli);
    gpuMult(ct1->sub_1, ct2->sub_1, ct1->sub_2, ct1->N, ct1->numRes, ct1->moduli);

    gpuAdd(ct1->sub_2,ct2->sub_2, ct1->sub_1,ct1->N,ct1->numRes,ct1->moduli);
    // hipSync();
}

NTT_params get_NTT_params(RawCipherText* ct1, int logN, int qbit) {
    NTT_params params;
    params.N=ct1->N;
    params.L=ct1->numRes;
    params.qbit = qbit;
    params.logN = logN;
    params.moduli=moveArrayToGPU(ct1->moduli, params.L);

    params.mus= (uint64_t*)malloc(sizeof(uint64_t)*params.L);
    params.psi_arrays = (uint64_t*)malloc(sizeof(uint64_t) * params.N * params.L);
    params.inv_psi_arrays = (uint64_t*)malloc(sizeof(uint64_t) * params.N * params.L);

    uint64_t psi;
    for (int i=0; i<params.L; i++) {
        params.mus[i]=((__uint128_t)1<<(2*params.qbit)+1) / ct1->moduli[i];
        psi = gen_primitive_root(2*params.N, ct1->moduli[i]);
        generate_psi_array(params.psi_arrays+ i * params.N, psi, ct1->moduli[i], logN);
        generate_invpsi_array(params.inv_psi_arrays + i * params.N, psi, ct1->moduli[i], logN);
    }
    
    params.mus= moveArrayToGPU(params.mus, params.L);
    params.psi_arrays = moveArrayToGPU(params.psi_arrays, params.N * params.L);
    params.inv_psi_arrays = moveArrayToGPU(params.inv_psi_arrays, params.N * params.L);

    return params;
}

void GPU_NTT(RawCipherText* ct1, NTT_params params) {
    if (ct1->format==EVALUATION) {
        std::cout << "Already in Evaluation Format" << std::endl;
        return;
    }
    gpuNTT(ct1->sub_0, params.psi_arrays, params.logN, params.N, params.L, params.moduli, params.mus, params.qbit);
    gpuNTT(ct1->sub_1, params.psi_arrays, params.logN, params.N, params.L, params.moduli, params.mus, params.qbit);
    ct1->format = EVALUATION;
}

void GPU_INTT(RawCipherText* ct1, NTT_params params) {
    if (ct1->format==COEFFICIENT) {
        std::cout << "Already in Coefficient Format" << std::endl;
        return;
    }
    gpuINTT(ct1->sub_0, params.inv_psi_arrays, params.logN, params.N, params.L, params.moduli, params.mus, params.qbit);
    gpuINTT(ct1->sub_1, params.inv_psi_arrays, params.logN, params.N, params.L, params.moduli, params.mus, params.qbit);
    ct1->format = COEFFICIENT;
}
/**
* Evaluates homomorphic multiplication on the GPU
* does not include relinearization and rescaling */
// Ciphertext<DCRTPoly> EvalMultGPU(RawCipherText* ct1, RawCipherText* ct2) {
//     int numElems=ct1->N * ct1->numRes;

//     auto d0=moveArrayToGPU(ct1->sub_0,numElems);
//     auto d1=moveArrayToGPU(ct1->sub_0,numElems);
    
//     auto d1_temp=moveArrayToGPU(ct1->sub_0,numElems);
//     auto d2=moveArrayToGPU(ct1->sub_0,numElems);

//     auto start = std::chrono::high_resolution_clock::now();

//     // gpuMult(ct1->sub_0, ct2->sub_1, ct1->sub_2, ct1->N, ct1->numRes, ct1->moduli);
//     // gpuMult(ct1->sub_1, ct2->sub_0, ct2->sub_2, ct1->N, ct1->numRes, ct1->moduli);

//     // gpuMult(ct1->sub_0, ct2->sub_0, ct1->sub_0, ct1->N, ct1->numRes, ct1->moduli);
//     // gpuMult(ct1->sub_1, ct2->sub_1, ct1->sub_2, ct1->N, ct1->numRes, ct1->moduli);

//     // gpuAdd(ct1->sub_2,ct2->sub_2, ct1->sub_1,ct1->N,ct1->numRes,ct1->moduli);
    
//     // transition back to OpenFHE for relinearization
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//     std::cout.precision(8);
//     std::cout << "mult time " << duration << "us" << std::endl;
    
//     gpuMult(ct1->sub_0, ct2->sub_0, d0, ct1->N, ct1->numRes, ct1->moduli);

//     gpuMult(ct1->sub_0, ct2->sub_1, d1, ct1->N, ct1->numRes, ct1->moduli);
//     gpuMult(ct1->sub_1, ct2->sub_0, d1_temp, ct1->N, ct1->numRes, ct1->moduli);
//     gpuAdd(d1,d1_temp,d1,ct1->N,ct1->numRes,ct1->moduli);

//     gpuMult(ct1->sub_1, ct2->sub_1, d2, ct1->N, ct1->numRes, ct1->moduli);
    
//     hipSync();
    
    
//     auto result = ct1->originalCipherText;
//     auto sub_0=result->GetElements()[0];
//     auto sub_1=result->GetElements()[1];
//     auto sub_2=sub_0.Clone();

//     auto dcrt_0=sub_0.GetAllElements();
//     auto dcrt_1=sub_1.GetAllElements();
//     auto dcrt_2=sub_2.GetAllElements();
//     for (int r=0;r<ct1->numRes;r++) {
//         for (int i=0; i<ct1->N; i++) {
//             (*dcrt_0[r].m_values)[i].SetValue(d0[r*ct1->N + i]);
//             (*dcrt_1[r].m_values)[i].SetValue(d1[r*ct1->N + i]);
//             (*dcrt_2[r].m_values)[i].SetValue(d2[r*ct1->N + i]);
//         }
//     }

//     sub_0.m_vectors=dcrt_0;
//     sub_1.m_vectors=dcrt_1;
//     sub_2.m_vectors=dcrt_2;
    
//     std::vector<lbcrypto::DCRTPoly> ct_new= { sub_0, sub_1, sub_2};
//     result->SetElements(ct_new);

//     result->SetNoiseScaleDeg(result->GetNoiseScaleDeg() + result->GetNoiseScaleDeg());
//     result->SetScalingFactor(result->GetScalingFactor() * result->GetScalingFactor());
//     const auto plainMod = result->GetCryptoParameters()->GetPlaintextModulus();
//     result->SetScalingFactorInt(
//         result->GetScalingFactorInt().ModMul(result->GetScalingFactorInt(), plainMod));


//     //auto evk=ct1->cc->GetEvalMultKeyVector(ct1->originalCipherText->GetKeyTag())[0];

//     //Relinearize

//     // pass original ciphertext object for parameters
//     // ab = gpuKeySwitch(ct1->originalCipherText->GetElements()[0],d2, ct1->N, ct1->numRes, evk);

//     // add d0 + ab[0]
//     // add d1 + ab[1]

//     return result;
// }

/**
* c is passed for parameters, the actual operations are performed on d2 with N and L
* EvalKey probably needs to be put on GPU as well, I believe it has the same parameters N and L?
* 
*/
// void gpuKeySwitch(const DCRTPoly& c, uint64_t* d2, int N, int L, const EvalKey<DCRTPoly> evalKey){
//     // precompute, I think this is breaking down into digits
//     std::shared_ptr<CryptoParametersBase<DCRTPoly>> cryptoParamsBase=evk->GetCryptoParameters();


//     const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cryptoParamsBase);

//     const std::shared_ptr<ParmType> paramsQl  = c.GetParams();
//     const std::shared_ptr<ParmType> paramsP   = cryptoParams->GetParamsP();
//     const std::shared_ptr<ParmType> paramsQlP = c.GetExtendedCRTBasis(paramsP);

//     size_t sizeQl  = paramsQl->GetParams().size();
//     size_t sizeP   = paramsP->GetParams().size();
//     size_t sizeQlP = sizeQl + sizeP;

//     uint32_t alpha = cryptoParams->GetNumPerPartQ();
//     // The number of digits of the current ciphertext
//     uint32_t numPartQl = ceil((static_cast<double>(sizeQl)) / alpha);
//     if (numPartQl > cryptoParams->GetNumberOfQPartitions())
//         numPartQl = cryptoParams->GetNumberOfQPartitions();

//     /***********/
//     // Digit decomposition
//     // Zero-padding and split
//     std::vector<DCRTPoly> partsCt(numPartQl);

//     for (uint32_t part = 0; part < numPartQl; part++) {
//         if (part == numPartQl - 1) {
//             auto paramsPartQ = cryptoParams->GetParamsPartQ(part);

//             uint32_t sizePartQl = sizeQl - alpha * part;

//             std::vector<NativeInteger> moduli(sizePartQl);
//             std::vector<NativeInteger> roots(sizePartQl);

//             for (uint32_t i = 0; i < sizePartQl; i++) {
//                 moduli[i] = paramsPartQ->GetParams()[i]->GetModulus();
//                 roots[i]  = paramsPartQ->GetParams()[i]->GetRootOfUnity();
//             }

//             auto params = DCRTPoly::Params(paramsPartQ->GetCyclotomicOrder(), moduli, roots, {}, {}, 0);

//             partsCt[part] = DCRTPoly(std::make_shared<ParmType>(params), Format::EVALUATION, true);
//         }
//         else {
//             partsCt[part] = DCRTPoly(cryptoParams->GetParamsPartQ(part), Format::EVALUATION, true);
//         }

//         usint sizePartQl   = partsCt[part].GetNumOfElements();
//         usint startPartIdx = alpha * part;
//         for (uint32_t i = 0, idx = startPartIdx; i < sizePartQl; i++, idx++) {
//             partsCt[part].SetElementAtIndex(i, c.GetElementAtIndex(idx));
//         }
//     }


//     // FastKeySwitchCore, I think this is processing each digit

//     const std::shared_ptr<std::vector<DCRTPoly>> digits= // result of precompute 
//     const EvalKey<DCRTPoly> evalKey=evk;
//     const std::shared_ptr<ParmType> paramsQl=ct1->originalCipherText->GetElements()[0].GetParams();

//     //CoreExt???? Huh???
// }



// void GPUNTT(RawCipherText* ct) {
//     if (ct->format==EVALUATION) {
//         std::cout << "Already in Evaluation Format" << std::endl;
//     }
//     else {
//         // get twiddles
//         // do gpuNTT
//         return;
//     }
// }

// void GPUiNTT(RawCipherText* ct) {
//     if (ct->format==COEFFICIENT) {
//         std::cout << "Already in Coefficient Format" << std::endl;
//     }
//     else {
//         // get twiddles
//         // do gpuiNTT
//         return;
//     }
// }


