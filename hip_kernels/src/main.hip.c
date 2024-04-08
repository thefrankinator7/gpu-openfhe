#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "functions.hpp"
#include "ntt_class.hpp"
#include "debugger.hpp"

void barrett_test(uint64_t *h_input, uint64_t *h_ntt, uint64_t *h_intt, uint64_t logN, uint64_t psi, uint64_t q, uint64_t mu, int qbit);
void NTT2D_test(uint64_t *h_input, uint64_t *h_ntt, uint64_t *h_intt, uint64_t logN, uint64_t psi, uint64_t q, uint64_t mu, int qbit); 

void printGPUInfo();

void print_help(const char* progname){
    printf("Usage: %s [option1] [option1 value]...\n", progname);
    printf("Run NTT INTT kernels for builtin, out barrett, and classic barrett\n");
    printf("Program options and option values:\n");
    printf("  -e  <INT between 11 and 17>               Change size exponent (default=%d)\n", DEFAULT_LOG2N);
    // printf("  -p  <negacyclic/identity/ntt/nfused>      Select operation to run (default=identity with sequential input)\n");
    // printf("  -c                                        Check correctness of CUDA output against CPU reference\n");
    printf("  -?                                        This message\n");
}

int main(int argc, char** argv)
{
    std::string program;
    int i = 1;
    // int compare_with_CPU = 0;
    
    int logN = 16;
    
    // Reading the arguments and setup the configurations
    while(i<argc){
        if(argv[i][0] != '-'){
            std::cout<<"Error: Invalid argument -> "<< argv[i] <<std::endl;
            break; //break out the for loop
        }
        switch(argv[i][1]){
            // case 'c': /* compare correctness with CPU */
            //     compare_with_CPU = 1;
            //     i++; //next argument
            //     break;
            case 'e': /* input size exponent*/
                // check correctness
                i++;
                logN = std::stoi(argv[i]);
                assert(logN>10 && logN<18);
                i++; // next argument
                break;
            // case 'p':
            //     i++;
            //     if( (std::string(argv[i]).compare("ntt")==0) ||
            //         (std::string(argv[i]).compare("identity")==0) ||
            //         (std::string(argv[i]).compare("nfused")==0) ||
            //         (std::string(argv[i]).compare("negacyclic")==0)) {
            //         program = argv[i];
            //     } else {
            //         printf("Error: Unknown operation type: %s\n", argv[i]);
            //         print_help(argv[0]);
            //         return 1;
            //     }
            //     i++; // next argument
            //     break;
            // case '?':
            default:
                print_help(argv[0]);
                return 1;
        }
    }
    
    printGPUInfo();
    int count = 0;
    int count2 = 0;
    int N = 1<<logN;

    uint64_t *input_data = (uint64_t*)malloc(sizeof(uint64_t)*N);
    uint64_t *barrett_ntt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    uint64_t *barrett_intt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    uint64_t *barrett_2Dntt = (uint64_t*)malloc(sizeof(uint64_t)*N);
    uint64_t *barrett_2Dintt = (uint64_t*)malloc(sizeof(uint64_t)*N);

    myNTT CPU_in(logN, 62);

    // initiate random input
    CPU_in.rand_init(time(0));
    
    // copy input to all gpu input
    for(int j = 0; j < N ; j++){
        input_data[j] = CPU_in.data[j];
    }

    printf("Input value:\n");
    CPU_in.print_value(5);

    // NTT INTT using our barrett
    printf("-------------------------------------------------------------------------------------------------\n");
    printf("NTT-INTT (OUR BARRETT)...\n");
    barrett_test(input_data, barrett_ntt, barrett_intt, CPU_in.logn, CPU_in.psi, CPU_in.q, CPU_in.mu, CPU_in.qbit);
    printf("-------------------------------------------------------------------------------------------------\n");
    // NTT INTT using 2D NTT
    printf("NTT-INTT (2D NTT)...\n");
    NTT2D_test(input_data, barrett_2Dntt, barrett_2Dintt, CPU_in.logn, CPU_in.psi, CPU_in.q, CPU_in.mu, CPU_in.qbit);
    printf("-------------------------------------------------------------------------------------------------\n");
    printf("COMPARING RESULT TO CPU IMPLEMENTATION...\n");
    CPU_in.do_NTT();
    CPU_in.print_value(5);
    // check for NTT correctness
    for(int j = 0; j<N; j++){
        if(CPU_in.data[j]!=barrett_ntt[j]){
            count++;
        }
        if(CPU_in.data[j]!=barrett_2Dntt[j]){
            count2++;
        }
    }
    if(count>0){
        printf("\nBarrett NTT Kernel is INCORRECT (GPU <> CPU -> %d)\n", count);
    } else {
        printf("\nBarrett NTT Kernel is CORRECT (GPU = CPU)\n");
    }
    if(count2>0){
        printf("\nBarrett 2D NTT Kernel is INCORRECT (GPU <> CPU -> %d)\n", count2);
    } else {
        printf("\nBarrett 2D NTT Kernel is CORRECT (GPU = CPU)\n\n");
    }

    count = 0;
    count2 = 0;
    CPU_in.do_iNTT();
    CPU_in.print_value(5);
    for(int j = 0; j<N; j++){
        if(CPU_in.data[j]!=barrett_intt[j]){
            count++;
        }
        if(CPU_in.data[j]!=barrett_2Dintt[j]){
            count2++;
        }
    }
    if(count>0){
        printf("\nBarrett INTT Kernel is INCORRECT (GPU <> CPU -> %d)\n", count);
    } else {
        printf("\nBarrett INTT Kernel is CORRECT (GPU = CPU)\n");
    }
    if(count2>0){
        printf("\nBarrett 2D iNTT Kernel is INCORRECT (GPU <> CPU -> %d)\n", count2);
    } else {
        printf("\nBarrett 2D iNTT Kernel is CORRECT (GPU = CPU)\n\n");
    }

    if(CPU_in.validate()){
        printf("CPU works correctly\n"); 
    }else{
        printf("CPU fails\n");
    }

    CPU_in.print_info();

    return 1;
}
