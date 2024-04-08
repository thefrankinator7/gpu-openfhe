#ifndef __NTT_CLASS__
#define __NTT_CLASS__


#include "functions.hpp"

class myNTT 
{
    public:
        int n;
        int logn;
        int qbit;
        uint64_t q;
        uint64_t omega;
        uint64_t psi;
        uint64_t mu;
        uint64_t *data;
        uint64_t *data_old;
        int status; // 0 : zero array, 1: time domain, 2: NTT domain, 
        // NTT constructor
        myNTT (uint64_t x, uint64_t q_bit){
            status = 0;
            // log n
            logn = x;
            // n is a power of two
            n = 1<<x;
            // prime bit
            qbit = q_bit;
            // generate good prime
            q = gen_good_prime(2*n, 2, qbit);
            // precompute mu for barrett reduction
            mu = ((__uint128_t)1<<(2*qbit)+1) / q; 
            // generate psi
            psi = gen_primitive_root(2*n, q);
            // generate omega
            omega = mult_mod_naive(psi, psi, q);
            // initialize coefficient memory with 0
            data = (uint64_t*)calloc(n, sizeof(uint64_t));
            data_old = (uint64_t*)calloc(n, sizeof(uint64_t));
        }
        void insert(uint64_t *new_poly, int num, int domain);
        // perform NTT
        void do_NTT();
        // perform iNTT
        void do_iNTT();
        void print_value(int k);
        void print_info();
        void init_value();
        void half_ones();
        void rand_init(int s);
        bool validate();
};

// NTT class function
void myNTT::insert(uint64_t *new_poly, int num, int domain){
    /* insert value into NTT object */
    status = domain;
    for(int i=0; i<num; i++){
        data[i] = new_poly[i];
    }
}
void myNTT::do_NTT(){
    status = 2;
    ntt_ct_nobo_merged(data, omega, psi, logn, q, 5);
}
void myNTT::do_iNTT(){
    status = 1;
    intt_gs_bono_merged(data, omega, psi, logn, q, 5);
}
void myNTT::print_value(int k){
    std::cout<<"Value : ";
    for(int i = 0; i < k; i++){
        std::cout<<" "<<data[i];
    }
    std::cout<<std::endl;
}

void myNTT::print_info(){
    std::cout<<"NTT Parameters:"<<std::endl;
    std::cout<<"  logn  = "<<logn<<std::endl;
    std::cout<<"  psi   = "<<psi<<std::endl;
    std::cout<<"  q     = "<<q<<std::endl;
}

void myNTT::init_value(){
    /* insert value into NTT object */
    for(int i=0; i<n; i++){
        data[i] = i;
    }
}

void myNTT::half_ones(){
    for(int i = 0; i<(n>>1); i++){
        data[i] = 1;
    }
}

void myNTT::rand_init(int s){
    // seed for the random number generator
    srand(s);
    
    // generate random number
    for(int i = 0; i<n; i++){
        data[i] = rand() % q;
        data_old[i] = data[i];
    }
}

bool myNTT::validate(){
    for(int i = 0; i<n; i++){
        if(data_old[i] != data[i]){
            return false;
        }
    }
    return true;
}

#endif
