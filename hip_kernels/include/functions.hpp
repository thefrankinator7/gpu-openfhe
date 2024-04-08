#ifndef __CPU_FUNCTIONS__
#define __CPU_FUNCTIONS__

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>


#define uint128_t          unsigned __int128
#define DEFAULT_LOG2N   16
#define BLOCK_SIZE      1024
#define BLOCK_SIZE2D    128
#define Nper2           32768
#define logNper2        15

// define functions
int is_power_of_two(uint64_t n);
uint64_t gcd(uint64_t a, uint64_t b);
uint64_t int_sqrt(uint64_t k);

int is_prime(uint64_t x);

uint64_t mult_mod_naive(uint64_t a, uint64_t b, uint64_t q);
uint64_t exp_mod_naive(uint64_t base, uint64_t exp, const uint64_t q);
uint64_t inverse_mod_naive(const uint64_t x, const uint64_t q);

int is_primitive(const uint64_t x, const uint64_t n, const uint64_t q);

uint64_t add_mod(const uint64_t a, const uint64_t b, const uint64_t q);
uint64_t sub_mod(const uint64_t a, const uint64_t b, const uint64_t q);

uint64_t bit_reverse(const uint64_t x, const uint64_t width);
uint64_t gen_good_prime(const uint64_t n, uint64_t k, const uint64_t bit_width);
uint64_t gen_primitive_root(uint64_t n, const uint64_t q);
uint64_t get_omega(const uint64_t stage, const uint64_t k, const uint64_t base_omega, const uint64_t q);
void generate_psi_array(uint64_t *a_psi, const uint64_t psi, const uint64_t q, const uint64_t logn);
void generate_invpsi_array(uint64_t *a_inv_psi, const uint64_t psi, const uint64_t q, const uint64_t logn);


void ntt_ct_no_bo(  uint64_t *a, 
                    const uint64_t omega_n, 
                    const uint64_t logn, 
                    const uint64_t q, 
                    const int max_print);
void ntt_ct_nobo_merged(uint64_t *a, 
                        const uint64_t omega_n, 
                        const uint64_t psi_n, 
                        const uint64_t logn, 
                        const uint64_t q, 
                        const int max_print);
void intt_gs_bo_no( uint64_t *a, 
                    const uint64_t omega_n, 
                    const uint64_t logn, 
                    const uint64_t q, 
                    const int max_print);
void intt_gs_bono_merged(   uint64_t *a, 
                            const uint64_t omega_n, 
                            const uint64_t psi_n, 
                            const uint64_t logn, 
                            const uint64_t q, 
                            const int max_print);

#endif