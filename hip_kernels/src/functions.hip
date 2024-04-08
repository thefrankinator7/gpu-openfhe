#include "functions.hpp"

int is_power_of_two(uint64_t n){
    /* return true only if n is a power of two */
    return !(n&(n-1));
}

uint64_t gcd(uint64_t a, uint64_t b){
    /* returns the greatest common divisor of integers a and b
    implementation: Euclidean Algorithm */
    if(b==0)    
        return a;
    else        
        return gcd(b,a%b);
}

uint64_t int_sqrt(uint64_t k){
    /* returns the integer square root of k
    implementation: Newton's method */
    uint64_t x = k;
    uint64_t y = floor((x + 1)/2);
    while(y<x){
        x = y;
        y = floor((x + floor(k/x))/2);
    }
    return x;
}

int is_prime(uint64_t x){ 
    // x is assumed to be greater than 2
    // check if x is even
    if(x%2==0)  return 0;
    uint64_t divisor, sqrt_x_plusOne = int_sqrt(x);
    // check for all possible odd divisor
    for (divisor=3; divisor<sqrt_x_plusOne; divisor+=2)
        if (x%divisor==0) return 0;
    return 1;
}

uint64_t mult_mod_naive(uint64_t a, uint64_t b, uint64_t q){
    return (uint64_t)(((__uint128_t)a * (__uint128_t)b) % q); //naive implementation of modular multiplication
}

uint64_t exp_mod_naive(uint64_t base, uint64_t exp, const uint64_t q){
    /* 
    modular exponentation using "square-and-multiply" algorithm + naive modular multiplication
    OUTPUT : base^exp mod q
    */
    uint64_t result = 1&exp ? base : 1;
    while (exp!=0){
        exp>>=1; //devide by 2 or right shift
        // square the base
        base = mult_mod_naive(base, base, q);
        // check if we need to multiply the result by current base
        if(1&exp){
            result = mult_mod_naive(base, result, q);
        }
    }
    return result;
}

uint64_t inverse_mod_naive(const uint64_t x, const uint64_t q){
    /* 
    INPUT   : q a prime, 0<x<q
    OUTPUT  : inverse of x modulo q

    NOTES:  1. Euler's Theorem states that x**(q-1) % q == 1
            2. uses exp_mod_naive
    */
    return exp_mod_naive(x, q-2, q);
}

int is_primitive(const uint64_t x, const uint64_t n, const uint64_t q){
    /* 
    INPUT   : q prime, n power of 2, with n dividing q-1, 0<x<q
    OUTPUT  : 1 if x is primitive nth root of unity modulo q, 0 otherwise 
    */
    return exp_mod_naive(x, n>>1, q)== q-1;
}

// modular addition
uint64_t add_mod(const uint64_t a, const uint64_t b, const uint64_t q){
    /* INPUT: modulus q; operands a,b<q
        OUTPUT: (a+b)%q
        SOURCE: Harvey's "Faster arithmetic..." demo code
        LINK: https://web.maths.unsw.edu.au/~davidharvey/index.php?page=research
    */
    uint64_t z = a + b;
    return z -= ((z >= q) ? q : 0);
}

uint64_t sub_mod(const uint64_t a, const uint64_t b, const uint64_t q) {
    return a<b ? (q+a)-b : a-b;
}

uint64_t bit_reverse(const uint64_t x, const uint64_t width){
    /*
    perform bit reverse with max width = 32 bit
    */
    uint64_t y = x, c = 32-width;
    // bit reverse 32 bit
    y = (((y & 0xaaaaaaaa) >> 1)|((y & 0x55555555) << 1 ));
    y = (((y & 0xcccccccc) >> 2)|((y & 0x33333333) << 2 ));
    y = (((y & 0xf0f0f0f0) >> 4)|((y & 0x0f0f0f0f) << 4 ));
    y = (((y & 0xff00ff00) >> 8)|((y & 0x00ff00ff) << 8 ));
    y = ((y >> 16)|(y << 16));
    // shift into width
    return (y >> c);
}

uint64_t gen_good_prime(const uint64_t n, uint64_t k, const uint64_t bit_width) {
  /* INPUT: n power of 2, k>0
     OUTPUT: kth "good prime" for n;
     i.e., the kth prime q with the property that n divides q-1
  */
  if (k==0)	// error, k should be positive
    return 0;
  uint64_t good_prime = ((uint64_t)1<<(bit_width-1))+1;
  if (is_prime(good_prime))
    k -= 1;
  while (k--) {
    good_prime += n;
    while (!is_prime(good_prime))
      good_prime += n;
  }
  return good_prime;
}

uint64_t gen_primitive_root(uint64_t n, const uint64_t q) {
    /* INPUT: q prime, n power of 2, with n dividing q-1, 0<x<q
        OUTPUT: a primitive nth root of unity modulo q
        NOTES: conditions on q and n above guarantee existence of primitive root;
        in particular, for any 0<i<q, i**((q-1)/n) is guaranteed to be a root.
    */
    assert(is_prime(q));            // q must be a prime
    assert(is_power_of_two(n));     // n must be a power of 2
    assert((q-1) % n == 0);         // n must divide q-1

    uint64_t root, i = 2, exp = (q-1)/n;
    n >>= 1;

    do {
        root = exp_mod_naive(i++, exp, q);	// generate a root
    } while (exp_mod_naive(root, n, q) == 1); // if root not primitive, get another
    return root;
}


// Neal's code ends

// my NTTs
// Note: CT is also known as DIT (decimation in time)
// Note: GS is also known as DIF (decimation in frequency)

uint64_t get_omega(const uint64_t stage, const uint64_t k, const uint64_t base_omega, const uint64_t q){
    /*
    calculate omega on the fly
    */
    return exp_mod_naive(base_omega, bit_reverse(k, stage), q);
}

void generate_psi_array(uint64_t *a_psi, const uint64_t psi, const uint64_t q, const uint64_t logn){
    /* CPU pre-compute: calculate psi and psi inverse array in bit reverse order */
    // number of elements
    uint64_t N = 1<<logn;
    for(uint64_t i=0 ; i < N ; i++){
        a_psi[i] = exp_mod_naive(psi, bit_reverse(i, logn), q);
    }
}

void generate_invpsi_array(uint64_t *a_inv_psi, const uint64_t psi, const uint64_t q, const uint64_t logn){
    /* CPU pre-compute: calculate psi and psi inverse array in bit reverse order */
    // number of elements
    uint64_t N = 1<<logn;
    uint64_t inv_psi = inverse_mod_naive(psi, q);
    for(uint64_t i=0 ; i < N ; i++){
        a_inv_psi[i] = exp_mod_naive(inv_psi, bit_reverse(i, logn), q);
    }
}

void ntt_ct_no_bo(uint64_t *a, const uint64_t omega_n, const uint64_t logn, const uint64_t q, const int max_print){
    /*
    generate bit-reverse order ntt of array a, using given parameters and CT's butterfly
    */
    uint64_t stage, k, j, i, omega, omega_stage, v, u;
    //get number of input
    const uint64_t n = 1<<logn;
    // # of butterflies in each block
    uint64_t m = n>>1;
    for(stage = 0 ; stage < logn ; stage++, m>>=1){ //# of butterflies in each block halves every stage
        i = 0; //box index in each stage
        omega_stage = exp_mod_naive(omega_n, pow(2, logn-stage-1), q);
        for(k = 0 ; k < n ; k += (m<<1), i++){
            // update omega value
            omega = get_omega(stage, i, omega_stage, q);
            for(j = 0 ; j < m ; j++){
                //do CT butterfly operation
                u = a[k+j];
                v = mult_mod_naive(a[k+j+m], omega, q);
                a[k+j] = add_mod(u, v, q);
                a[k+j+m] = sub_mod(u, v, q);
            }
        }
        std::cout<<"Stage "<<stage<<": "; 
        for(int i=0; i<max_print; i++){
            std::cout<<a[i]<<" ";
        }
        std::cout<<std::endl;
    }
}

void ntt_ct_nobo_merged(uint64_t *a, const uint64_t omega_n, const uint64_t psi_n, const uint64_t logn, const uint64_t q, const int max_print){
    /*
    generate bit-reverse order ntt of array a, using given parameters and CT's butterfly
    */
    uint64_t stage, k, j, i, omega, psi, twiddle, omega_stage, v, u;
    //get number of input
    const uint64_t n = 1<<logn;
    // # of butterflies in each block
    uint64_t m = n>>1;
    for(stage = 0 ; stage < logn ; stage++, m>>=1){ //# of butterflies in each block halves every stage
        i = 0; //box index in each stage
        omega_stage = exp_mod_naive(omega_n, pow(2, logn-stage-1), q);
        for(k = 0 ; k < n ; k += (m<<1), i++){
            // update omega value
            omega = get_omega(stage, i, omega_stage, q);
            psi = exp_mod_naive(psi_n, m, q);
            twiddle = mult_mod_naive(omega, psi, q);
            for(j = 0 ; j < m ; j++){
                //do CT butterfly operation
                u = a[k+j];
                v = mult_mod_naive(a[k+j+m], twiddle, q);
                a[k+j] = add_mod(u, v, q);
                a[k+j+m] = sub_mod(u, v, q);
            }
        }
        std::cout<<"Stage "<<stage<<": "; 
        for(int i=0; i<max_print; i++){
            std::cout<<a[i]<<" ";
        }
        std::cout<<std::endl;
    }
}

void intt_gs_bo_no(uint64_t *a, const uint64_t omega_n, const uint64_t logn, const uint64_t q, const int max_print){
    /*
    generate natural order intt of array a, using given parameters and GS's butterfly
    */
    uint64_t stage, k, j, i, omega, omega_stage, inv_omega, u;
    //get number of input
    const uint64_t n = 1<<logn;
    uint64_t m = 1;
    inv_omega = inverse_mod_naive(omega_n, q);
    // perform intt iteration
    for(stage = 0 ; stage < logn ; stage++, m<<=1){
        i = 0;
        omega_stage = exp_mod_naive(inv_omega, pow(2, stage), q);
        for(k = 0 ; k < n ; k += (m<<1), i++){
            //update omega
            omega = get_omega((logn-1)-stage, i, omega_stage, q);
            for(j = 0 ; j < m ; j++){
                u = sub_mod(a[k+j], a[k+j+m], q);
                a[k+j] = add_mod(a[k+j], a[k+j+m], q);
                a[k+j+m] = mult_mod_naive(u, omega, q);
            }
        }
        std::cout<<"Stage "<<stage<<": "; 
        for(int i=0; i<max_print; i++){
            std::cout<<a[i]<<" ";
        }
        std::cout<<std::endl;
    }
}

void intt_gs_bono_merged(uint64_t *a, const uint64_t omega_n, const uint64_t psi_n, const uint64_t logn, const uint64_t q, const int max_print){
    /*
    generate natural order intt of array a, using given parameters and GS's butterfly
    */
    uint64_t stage, k, j, i, omega, psi, omega_stage, inv_omega, twiddle, inv_psi, u;
    //get number of input
    const uint64_t n = 1<<logn;
    uint64_t m = 1;
    inv_omega = inverse_mod_naive(omega_n, q);
    inv_psi = inverse_mod_naive(psi_n, q);
    // perform intt iteration
    for(stage = 0 ; stage < logn ; stage++, m<<=1){
        i = 0;
        omega_stage = exp_mod_naive(inv_omega, pow(2, stage), q);
        for(k = 0 ; k < n ; k += (m<<1), i++){
            //update omega
            omega = get_omega((logn-1)-stage, i, omega_stage, q);
            psi = exp_mod_naive(inv_psi, m, q);
            twiddle = mult_mod_naive(omega, psi, q);
            for(j = 0 ; j < m ; j++){
                u = sub_mod(a[k+j], a[k+j+m], q);
                a[k+j] = add_mod(a[k+j], a[k+j+m], q);
                a[k+j+m] = mult_mod_naive(u, twiddle, q);
            }
        }
        std::cout<<"Stage "<<stage<<": "; 
        // printf("\nStage %d: ", stage);
        for(int i=0; i<max_print; i++){
            std::cout<<a[i]<<" ";
            // printf("%d ", a[i]);
        }
        std::cout<<std::endl;
    }
    // multiply by inverse n
    for(uint64_t z=0; z<n;z++){
        a[z] = mult_mod_naive(a[z], inverse_mod_naive(n,q), q);
    }
}