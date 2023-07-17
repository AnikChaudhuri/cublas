#include <cublas_v2.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

int main(){
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    // for simplicity we are going to use square arrays
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 2;
    int m, n, k, lda, ldb, ldc; 
    m = n = k = lda = ldb = ldc= nr_rows_A;
    
    thrust::device_vector<double> d_A(nr_rows_A * nr_cols_A), d_B(nr_rows_B * nr_cols_B), d_C(nr_rows_C * nr_cols_C);
    thrust::sequence(d_A.begin(), d_A.end(),1);
    thrust::sequence(d_B.begin(), d_B.end(),5);

    double * dv_ptra = thrust::raw_pointer_cast(d_A.data());
    double * dv_ptrb = thrust::raw_pointer_cast(&d_B[0]); 
    double * dv_ptrc = thrust::raw_pointer_cast(&d_C[0]); 


    cublasHandle_t handle;
    cublasCreate(&handle);
    
    double alpha = 1;
    double beta =0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dv_ptra, lda, dv_ptrb, ldb, &beta, dv_ptrc, ldc);

    cublasDestroy(handle);

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
        std::cout<<d_C[i]<<std::endl;
    }
    thrust::minstd_rand rng;
    // create a uniform_real_distribution to produce floats from [-7,13)
  thrust::uniform_real_distribution<float> dist(-7,13);
  std::cout << dist(rng) << std::endl;


    
}