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
    int nr_rows_A, nr_cols_A;
    // for simplicity we are going to use square arrays
    nr_rows_A = nr_cols_A  = 2;
    int m, n, k, lda, ldb, ldc; 
    m = n = k = lda = ldb = ldc= nr_rows_A;
	int N = 2;
    
    float *a, *b, *c;
    float *d_A, *d_B, *d_C;

	a = (float*)malloc(N*N*sizeof(float));
	b = (float*)malloc(N*N*sizeof(float));
	c = (float*)malloc(N*N*sizeof(float));

	cudaMalloc( (void**)&d_A, N * N * sizeof(float) );
    cudaMalloc( (void**)&d_B, N * N * sizeof(float) );
    cudaMalloc( (void**)&d_C, N * N * sizeof(float) );

   a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
   b[0] = 5; b[1] = 6; b[2] = 7; b[3] = 8;

	cudaMemcpy(d_A, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, c, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1;
    float beta =0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

	cudaMemcpy(a, d_A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_B, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < nr_rows_A * nr_cols_A; i++){
        std::cout<<c[i]<<std::endl;
    }
    thrust::minstd_rand rng;
    // create a uniform_real_distribution to produce floats from [-7,13)
  thrust::uniform_real_distribution<float> dist(-7,13);
  std::cout << dist(rng) << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);

}