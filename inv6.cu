#include <stdio.h>

#include <cublas_v2.h>

__global__ void copy(float** a, float* b, float** c, float* d, int count, float** e, float* f){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < count){
        a[idx] = b + idx*2*2 ;
        c[idx] = d + idx*2*2;
        e[idx] = f + idx*2*2;
    }
    
}

int main() {

int N = 2;

int Nmatrices = 3;//number of batches

cublasHandle_t handle;
cublasCreate(&handle);

// --- Matrices to be inverted
float *h_A = new float[N*N*Nmatrices];
float *h_B = new float[N*N*Nmatrices];
float *r_A = new float[N*N*Nmatrices];//result

h_A[0] = 4.f;
h_A[1] = 3.f;
h_A[2] = 8.f;
h_A[3] = 9.f;
h_A[4] = 4.f;
h_A[5] = 3.f;
h_A[6] = 8.f;
h_A[7] = 9.f;
h_A[8] = 4.f;
h_A[9] = 3.f;
h_A[10] = 8.f;
h_A[11] = 9.f;

h_B[0] = 4.f;
h_B[1] = 3.f;
h_B[2] = 8.f;
h_B[3] = 9.f;
h_B[4] = 4.f;
h_B[5] = 3.f;
h_B[6] = 8.f;
h_B[7] = 9.f;
h_B[8] = 4.f;
h_B[9] = 3.f;
h_B[10] = 8.f;
h_B[11] = 9.f;


int count = Nmatrices;
// --- Allocate device matrices
float *d_A; cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float));
float *d_B; cudaMalloc((void**)&d_B, N*N*Nmatrices*sizeof(float));
float *c_A; cudaMalloc((void**)&c_A, N*N*Nmatrices*sizeof(float));

cudaMemcpy(d_A,h_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_B,h_B,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice);

float **d_inout_pointers;
cudaMalloc((void**)&d_inout_pointers, Nmatrices*sizeof(float *));

float **db_inout_pointers;
cudaMalloc((void**)&db_inout_pointers, Nmatrices*sizeof(float *));

float **rd_inout_pointers;
cudaMalloc((void**)&rd_inout_pointers, Nmatrices*sizeof(float *));

    int lda, ldb, ldc;
    lda = ldb = ldc = 2;
    float alpha = 1; float beta = 0;
    copy<<<1,10>>>(d_inout_pointers, d_A, rd_inout_pointers, c_A, count, db_inout_pointers, d_B);
    
    cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_inout_pointers, lda, d_inout_pointers, ldb, &beta,rd_inout_pointers,ldc,Nmatrices);


    cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(r_A,c_A,Nmatrices*N*N*sizeof(float),cudaMemcpyDeviceToHost);
    //gpuErrchk(cudaMemcpy(h_PivotArray,d_PivotArray,N*Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

    for (int i=0; i<N*N*Nmatrices; i++) printf("A[%i]=%f\n", i, r_A[i]);
    
cudaFree(c_A);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_inout_pointers);
cudaFree(rd_inout_pointers);
cudaFree(db_inout_pointers);
cublasDestroy(handle);


return 0;
}