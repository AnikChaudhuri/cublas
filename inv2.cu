#include <stdio.h>

#include <cublas_v2.h>



int main() {

const unsigned int N = 2;

const unsigned int Nmatrices = 2;

cublasHandle_t handle;
cublasCreate(&handle);

// --- Matrices to be inverted
float *h_A = new float[N*N*Nmatrices];
float *r_A = new float[N*N*Nmatrices];//result

h_A[0] = 4.f;
h_A[1] = 3.f;
h_A[2] = 8.f;
h_A[3] = 9.f;
h_A[4] = 5.f;
h_A[5] = 1.f;
h_A[6] = 2.f;
h_A[7] = 7.f;


// --- Allocate device matrices
float *d_A; cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float));
float *c_A; cudaMalloc((void**)&c_A, N*N*Nmatrices*sizeof(float));

// --- Move the matrix to be inverted from host to device
cudaMemcpy(d_A,h_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice);

// --- Creating the array of pointers needed as input to the batched getrf
float **h_inout_pointers = (float **)malloc(Nmatrices*sizeof(float *));
//for (int i=0; i<Nmatrices; i++) h_inout_pointers[i]=(float *)((char*)d_A+i*((size_t)N*N)*sizeof(float));
h_inout_pointers[0]=d_A;
h_inout_pointers[1]=d_A+N*N;

float **d_inout_pointers;
cudaMalloc((void**)&d_inout_pointers, Nmatrices*sizeof(float *));
cudaMemcpy(d_inout_pointers,h_inout_pointers,Nmatrices*sizeof(float *),cudaMemcpyHostToDevice);
//free(h_inout_pointers);

float **r_inout_pointers = (float **)malloc(Nmatrices*sizeof(float *));
//for (int i=0; i<Nmatrices; i++) h_inout_pointers[i]=(float *)((char*)d_A+i*((size_t)N*N)*sizeof(float));
r_inout_pointers[0]=c_A;
r_inout_pointers[1]=c_A+N*N;

float **rd_inout_pointers;
cudaMalloc((void**)&rd_inout_pointers, Nmatrices*sizeof(float *));
cudaMemcpy(rd_inout_pointers,r_inout_pointers,Nmatrices*sizeof(float *),cudaMemcpyHostToDevice);

int *d_PivotArray; cudaMalloc((void**)&d_PivotArray, N*Nmatrices*sizeof(int));
int *d_InfoArray;  cudaMalloc((void**)&d_InfoArray,  Nmatrices*sizeof(int));

int *h_PivotArray = (int *)malloc(N*Nmatrices*sizeof(int));
int *h_InfoArray  = (int *)malloc(  Nmatrices*sizeof(int));

cublasSgetrfBatched(handle, N, d_inout_pointers, N, d_PivotArray, d_InfoArray, Nmatrices);
//cublasSafeCall(cublasSgetrfBatched(handle, N, d_inout_pointers, N, NULL, d_InfoArray, Nmatrices));

//gpuErrchk(cudaMemcpy(h_InfoArray,d_InfoArray,Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));
cublasSgetriBatched(handle, N, d_inout_pointers, N, d_PivotArray, rd_inout_pointers, N, d_InfoArray,
Nmatrices);


cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost);
cudaMemcpy(r_A,c_A,Nmatrices*N*N*sizeof(float),cudaMemcpyDeviceToHost);
//gpuErrchk(cudaMemcpy(h_PivotArray,d_PivotArray,N*Nmatrices*sizeof(int),cudaMemcpyDeviceToHost));

for (int i=0; i<N*N*Nmatrices; i++) printf("A[%i]=%f\n", i, r_A[i]);
cudaFree(c_A);
cudaFree(d_A);
cudaFree(d_inout_pointers);
cudaFree(rd_inout_pointers);
cudaFree(d_InfoArray);
cudaFree(d_PivotArray);
cublasDestroy(handle);
return 0;
}