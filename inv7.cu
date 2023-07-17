#include <iostream>

#include <cublas_v2.h>

__global__ void copy(double** aa, double* ab, double** ac, double* ad, double** ae, double* af, double** ag, double* ah, double** dph, double* dh,
                    double** dpi, double* di, int count){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < count){
        aa[idx] = ab + idx*2*1;
        ac[idx] = ad + idx*2*2;
        ae[idx] = af + idx*2*1;
        ag[idx] = ah + idx;
        dph[idx] = dh + idx*2*2;
        dpi[idx] = di + idx*1*1;
    }
}

__global__ void copy1(double** dpb, double* db, double** dph, double* dh, double** dpi, double* di, int group){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < group){
        dpb[idx] = db + idx*2*1;
        dph[idx] = dh + idx*2*1;
        dpi[idx] = di + idx*1*1;
    }
}

__global__ void mult(double* da){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    da[idx] = 2*da[idx];
}
int main(){
    int N = 2;

    int Nmatrices = 3;//number of batches

    cublasHandle_t handle;
    cublasCreate(&handle);
    double *a = new double[N*Nmatrices];
    double *b = new double[N*N*Nmatrices];
    double *c = new double[N*Nmatrices];
    double *d = new double[Nmatrices];
    double *e = new double[N*N];
    double *f = new double[N];
    double *g = new double[N];
    double *h = new double[N*N*Nmatrices];
    double *i = new double[N*Nmatrices];

    a[0] = 1; a[1] = 2; 
    a[2] = 3; a[3] = 4; a[4] = 5; a[5] = 6;
    e[0] = 2; e[1] = 4; e[2] = 3; e[3] = 5;
    f[0] = 1; f[1] = 2;

    b[0] = 2; h[0] = 2; h[1] = 3; h[2] = 4; h[3] = 5; h[4] = 2;h[5] = 3; h[6] = 4; h[7] = 5;h[8] = 2; h[9] = 3; h[10] = 4; h[11] = 5;
    b[1] = 3;
    b[2] = 4;
    b[3] = 5;
    b[4] = 2; 
    b[5] = 3;
    b[6] = 4;
    b[7] = 5;
    b[8] = 2; 
    b[9] = 3;
    b[10] = 4;
    b[11] = 5;

    double *da; cudaMalloc((void**)&da, N*Nmatrices*sizeof(double));
    double *db; cudaMalloc((void**)&db, N*N*Nmatrices*sizeof(double));
    double *dc; cudaMalloc((void**)&dc, N*Nmatrices*sizeof(double));
    double *dd; cudaMalloc((void**)&dd, Nmatrices*sizeof(double));
    double *de; cudaMalloc((void**)&de, N*N*sizeof(double));
    double *df; cudaMalloc((void**)&df, N*sizeof(double));
    double *dg; cudaMalloc((void**)&dg, N*sizeof(double));
    double *dh; cudaMalloc((void**)&dh, N*N*Nmatrices*sizeof(double));
    double *di; cudaMalloc((void**)&di, N*Nmatrices*sizeof(double));

    cudaMemcpy(da, a , N*Nmatrices*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(db, b , N*N*Nmatrices*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(de, e , N*N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(df, f , N*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dh, h , N*N*Nmatrices*sizeof(double),cudaMemcpyHostToDevice);

    double **dpa; cudaMalloc((void**)&dpa, Nmatrices*sizeof(double *));
    double **dpb; cudaMalloc((void**)&dpb, Nmatrices*sizeof(double *));
    double **dpc; cudaMalloc((void**)&dpc, Nmatrices*sizeof(double *));
    double **dpd; cudaMalloc((void**)&dpd, Nmatrices*sizeof(double *));
    double **dph; cudaMalloc((void**)&dph, Nmatrices*sizeof(double *));
    double **dpi; cudaMalloc((void**)&dpi, Nmatrices*sizeof(double *));

    //copy<<<1,20>>>(dpa, da, dpb, db, dpc, dc, dpd, dd, dph, dh, dpi, di, Nmatrices);
    int group = 6;
    copy1<<<1,20>>>(dpb, db, dph, dh, dpi, di, group);

    int m = 1; int k= 2; int n = 2; double alpha = 1; double beta = 0;
    int m1 = 1; int k1 = 2; int n1 = 1; double alpha1 =1; double beta1 = 1;

    for(int i =0; i<2; i++){
        //cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, dpa, n, dpb, n, &beta, dpc, m, Nmatrices);
       // mult<<<1, 6>>>(da);
    }
    //cublasDcopy(handle, N*N*Nmatrices, db, 1, di,1);
    //cudaMemcpy(i, di, N*N*Nmatrices*sizeof(double), cudaMemcpyDeviceToHost);
    //cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1, n1, k1, &alpha, dpc, m1, dpa, k1, &beta, dpd, m1, Nmatrices);
    //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 1, 2, &alpha, de, 2, df, 2, &beta, dg, 2);
    cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 2, &alpha, dpb, 1, dph, 2, &beta, dpi, 1, group);
    //cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 1, 2, &alpha, dpb, 2, dpa, 2, &beta, dpc, 2, Nmatrices);

    //cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, da, n, db, n, &beta, dc, 1);

    //cudaMemcpy(d, dd, Nmatrices*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(c, dc, N*Nmatrices*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(a, da, N*Nmatrices*sizeof(double), cudaMemcpyDeviceToHost);

    //cudaMemcpy(g, dg, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(i, di, N*Nmatrices*sizeof(double), cudaMemcpyDeviceToHost);

    //for(int i = 0; i<N*Nmatrices; i++)std::cout<<c[i]<<std::endl;
    for(int k = 0; k<N*Nmatrices; k++)std::cout<<i[k]<<std::endl;

   //for(int i = 0; i<N*N*Nmatrices; i++)std::cout<<h[i]<<std::endl;
        
    cudaFree(dpa); cudaFree(dpb); cudaFree(dpc);cudaFree(dpd);cudaFree(dph);
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd); cudaFree(de); cudaFree(df); cudaFree(dg); cudaFree(dh);
    cublasDestroy(handle);
}