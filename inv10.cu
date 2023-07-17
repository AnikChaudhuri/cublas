#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <ctime>
#include <random>
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <iomanip>
#include <ctime>
using namespace std;

__global__ void copy(double** aa, double* ab, double** ac, double* ad, double** e, double* f, double** g, double*h, double** m, double* n,
                     double** o, double* p, double** q, double* r, double** s, double* t, double** u, double* v, double** rAb_ino, 
                     double* dAb_ino, double** w, double* dres, int count){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < count){
        aa[idx] = ab + idx*2*2;
        //__syncthreads();
        ac[idx] = ad + idx*2*2;
        //__syncthreads();
        e[idx] = f + idx*1*2;
        //__syncthreads();
        g[idx] = h + idx*2*2;
       // __syncthreads();
        m[idx] = n + idx*2*1;
        //__syncthreads();
        o[idx] = p + idx*2*1;
        //__syncthreads();
        q[idx] = r + idx*2*2;
        //__syncthreads();
        s[idx] = t + idx*1;
        //__syncthreads();
        u[idx] = v + idx*1;
        //__syncthreads();
        rAb_ino[idx] = dAb_ino + idx*2*2;
        //__syncthreads();
        w[idx] = dres + idx*2*1;
        //__syncthreads();
    }
    
    
}

__global__ void fcopy(double** i1, double* j1,double** k, double* l, int batch_size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < batch_size){
        i1[idx] = j1 + idx*2*2;
        //__syncthreads();
        k[idx] = l + idx*2*2;
    }
    
}

__global__ void ccopy(double* Drh, double* D_B, double* cont, double* tem, int gene, int batch){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < gene){
        D_B[idx] = Drh[idx];
        //__syncthreads();
        cont[4*idx] = tem[0];
        //__syncthreads();
        cont[(4*idx) + 1] = tem[1];
        //__syncthreads();
        cont[(4*idx) + 2] = tem[2];
        //__syncthreads();
        cont[(4*idx) + 3] = tem[3];
        //__syncthreads();
    }   
       
}


__global__ void ccopy1(double** dc, double* dco){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dc[idx] = dco + idx*2*2;
    
}

__global__ void add(double* Ab, double* x_a, double* ex){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    ex[idx] = Ab[idx] + x_a[idx];
}

__global__ void mult(double* dAoa, double* dexA, int V){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dexA[idx] = (1.1 + V) * dAoa[idx];
}

__global__ void rho(double* data, double* D3, double* dv1, double* dv, double* Drh){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    Drh[idx] = ((data[idx] - D3[idx])*(data[idx] - D3[idx])) -(2*(data[idx] - D3[idx])*dv1[idx]) + dv[idx];
}

__global__ void mult1(double* dD, double* data, double* dexAK, double alpha2, double* dres, double* D3){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    dres[2*idx] = (dD[2*idx] * (data[idx] - D3[idx] ) )*alpha2 + dexAK[0];
    //__syncthreads();
    dres[(2*idx) + 1] = (dD[(2*idx) + 1] * (data[idx] - D3[idx] ) )*alpha2 + dexAK[1];
    //__syncthreads();
    
}

__global__ void K(double* dex_a, double* D_C, double* D_C1, int V){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    D_C[idx] = dex_a[2*idx];
    //__syncthreads();
    D_C1[idx] = dex_a[(2*idx) +1];
    //__syncthreads();
    //DKok[0] = (S + con)/(V + q0); DKok[1] = (S1 + con)/(V + q0);
}

__global__ void K_f(double* DKok, double S, double S1){
    //int idx = threadIdx.x + blockIdx.x * blockDim.x;
    DKok[0] = S; DKok[1] = S1;
}

__global__ void lambda_oa(double* dAoa_i,double* dsum, double* dqk,double* drKo, double* dAo){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dAoa_i[idx] = dAo[idx] + dsum[idx] + dqk[idx] - drKo[idx]; 
}
__global__ void mult2(double* dex_aa, double* D_E1, double* D_E2, double*D_E3, double* D_E4, int gene){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < gene){
        D_E1[idx] = dex_aa[4*idx];
        //__syncthreads();
        D_E2[idx] = dex_aa[(4*idx) + 1];
        //__syncthreads();
        D_E3[idx] = dex_aa[(4*idx) + 2];
        //__syncthreads();
        D_E4[idx] = dex_aa[(4*idx) + 3];
        //__syncthreads();
    }
    
}

int main(){
    int num = 4000; 
    int gene = 8040; int V = gene; 
    double* d1 = new double[gene];
    double* d2 = new double[gene];
    double* d3 = new double[gene];
    double* mu = new double[gene];
    double* r = new double[gene];
    double* t = new double[gene];
    double* t1 = new double[gene];
    double* D = new double[2*gene];

    ifstream ifs("rwork.txt", ifstream::in);
			
	std::vector<double> a;//r is loaded in a
 
	while((!ifs.eof()) && ifs){

		double iNumber = 0;
 
		ifs >> iNumber;
		a.push_back(iNumber);
		}
	ifs.close();
			

	ifstream ifu("file1.txt", ifstream::in);
			
	std::vector<double> b;//d1 is loaded in b
 
	while((!ifu.eof()) && ifu){
		double iNumber2 = 0;
 
		ifu >> iNumber2;
		b.push_back(iNumber2);
		}
	ifu.close();

    ifstream ifv("file2.txt", ifstream::in);
			
	std::vector<double> c;//d2 is loaded in c
 
	while((!ifv.eof()) && ifv){
		double iNumber3 = 0;
 
		ifv >> iNumber3;
		c.push_back(iNumber3);
		}
	ifv.close();

    ifstream ifw("file3.txt", ifstream::in);
			
	std::vector<double> d;//d3 is loaded in d
 
	while((!ifw.eof()) && ifw){
		double iNumber4 = 0;
 
		ifw >> iNumber4;
		d.push_back(iNumber4);
		}
	ifw.close();
    
    for(int i = 0; i<gene; i++){
        r[i] = a[i]; d1[i] = b[i]; d2[i] = c[i]; d3[i] = d[i];
    }

    std::default_random_engine generator;
    //std::uniform_int_distribution<int> distribution(0,1);
    //std::normal_distribution<double> dist(0, 1);
   
    cout<<"**************************************new code1*****************************************"<<endl;
    
    for(int i =0; i< gene; i++){
                
        //cout<<"r "<<endl;
        t[i] = d1[i]-d3[i];
        t1[i] = d2[i]-d3[i];

        D[2*i] = t[i];
        D[(2*i) + 1] = t1[i];
        //mu[i] = t[i]*a[i] + t1[i]*b[i] + d3[i];
        
        //r[i] = dist(generator)*0.1 + mu[i];
        
        //std::cout<<"r is "<<r[i]<<std::endl;
        //std::cout<<"mu is "<<mu[i]<<std::endl;

    }

    //for(int i = 0; i < gene; i++){cout<<"D"<<i<<"is "<<a[i]<<endl;}
    int N = 2; //batches
    int batch = N*N; int batch_size = 1; double q0 = .001; double con = .001*1.0/3;
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    thrust::device_vector<double> B(gene);
    thrust::device_vector<double> C(gene);
    thrust::device_vector<double> C1(gene);
    thrust::device_vector<double> E1(gene);
    thrust::device_vector<double> E2(gene);
    thrust::device_vector<double> E3(gene);
    thrust::device_vector<double> E4(gene);

    double * D_B = thrust::raw_pointer_cast(B.data());
    double * D_C = thrust::raw_pointer_cast(C.data());
    double * D_C1 = thrust::raw_pointer_cast(C1.data());
    double * D_E1 = thrust::raw_pointer_cast(E1.data());
    double * D_E2 = thrust::raw_pointer_cast(E2.data());
    double * D_E3 = thrust::raw_pointer_cast(E3.data());
    double * D_E4 = thrust::raw_pointer_cast(E4.data());

    double *K1 = new double[num];//
    double *K2 = new double[num];//
    double *lam1 = new double[num];//
    double *lam2 = new double[num];//
    double *lam3 = new double[num];//
    double *lam4 = new double[num];//
    double *duk = new double[N];//K
    double *duAoa = new double[batch];//(A_0A)^-1

    double *ex_a = new double[N*gene];//u_beta_i
    double *Ab = new double[batch*gene];//A_beta_i
    double *inAb = new double[batch*gene];//(A_beta_i)^-1
    double *ex_aa = new double[batch*gene];//E[beta_i*beta_i^t]
    double *rex_aa = new double[batch*gene];//beta_i*beta_i^t
    double *bg = new double[num]; double *gam = new double[num];
    double *Aoa_i = new double[batch];//(A_0A)^-1
    double *Aoa = new double[batch];//(A_0A)
    double *exA = new double[batch];//E[A]
    double *Ao = new double[batch];//(A_0)^-1
    double *Kok = new double[N];
    double *rKo = new double[N*N];
    double *Dout = new double[2*gene];//D^T*E[beta*beta^T]
    double *cov = new double[gene];//D^T(E[beta*beta^T])D
    double *cov1 = new double[gene];//D^T(E[beta])
    double *rh = new double[gene];// result before b_rho
    double *exAK = new double[N]; //E[AK]
    double *res = new double[2*gene];
    double *Ab_ino = new double[batch*gene];//second A_beta_i inverse
    double *sum = new double[N*N];
    double *qk = new double[N*N];//q_0*K_0*K_o^T
    double *S = new double[N];
    int m = 2; int n = 2; int k = 1; double alpha =1 ; double beta = 0; double alpha3 = (1.1 + V); double a_rho = 0.5 + 0.5*(gene);
    int m1 = 1; int n1 = 2; int k1 = 2; double alpha1 = 1; double beta1 = 0; int n2 = 1; double beta2 =1; double alpha4 = q0 + V;
    
    double *dqk; cudaMalloc((void**)&dqk, N*N*sizeof(double));//device vector of qk
    double *dAo; cudaMalloc((void**)&dAo, batch*sizeof(double));//device vector of Ao
    double *dex_a; cudaMalloc((void**)&dex_a, N*gene*sizeof(double));//device vector of ex_a
    double *rdex_a; cudaMalloc((void**)&rdex_a, batch*gene*sizeof(double));//u_beta_i*u_beta_i
    double *dAb; cudaMalloc((void**)&dAb, batch*gene*sizeof(double));// device vector of A_beta_i
    double *dinAb; cudaMalloc((void**)&dinAb, batch*gene*sizeof(double));// device vector of (A_beta_i)^-1
    double *dex_aa; cudaMalloc((void**)&dex_aa, batch*gene*sizeof(double));// device vector of E[beta_i*beta_i^t]
    double *dD; cudaMalloc((void**)&dD, N*gene*sizeof(double));//device vector of D
    double *dDout; cudaMalloc((void**)&dDout, N*gene*sizeof(double));//device vector of Dout
    double *dcov; cudaMalloc((void**)&dcov, gene*sizeof(double));//device vector of cov
    double *dcov1; cudaMalloc((void**)&dcov1, gene*sizeof(double));//device vector of cov1
    double *data; cudaMalloc((void**)&data, gene*sizeof(double));//device vector of r
    double *D3; cudaMalloc((void**)&D3, gene*sizeof(double));//device vector of mu_i = d3
    double *Drh; cudaMalloc((void**)&Drh, gene*sizeof(double));//device vector of rh
    double *DKok; cudaMalloc((void**)&DKok, N*sizeof(double));//device vector of Kok
    double *dexAK; cudaMalloc((void**)&dexAK, N*sizeof(double));//device vector of exAK
    double *dres; cudaMalloc((void**)&dres, N*gene*sizeof(double));//device vector of res
    double *dAb_ino; cudaMalloc((void**)&dAb_ino, batch*gene*sizeof(double));//device vector of Ab_ino
    double *dsum; cudaMalloc((void**)&dsum, N*N*sizeof(double));// device vector of sum
    double *drKo; cudaMalloc((void**)&drKo, N*N*sizeof(double));// device vector of rKo

    double **Ab_inout; cudaMalloc((void**)&Ab_inout, gene*sizeof(double ));// pointer of Ab
    double **rAb_inout; cudaMalloc((void**)&rAb_inout, gene*sizeof(double ));// pointer of inAb
    double **dcov_inout; cudaMalloc((void**)&dcov_inout, gene*sizeof(double ));//pointer of cov
    double **dcov1_inout; cudaMalloc((void**)&dcov1_inout, gene*sizeof(double ));//pointer of cov1
    double **dex_a_in; cudaMalloc((void**)&dex_a_in, gene*sizeof(double ));//pointer of ex_a
    double **rdex_a_in; cudaMalloc((void**)&rdex_a_in, gene*sizeof(double ));//pointer of rex_aa
    double **dAoa_a_in; cudaMalloc((void**)&dAoa_a_in, batch_size*sizeof(double ));//pointer of Aoa_i
    double **dAoa_out; cudaMalloc((void**)&dAoa_out, batch_size*sizeof(double ));//pointer of Aoa
    double **dD_in; cudaMalloc((void**)&dD_in, gene*sizeof(double ));//pointer of dD
    double **dD_out; cudaMalloc((void**)&dD_out, gene*sizeof(double ));//pointer of dDout
    double **dex_aa_in; cudaMalloc((void**)&dex_aa_in, gene*sizeof(double ));// pointer of dex_aa = E[beta*beta^T]
    double **rAb_ino; cudaMalloc((void**)&rAb_ino, gene*sizeof(double ));//pointer vector of dAb_ino
    double **dres_in; cudaMalloc((void**)&dres_in, gene*sizeof(double ));//pointer of dres
    //double **dcon_in; cudaMalloc((void**)&dcon_in, gene*sizeof(double *));//pointer of dcon
   
/*
    double **Ab_inout = (double **)malloc(gene*sizeof(double *));// pointer of Ab
    double **rAb_inout = (double **)malloc(gene*sizeof(double *)); // pointer of inAb
    double **dcov_inout = (double **)malloc(gene*sizeof(double *));//pointer of cov
    double **dcov1_inout = (double **)malloc(gene*sizeof(double *));//pointer of cov1
    double **dex_a_in = (double **)malloc(gene*sizeof(double *));//pointer of ex_a
    double **rdex_a_in = (double **)malloc(gene*sizeof(double *));//pointer of rex_aa
    double **dAoa_a_in = (double **)malloc(batch_size*sizeof(double *));//pointer of Aoa_i
    double **dAoa_out = (double **)malloc(batch_size*sizeof(double *));//pointer of Aoa
    double **dD_in = (double **)malloc(gene*sizeof(double *));//pointer of dD
    double **dD_out = (double **)malloc(gene*sizeof(double *));//pointer of dDout
    double **dex_aa_in = (double **)malloc(gene*sizeof(double *));// pointer of dex_aa = E[beta*beta^T]
    double **rAb_ino = (double **)malloc(gene*sizeof(double *));//pointer vector of dAb_ino
    double **dres_in= (double **)malloc(gene*sizeof(double *));//pointer of dres
*/

    int *d_InfoArray;  cudaMalloc((void**)&d_InfoArray,  gene*sizeof(int));
    double *dAoa_i; cudaMalloc((void**)&dAoa_i, batch*sizeof(double));//device vecor of Aoa_i
    double *dAoa; cudaMalloc((void**)&dAoa, batch*sizeof(double));//device vecor of Aoa
    double *dexA; cudaMalloc((void**)&dexA, batch*sizeof(double));//device vector of E[A]
    
    int *d1_InfoArray;  cudaMalloc((void**)&d1_InfoArray,  batch_size*sizeof(int));//used in inverse of (Aoa)
    
    for(int i = 0; i < gene*N; i++){ex_a[i] = 1./3;}

    Ab[0] = .008/((.01*.008)-(.005*.005)); Ab[1] = -0.005/((.01*.008)-(.005*.005)); 
    Ab[2] = -0.005/((.01*.008)-(.005*.005));Ab[3] = .01/((.01*.008)-(.005*.005));

    Ao[0] = 0.01; Aoa_i[0] = 0.01; 
    Ao[1] = .005; Aoa_i[1] = .005; 
    Ao[2] = 0.005; Aoa_i[2] = 0.005; 
    Ao[3] = 0.008; Aoa_i[3] = 0.008;

    Kok[0] = Kok[1] = 1.0/3.0; qk[0] = qk[1] = qk[2] = qk[3] = q0*1.0/9;
    
    for(int i = 1; i < (gene); i++){
        Ab[4*i] = Ab[0];
        Ab[(4*i) + 1] = Ab[1];
        Ab[(4*i) + 2] = Ab[1];
        Ab[(4*i) + 3] = Ab[3];
        
    }
    std::cout<<"copy"<<std::endl;
    //for(int i = 0; i< gene*batch; i++)cout<<"element no. "<< i << " is "<<Ab[i]<<endl;
    
    cudaMemcpy(dex_a, ex_a, N*gene*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dAb, Ab, batch*gene*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dAoa_i, Aoa_i, batch*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dD, D, 2*gene*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data, r, gene*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(D3, d3, gene*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(DKok, Kok, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dqk, qk, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dAo, Ao, batch*sizeof(double), cudaMemcpyHostToDevice);

    
    double alpha2; double b_rho = 0;

/*
    for(int i = 0; i<gene; i++){
        //std::cout<<i<<std::endl;
        Ab_inout[i] = dAb + i*2*2;
        rAb_inout[i] = dinAb + i*2*2;
        rdex_a_in[i] = rdex_a + i*2*2;
        dex_aa_in[i] = dex_aa + i*2*2;
        
        rAb_ino[i] = dAb_ino + i*2*2;       
    }
    for(int i = 0; i<gene; i++){
        dex_a_in[i] = dex_a + i*1*2;
        
        dD_in[i] = dD + i*2*1;
        dD_out[i] = dDout + 2*1;
        dres_in[i] = dres + i*2*1;
    }
    for(int i =0; i<gene; i++){
        dcov_inout[i] = dcov + i*1;
        dcov1_inout[i] = dcov1 + i*1;
    }

    for(int i = 0; i<batch_size; i++){
        dAoa_a_in[i] = dAoa_i + i*2*2;
        dAoa_out[i] = dAoa + i*2*2;
    }*/
    copy<<<100,900>>>(Ab_inout, dAb, rAb_inout, dinAb, dex_a_in, dex_a, rdex_a_in, rdex_a, dD_in, dD, dD_out, 
        dDout, dex_aa_in, dex_aa, dcov_inout, dcov, dcov1_inout, dcov1, rAb_ino, dAb_ino, dres_in, dres, gene);

    fcopy<<<1,10>>>( dAoa_a_in, dAoa_i, dAoa_out, dAoa, batch_size);

    std::cout<<"done"<<std::endl;
    for(int j = 0; j < num; j++){
        std::cout<<"loop no: "<<j<<std::endl;
        cublasDmatinvBatched(handle, N, Ab_inout, N, rAb_inout, N, d_InfoArray, gene);//result in dinAb (inverse of A_beta_i)
               
        cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, dex_a_in, m, dex_a_in, m, &beta, rdex_a_in, m, gene);//ubeta_i*ubeta_i
        add<<<40, 804>>>(dinAb, rdex_a, dex_aa);//calculating E[beta_i*beta_i^t], result in dex_aa, gene*batch threads needed.
        cublasDmatinvBatched(handle, N, dAoa_a_in, N, dAoa_out, N, d1_InfoArray, batch_size);//result in dAoa (inverse of Aoa_i)
        mult<<<1, batch>>>(dAoa, dexA, V);//E[A], result in dexA
        
        cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, m1, n1, k1, &alpha1, dD_in, n1, dex_aa_in, n1, &beta1, dD_out, 1, gene);//D^T*E[beta*beta^T]
        //dcov_inout[0] = dcov;
        cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1, n2, k1, &alpha1, dD_out, m1, dD_in, k1, &beta1, dcov_inout, m1, gene);//D^T*E[beta*beta^T]D

        cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1, n2, k1, &alpha1, dD_in, m1, dex_a_in, k1, &beta1, dcov1_inout, m1, gene);//D^T*E[beta]

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 1, 2, &alpha3, dAoa, 2, DKok, 2, &beta, dexAK, 2);//E[AK]
        
        //calculating b_rho
        rho<<<10,804>>>(data, D3, dcov1, dcov, Drh); //gene threads needed.
        
        ccopy<<<40,804>>>(Drh, D_B, dAb, dexA, gene, batch); //gene threads needed.
        
        b_rho = thrust::reduce(B.begin(),B.end());//b_rho
        
        alpha2 = a_rho / ((b_rho * 0.5) + 0.5); //E[rho]
        bg[j] = ((b_rho * 0.5) + 0.5);
        //std::cout<<"b"<<std::endl;

        //calculating Kok
        K<<<10,804>>>(dex_a, D_C, D_C1, V); //gene threads needed.
        //long start = clock();//clock
        S[0] = (thrust::reduce(C.begin(), C.end()) + con)/ (q0 + V);
        S[1] = (thrust::reduce(C1.begin(), C1.end()) + con) / (q0 + V);
        cudaMemcpy(DKok, S, N*sizeof(double), cudaMemcpyHostToDevice); // update Kok
        //K_f<<<1, N>>>(DKok, S, S1);// update Kok
        //long stop = clock();//stop clock
        //long fast = stop-start;//time taken

        //update(A0A)^-1
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha4, DKok, m, DKok, m, &beta, drKo, m);//(q0+V)*Kok*Kok^T result in drKo
        mult2<<<40,804>>>(dex_aa, D_E1, D_E2, D_E3, D_E4, gene); //gene threads needed.
        sum[0] = thrust::reduce(E1.begin(), E1.end());
        sum[1] = thrust::reduce(E2.begin(), E2.end());
        sum[2] = thrust::reduce(E3.begin(), E3.end());
        sum[3] = thrust::reduce(E4.begin(), E4.end());
        cudaMemcpy(dsum, sum, N*N*sizeof(double), cudaMemcpyHostToDevice);
        lambda_oa<<<1, N*N>>>(dAoa_i, dsum, dqk, drKo, dAo);

        //calculating A_beta_i
        cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 1, &alpha2, dD_in, 2, dD_in, 2, &beta2, Ab_inout, 2,gene);//A_beta_i

        //calculating mu_beta_i
        mult1<<<10,804>>>(dD, data, dexAK, alpha2, dres, D3);//result in dres. gene threads needed.
        cublasDmatinvBatched(handle, N, Ab_inout, N, rAb_ino, N, d_InfoArray, gene);//result in dAb_ino (inverse of A_beta_i)
        cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 1, 2, &alpha, rAb_ino, 2, dres_in, 2, &beta, dex_a_in, 2, gene);//update mu_beta_i
        cudaMemcpy(duk, DKok, N*sizeof(double), cudaMemcpyDeviceToHost);//  Kok
        cudaMemcpy(duAoa, dAoa_i, batch*sizeof(double), cudaMemcpyDeviceToHost);//(A0A)^-1
        K1[j] = duk[0];
        K2[j] = duk[1];
        lam1[j] = duAoa[0]; lam2[j] = duAoa[1]; lam3[j] = duAoa[2]; lam4[j] = duAoa[3]; 
    }
    
    /*
        cudaMemcpy(ex_aa, dex_aa, gene*batch*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Aoa, dAoa, batch*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(exA, dexA, batch*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Dout, dDout, 2*gene*sizeof(double), cudaMemcpyDeviceToHost);//D^T*E[beta*beta^T]
        cudaMemcpy(cov, dcov, gene*sizeof(double), cudaMemcpyDeviceToHost);//D^T*E[beta*beta^T]D
        cudaMemcpy(cov1, dcov1, gene*sizeof(double), cudaMemcpyDeviceToHost);//D^T*E[beta]
        cudaMemcpy(exAK, dexAK, N*sizeof(double), cudaMemcpyDeviceToHost);//E[AK]
        cudaMemcpy(rh, Drh, gene*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ab, dAb, batch*gene*sizeof(double), cudaMemcpyDeviceToHost);// A_beta_i
        cudaMemcpy(Ab_ino, dAb_ino, batch*gene*sizeof(double), cudaMemcpyDeviceToHost);// A_beta_i inverse
        cudaMemcpy(res, dres, 2*gene*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(ex_a, dex_a, N*gene*sizeof(double), cudaMemcpyDeviceToHost);// update mu_beta_i
        //cudaMemcpy(Kok, DKok, N*sizeof(double), cudaMemcpyDeviceToHost);// update Kok
     */   

    

    
    for(int i =0; i < num; i++){
        //std::cout<<"data " <<i<<"= "<< std::setprecision(10) << bg[i]<<endl;
        std::gamma_distribution<double> dist1(a_rho, 1/bg[i]);
        gam[i] = (dist1(generator));
        std::cout<<"g"<<i<<"="<<gam[i]<<std::endl;
    }

    //saving K1, K2, lam1, lam2, lam3, lam4 on hard disk.
     std::ofstream myfile3;
     myfile3.open ("K1.txt");
     //replace num with counter to save the thinned data. di1, di2, di3 for thinned alpha, and Kth1, Kth2, Kth3 for thinned K.
     for(int p1 = 0; p1 < num; p1++){
         myfile3 << K1[p1] << std::endl;

     }
     myfile3.close();

     std::ofstream myfile4;
     myfile4.open ("K2.txt");
     
     for(int p1 = 0; p1 < num; p1++){
         myfile4 << K2[p1] << std::endl;

     }
     myfile4.close();

     std::ofstream myfile5;
     myfile5.open ("lam1.txt");
     
     for(int p1 = 0; p1 < num; p1++){
         myfile5 << lam1[p1] << std::endl;

     }
     myfile5.close();

     std::ofstream myfile6;
     myfile6.open ("lam2.txt");
     
     for(int p1 = 0; p1 < num; p1++){
         myfile6 << lam2[p1] << std::endl;

     }
     myfile6.close();

     std::ofstream myfile7;
     myfile7.open ("lam3.txt");
     
     for(int p1 = 0; p1 < num; p1++){
         myfile7 << lam3[p1] << std::endl;

     }
     myfile7.close();

     std::ofstream myfile8;
     myfile8.open ("lam4.txt");
     
     for(int p1 = 0; p1 < num; p1++){
         myfile8 << lam4[p1] << std::endl;

     }
     myfile8.close();

    cudaFree(dAoa); cudaFree(dAoa_i); cudaFree(dAoa_a_in); cudaFree(dAoa_out);cudaFree(d1_InfoArray);
    cudaFree(dex_a); cudaFree(dexA); cudaFree(dD_in); cudaFree(dD_out);cudaFree(dex_aa_in);cudaFree(dD);cudaFree(dDout);
    cudaFree(dex_aa); cudaFree(dcov); cudaFree(dcov_inout); cudaFree(dcov1); cudaFree(dcov1_inout); cudaFree(dsum);
    cudaFree(rdex_a);cudaFree(dexAK); cudaFree(DKok); cudaFree(dres); cudaFree(dres_in); cudaFree(rAb_ino); cudaFree(dAb_ino);
    cudaFree(dAb); cudaFree(dAo);
    cudaFree(dinAb);
    cudaFree(Ab_inout);
    cudaFree(rAb_inout);
    cudaFree(d_InfoArray);
    cublasDestroy(handle);
}
