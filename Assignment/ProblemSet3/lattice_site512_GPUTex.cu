// Solve the Laplace equation on a 2D lattice with boundary conditions.
//
// compile with the following command:
//
// (for GTX970)
// nvcc -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 -o laplace laplace.cu
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o laplace laplace.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// field variables
float* h_new;   // host field vectors
float* h_old;   
float* h_C;     // result of diff*diff of each block
float* g_new;   
float* d_new;   // device field vectors
float* d_old;  
float* d_C;

int     MAX=1000000;      // maximum iterations
double  eps=1.0e-10;      // stopping criterion

__align__(8) texture<float>  texOld;   // declare the texture
__align__(8) texture<float>  texNew;


__global__ void laplacian(float* phi_old, float* phi_new, float* C, bool flag)
{
    extern __shared__ float cache[];     
    float  t, l, c, r, b;     // top, left, center, right, bottom
    float  diff; 
    int    site, ym1, xm1, xp1, yp1;

    int Nx = blockDim.x*gridDim.x; // number of site in x direction
    int Ny = blockDim.y*gridDim.y; // number of site in y direction
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;  

    site = x + y*Nx;

    if((x == 0) || (x == Nx-1) || (y == 0) || (y == Ny-1) ) {  
      // do nothing on the boundaries 
    }
    else {
      xm1 = site - 1;    // x-1
      xp1 = site + 1;    // x+1
      ym1 = site - Nx;   // y-1
      yp1 = site + Nx;   // y+1
      if(flag) {
        b = tex1Dfetch(texOld, ym1);      // read d_old via texOld
        l = tex1Dfetch(texOld, xm1);
        c = tex1Dfetch(texOld, site);
        r = tex1Dfetch(texOld, xp1);
        t = tex1Dfetch(texOld, yp1);
        phi_new[site] = 0.25*(b+l+r+t);
        diff = phi_new[site]-c;
      }
      else {
        b = tex1Dfetch(texNew, ym1);     // read d_new via texNew
        l = tex1Dfetch(texNew, xm1);
        c = tex1Dfetch(texNew, site);
        r = tex1Dfetch(texNew, xp1);
        t = tex1Dfetch(texNew, yp1);
        phi_old[site] = 0.25*(b+l+r+t);
        diff = phi_old[site]-c;
      }
    }

    // each thread saves its error estimate to the shared memory

    cache[cacheIndex]=diff*diff;  
    __syncthreads();

    // parallel reduction in each block 

    int ib = blockDim.x*blockDim.y/2;  
    while (ib != 0) {  
      if(cacheIndex < ib)  
        cache[cacheIndex] += cache[cacheIndex + ib];
      __syncthreads();
      ib /=2;  
    } 

    // save the partial sum of each block to C

    int blockIndex = blockIdx.x + gridDim.x*blockIdx.y;
    if(cacheIndex == 0)  C[blockIndex] = cache[0];  
}

int main(void)
{

    int gid;              // GPU_ID
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old  
    float gputime;
    float gputime_tot;
    double flops;
    double error;
    
    printf("Enter the GPU ID (0/1): ");
    scanf("%d",&gid);
    printf("%d\n",gid);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Select GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    int Nx, Ny;   // lattice size
    int tx, ty;   // block size, threads (tx, ty) per block
    int bx, by;   // grid size, block (bx, by) per grid
    int N;        // total number of site
    int size;     // size of the array h_old, h_new
    int sb;       // size of the array h_C;
    int sm;       // size of shared memory
    float Intime;
    float Outime;


    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Write all the result to file.
    FILE *output;
    output = fopen("Output_GPUTex_site512.txt", "a");
    fprintf(output, "LatticeSize BlockSize GPUTexInput GPUTexerror GPUTexiteration GPUTexonly GPUTexflop GPUTexoutput GPUTextotal\n");
    fclose(output);
    
    Nx = 512;
    Ny = 512;
    
    for(int n = 2; n <= 5; n = n+1){

      tx = pow(2, n);
      ty = tx;

      dim3 threads(tx,ty);
      bx = Nx / tx;
      by = Ny / ty;
      dim3 blocks(bx,by);

      // Allocate field vector h_phi in host memory
      N = Nx * Ny;
      size = N * sizeof(float);
      sb = bx * by * sizeof(float);
      h_old = (float*)malloc(size);
      h_new = (float*)malloc(size);
      g_new = (float*)malloc(size);
      h_C = (float*)malloc(sb);

      // Initialize the array to 0
      memset(h_old, 0, size);
      memset(h_new, 0, size);

      // Initialize the field vector with boundary conditions
      for(int x = 0; x < Nx; x = x+1) {
        // phi = +1 on top
        h_new[x + Nx * (Ny - 1)] = 1.0;
        h_old[x + Nx * (Ny - 1)] = 1.0;

        // phi = +5 in bottom
        h_new[x] = 5.0;
        h_old[x] = 5.0;
      }

      for(int y = 0; y < Ny; y = y+1){
        //phi = -1 in left
        h_new[Nx * y] = -1.0;
        h_old[Nx * y] = -1.0;
        //phi = -2 in right
        h_new[(Nx - 1) + Nx * y] = -2.0;
        h_old[(Nx - 1) + Nx * y] = -2.0;
      }

      FILE *out1;                 // save initial configuration in phi_initial.dat
      out1 = fopen("phi_initial_GPUTex_site512.dat","w");
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(out1,"%.2e ",h_new[i+j*Nx]);
        }
        fprintf(out1,"\n");
      }
      fclose(out1);

      printf("\n");

      // start the timer
      cudaEventRecord(start,0);

      // Allocate vectors in device memory

      cudaMalloc((void**)&d_new, size);
      cudaMalloc((void**)&d_old, size);
      cudaMalloc((void**)&d_C, sb);

      cudaBindTexture(NULL, texOld, d_old, size);   // bind the texture to already existed variable on 
      cudaBindTexture(NULL, texNew, d_new, size);   // device memory
  
      // Copy vectors from host memory to device memory
      cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);
    
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      
      cudaEventElapsedTime( &Intime, start, stop);
      printf("Input time for GPU: %f (ms) \n",Intime);

      // start the timer
      cudaEventRecord(start,0);

      error = 10*eps;  // any value bigger than eps is OK
      iter = 0;        // counter for iterations
      flag = true; 
 
      sm = tx * ty * sizeof(float);   // size of the shared memory in each block

      while ( (error > eps) && (iter < MAX) ) {

        laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_C, flag);
        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
        error = 0.0;
        for(int i=0; i<bx*by; i++) {
          error = error + h_C[i];
        }
        error = sqrt(error);

//        printf("error = %.15e\n",error);
//        printf("iteration = %d\n",iter);

        iter++;
        flag = !flag;

      }

      printf("error (GPU) = %.15e\n",error);
      printf("total iterations (GPU) = %d\n",iter);
    
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      cudaEventElapsedTime( &gputime, start, stop);
      printf("Processing time for GPU: %f (ms) \n",gputime);
      flops = 7.0*(Nx-2)*(Ny-2)*iter;
      printf("GPU Gflops: %f\n",flops/(1000000.0*gputime));
    
      // Copy result from device memory to host memory
  
      // start the timer
      cudaEventRecord(start,0);

      // Because after the iteration, d_new and d_old are basically the same.
      cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);

      cudaFree(d_new);
      cudaFree(d_old);
      cudaFree(d_C);

      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);

      
      cudaEventElapsedTime( &Outime, start, stop);
      printf("Output time for GPU: %f (ms) \n",Outime);

      gputime_tot = Intime + gputime + Outime;
      printf("Total time for GPU: %f (ms) \n",gputime_tot);
      fflush(stdout);

      FILE *outg;                 // save GPU solution in phi_GPU.dat
      outg = fopen("phi_GPUTex_site512.dat","w");
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(outg,"%.2e ",g_new[i+j*Nx]);
        }
        fprintf(outg,"\n");
      }
      fclose(outg);

      // Write all the output to file
      output = fopen("Output_GPUTex_site512.txt", "a");        
      fprintf(output, "%d %d %f %f %d %f %f %f %f\n", Nx, tx, Intime, error, iter, gputime, flops/(1000000.0*gputime), Outime, gputime_tot);
      fclose(output);

      printf("\n");

      printf("Finish computing lattice size : %d, block size : %d\n", Nx, tx);

      free(h_new);
      free(h_old);
      free(g_new);
      free(h_C);

    }
    

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
    
}

