// Solve the Laplace equation on a 2D lattice with boundary conditions.
// (using texture memory)
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
float* h_C;     // sum of diff*diff of each block
float* g_new;   // device solution back to the host 
float* d_new;   // device field vectors
float* d_old;  
float* d_C;     // sum of diff*diff of each block 

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

    int Nx = blockDim.x*gridDim.x;
    int Ny = blockDim.y*gridDim.y;
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

    double  error;            // error estimate 

    int gid;

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
    cudaSetDevice(gid);
    printf("Select GPU with device ID = %d\n", gid);

    printf("Solve Laplace equation on a 2D lattice with boundary conditions (using texture)\n");

    int Nx,Ny;                // lattice size
    printf("Enter the size (Nx, Ny) of the 2D lattice: ");
    scanf("%d %d",&Nx,&Ny);        
    printf("%d %d\n",Nx,Ny);        

    // Set the number of threads (tx,ty) per block
   
    int tx,ty;
    printf("Enter the number of threads (tx,ty) per block: ");
    scanf("%d %d",&tx, &ty);
    printf("%d %d\n",tx, ty);
    if( tx*ty > 1024 ) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(0);
    }
    dim3 threads(tx,ty); 
    
    // The total number of threads in the grid is equal to the total number of lattice sites
    
    int bx = Nx/tx;
    if(bx*tx != Nx) {
      printf("The blocksize in x is incorrect\n"); 
      exit(0);
    }
    int by = Ny/ty;
    if(by*ty != Ny) {
      printf("The blocksize in y is incorrect\n"); 
      exit(0);
    }
    if((bx > 65535)||(by > 65535)) {
      printf("The grid size exceeds the limit ! \n");
      exit(0);
    }
    dim3 blocks(bx,by);
    printf("The dimension of the grid is (%d, %d)\n",bx,by); 

    int CPU;    
    printf("To compute the solution vector with CPU (1/0) ?");
    scanf("%d",&CPU);
    printf("%d\n",CPU);
    fflush(stdout);
   
    // Allocate field vector h_phi in host memory

    int N = Nx*Ny;
    int size = N*sizeof(float);
    int sb = bx*by*sizeof(float);
    h_old = (float*)malloc(size);
    h_new = (float*)malloc(size);
    g_new = (float*)malloc(size);
    h_C = (float*)malloc(sb);
   
    memset(h_old, 0, size);    
    memset(h_new, 0, size);

    // Initialize the field vector with boundary conditions

    for(int x=0; x<Nx; x++) {
      h_new[x+Nx*(Ny-1)]=1.0;  
      h_old[x+Nx*(Ny-1)]=1.0;
    }  

    FILE *out1;		        // save initial configuration in phi_initial.dat 
    out1 = fopen("phi_initial.dat","w");

    fprintf(out1, "Inital field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(out1,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(out1,"\n");
    }
    fclose(out1);

//   printf("\n");
//    printf("Inital field configuration:\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",h_new[i+j*Nx]);
//      }
//      printf("\n");
//    }
//    printf("\n");

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

    // start the timer
    cudaEventRecord(start,0);

    error = 10*eps;      // any value bigger than eps is OK
    int iter = 0;        // counter for iterations
    double diff; 

    volatile bool flag = true;     

    int sm = tx*ty*sizeof(float);   // size of the shared memory in each block

    while ( (error > eps) && (iter < MAX) ) {

      laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_C, flag);
      cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
      error = 0.0;
      for(int i=0; i<bx*by; i++) {
        error = error + h_C[i];
      }
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

      iter++;
      flag = !flag;
      
    }
     
    printf("error (GPU) = %.15e\n",error);
    printf("total iterations (GPU) = %d\n",iter);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    double flops;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    flops = 7.0*(Nx-2)*(Ny-2)*iter;
    printf("GPU Gflops: %f\n",flops/(1000000.0*gputime));
    
    // Copy result from device memory to host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(g_new, d_new, size, cudaMemcpyDeviceToHost);

    cudaFree(d_new);
    cudaFree(d_old);
    cudaFree(d_C);

    cudaUnbindTexture(texOld);
    cudaUnbindTexture(texNew);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);
    fflush(stdout);

    printf("\n");
//    printf("Final field configuration (GPU):\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",g_new[i+j*Nx]);
//      }
//      printf("\n");
//    }
//    printf("\n");

    FILE *outg;                 // save GPU solution in phi_GPU_Tex.dat 
    outg = fopen("phi_GPU_Tex.dat","w");

    fprintf(outg, "GPU (using texture) field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(outg,"%.2e ",g_new[i+j*Nx]);
      }
      fprintf(outg,"\n");
    }
    fclose(outg);

    // start the timer
    cudaEventRecord(start,0);

    if(CPU==0) {
      free(h_new);
      free(h_old);
      free(g_new);
      free(h_C);
      cudaDeviceReset();
    } 
 
    // to compute the reference solution

    error = 10*eps;      // any value bigger than eps 
    iter = 0;            // counter for iterations
    flag = true;     

    float t, l, r, b;    // top, left, right, bottom
    int site, ym1, xm1, xp1, yp1;

    while ( (error > eps) && (iter < MAX) ) {
      if(flag) {
        error = 0.0;
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1) {   
          }
          else {
            site = x+y*Nx;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            b = h_old[ym1]; 
            l = h_old[xm1]; 
            r = h_old[xp1]; 
            t = h_old[yp1]; 
            h_new[site] = 0.25*(b+l+r+t);
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
        } 
        } 
      }
      else {
        error = 0.0;
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1) {
          }
          else {
            site = x+y*Nx;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            b = h_new[ym1]; 
            l = h_new[xm1]; 
            r = h_new[xp1]; 
            t = h_new[yp1]; 
            h_old[site] = 0.25*(b+l+r+t);
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          } 
        }
        }
      }
      flag = !flag;
      iter++;
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

    }   // exit if error < eps
    
    printf("error (CPU) = %.15e\n",error);
    printf("total iterations (CPU) = %d\n",iter);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    flops = 7.0*(Nx-2)*(Ny-2)*iter;
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));
    fflush(stdout);

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FILE *outc;               // save CPU solution in phi_CPU.dat 
    outc = fopen("phi_CPU.dat","w");

    fprintf(outc,"CPU field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(outc,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(outc,"\n");
    }
    fclose(outc);

//    printf("\n");
//    printf("Final field configuration (CPU):\n");
//    for(int j=Ny-1;j>-1;j--) {
//      for(int i=0; i<Nx; i++) {
//        printf("%.2e ",h_new[i+j*Nx]);
//      }
//      printf("\n");
//    }

    free(h_new);
    free(h_old);
    free(h_C);
    free(g_new);

    cudaDeviceReset();
}

