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

int     MAX=1000000;      // maximum iterations
double  eps=1.0e-10;      // stopping criterion



int main(void)
{

    int gid;              // GPU_ID
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old  
    float cputime;
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
    int N;        // total number of site
    int size;     // size of the array h_old, h_new

    double diff;
    float t, l, r, b;    // top, left, right, bottom
    int site, ym1, xm1, xp1, yp1;

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Write all the result to file.
    FILE *output;
    output = fopen("Output_CPU_site512.txt", "a");
    fprintf(output, "LatticeSize ");
    fprintf(output, "CPUerror CPUiteration CPUonly CPUflop\n");
    fclose(output);

    
    Nx = 512;
    Ny = 512;


    // Allocate field vector h_phi in host memory
    N = Nx * Ny;
    size = N * sizeof(float);
  
    h_old = (float*)malloc(size);
    h_new = (float*)malloc(size);
    
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
    out1 = fopen("phi_initial_CPU_site512.dat","w");

    //fprintf(out1, "Inital field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(out1,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(out1,"\n");
    }
    fclose(out1);

    printf("\n");


    // Compute with CPU only
    // 
    // start the timer
    cudaEventRecord(start,0);
    // to compute the reference solution
    error = 10*eps;      // any value bigger than eps 
    iter = 0;            // counter for iterations
    flag = true;

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

    }                   // exit if error < eps


    printf("error (CPU) = %.15e\n",error);
    printf("total iterations (CPU) = %d\n",iter);
    fflush(stdout);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    flops = 7.0*(Nx-2)*(Ny-2)*iter;
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));
    fflush(stdout);

    FILE *outc;                 // save CPU solution in phi_CPU.dat
    outc = fopen("phi_CPU_site512.dat","w");

    //fprintf(outc, "CPU field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(outc,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(outc,"\n");
    }
    fclose(outc);

    // Write all the output to file
    output = fopen("Output_CPU_site512.txt", "a");        
    fprintf(output, "%d %f %d %f %f\n", Nx, error, iter, cputime, flops/(1000000.0*cputime));
    fclose(output);

    printf("Finish computing lattice size\n");

    free(h_new);
    free(h_old);

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
    
}

