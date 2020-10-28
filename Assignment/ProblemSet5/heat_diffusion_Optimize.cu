// Solve the Laplace equation on a 2D lattice with boundary conditions.
//


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

// field variables
float*  h_new;			// pointer to the working space
float*  h_old;			// pointer to the working space
float*  h_1;			// host field vectors (working space)
float*  h_2;			// host field vectors (working space)
float*  h_C;			// sum of diff*diff of each block
float*  g_new;			// device solution back to the host 
float** d_1;			// device field vectors (working space)
float** d_2;			// device field vectors (working space)
float** d_C;			// sum of diff*diff of each block 

int     MAX=10000000;		// maximum iterations
double  eps=1.0e-10;		// stopping criterion


__global__ void
laplacian(float* phi0_old, float* phiL_old, float* phiR_old, float* phiB_old,
	  float* phiT_old, float* phi0_new, float* C, float omega)
{
    extern __shared__ float cache[];     
    float  t, l, c, r, b;     // top, left, center, right, bottom
    float  diff; 
    int    site, skip;

    int Lx = blockDim.x*gridDim.x;
    int Ly = blockDim.y*gridDim.y;
    int x  = blockDim.x*blockIdx.x + threadIdx.x;
    int y  = blockDim.y*blockIdx.y + threadIdx.y;
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;  

    site = x + y*Lx;
    skip = 0;
    diff = 0.0;
    b = 0.0;
    l = 0.0;
    r = 0.0;
    t = 0.0;
    c = phi0_old[site];
    
    if (x == 0) {
      if (phiL_old != NULL) {
        l = phiL_old[(Lx-1)+y*Lx];
        r = phi0_old[site+1];
      } 
      else {
        skip = 1;
      }
    }
    else if (x == Lx-1) {
      if(phiR_old != NULL) {
        l = phi0_old[site-1];
        r = phiR_old[y*Lx];
      } 
      else {
        skip = 1;
      }
    }
    else {
      l = phi0_old[site-1];		// x-1
      r = phi0_old[site+1];		// x+1
    }

    if (y == 0) {
      if (phiB_old != NULL) {
         b = phiB_old[x+(Ly-1)*Lx];
         t = phi0_old[site+Lx];
      } 
      else {
	      skip = 1;
      }
    }
    else if (y == Ly-1) {
      if (phiT_old != NULL) {
        b = phi0_old[site-Lx];
        t = phiT_old[x];
      } 
      else {
        skip = 1;
      }
    }
    else {
      b = phi0_old[site-Lx];
      t = phi0_old[site+Lx];
    }

    if (skip == 0) {
      phi0_new[site] = 0.25 * omega * (b + l + r + t) + (1 - omega) * c;
      diff = phi0_new[site]-c;
    }

    // each thread saves its error estimate to the shared memory
    cache[cacheIndex]=diff*diff;  
    __syncthreads();

    // parallel reduction in each block 

    int ib = blockDim.x*blockDim.y/2;  
    while (ib != 0) {  
      if(cacheIndex < ib){
        cache[cacheIndex] += cache[cacheIndex + ib];
      }
      __syncthreads();
      ib /=2;  
    } 

    // save the partial sum of each block to C
    int blockIndex = blockIdx.x + gridDim.x*blockIdx.y;
    if(cacheIndex == 0)  C[blockIndex] = cache[0];    
}


int main(void)
{
    volatile bool flag;		// to toggle between two working spaces.    
    int      cpu_thread_id=0;
    int      NGPU;
    int     *Dev;		// GPU device numbers.
    int      Nx,Ny;		// lattice size in x and y directions.
    int      Lx,Ly;		// lattice size in each GPU.
    int      NGx,NGy;		// The partition of the lattice (NGx*NGy=NGPU).
    int      tx,ty;		// block size: tx*ty
    int      bx,by;		// # of blocks.
    int      sm;		// size of the shared memory in each block.
    int      iter;		// counter of iterations.
    float    gputime;
    float    gputime_tot;
    float    Intime,Outime;
    double   flops;
    double   error;		// error estimate 
    cudaEvent_t start, stop;
    float    omega;

 //  Get the input parameters.

    printf("* Initial parameters:\n");
    printf("  Enter the number of GPUs (NGx, NGy): ");
    scanf("%d %d", &NGx, &NGy);
    printf("%d %d\n", NGx, NGy);
    NGPU = NGx * NGy;
    Dev  = (int *)malloc(sizeof(int)*NGPU);
    for (int i=0; i < NGPU; i++) {
      printf("  * Enter the GPU ID (0/1/...): ");
      scanf("%d",&(Dev[i]));
      printf("%d\n", Dev[i]);
    }

    printf("  Solve Heat Diffusion on 2D lattice with boundary conditions\n");
    printf("  Enter the size (Nx, Ny) of the 2D lattice: ");
    scanf("%d %d",&Nx,&Ny);        
    printf("%d %d\n",Nx,Ny);        
    if (Nx % NGx != 0) {
      printf("!!! Invalid partition of lattice: Nx %% NGx != 0\n");
      exit(1);
    }
    if (Ny % NGy != 0) {
      printf("!!! Invalid partition of lattice: Ny %% NGy != 0\n");
      exit(1);
    }
    Lx = Nx / NGx;
    Ly = Ny / NGy;

    printf("  Enter omega of the alogrithm: ");
    scanf("%f", &omega);
    printf("%.3f\n", omega);

    int N, size, sb;
    N = Nx * Ny;
    size = N * sizeof(float);

    FILE *output;
    output = fopen("HeatDiffusion_result.txt", "a");
    fprintf(output, "BlockSize GPUonly TotalGPU Error\n");
    fclose(output);    

    for(int n = 2; n <= 5; n = n+1){
      tx = pow(2, n);
      ty = tx;
      bx = Nx/tx;
      by = Ny/ty;

      // equally seperate the blocks to NGPU
      dim3 threads(tx,ty); 
      dim3 blocks(bx/NGx,by/NGy);
      
      error = 10*eps;      // any value bigger than eps is OK
      flag  = true;
      
      // Allocate field vector h_phi in host memory
      sb = bx * by * sizeof(float);
      h_1   = (float*)malloc(size);
      h_2   = (float*)malloc(size);
      h_C   = (float*)malloc(sb);
      g_new = (float*)malloc(size);

      // Initialize the field vector with boundary conditions
      memset(h_1, 0, size);    
      memset(h_2, 0, size);
      for(int x = 0; x < Nx; x = x+1) {
        h_1[x+Nx*(Ny-1)]=400.0;  
        h_2[x+Nx*(Ny-1)]=400.0;
        h_1[x] = 273.0;
        h_2[x] = 273.0;
      } 
      for(int y = 0; y < Ny; y = y+1){
        h_1[y * Nx] = 273.0;
        h_2[y * Nx] = 273.0;
        h_1[(Nx - 1) + y * Nx] = 273.0;
        h_2[(Nx - 1) + y * Nx] = 273.0;
      }

      // Save initial configuration in phi_initial.dat 
      FILE *out1;
      if ((out1 = fopen("phi_initial.dat","w")) == NULL) {
        printf("!!! Cannot open file: phi_initial.dat\n");
        exit(1);
      }
      fprintf(out1, "Inital field configuration:\n");
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(out1,"%.2e ", h_1[i+j*Nx]);
        }
        fprintf(out1,"\n");
      }
      fclose(out1);

      //  Allocate working space for GPUs.
      printf("\n* Allocate working space for GPUs ....\n");
      sm = tx * ty * sizeof(float); // size of the shared memory in each block

      // For saving an array of GPU memory pointer that points to GPU memory, since there are many GPU
      d_1 = (float **)malloc(NGPU*sizeof(float *));
      d_2 = (float **)malloc(NGPU*sizeof(float *));
      d_C = (float **)malloc(NGPU*sizeof(float *));
      omp_set_num_threads(NGPU);
      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x       = cpu_thread_id % NGx;
        cpuid_y       = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);
  
        // In order to activiate the P2P access of all the GPUs
        int cpuid_r = ((cpuid_x+1)%NGx) + cpuid_y*NGx;         // GPU on the right
        cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
        int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;     // GPU on the left
        cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
        int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;         // GPU on the top
        cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
        int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;     // GPU on the bottom
        cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);
  
        // start the timer
        if (cpu_thread_id == 0) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
        }
  
        // Allocate vectors in device memory
        // d1 array is for saving the return pointer from cudaMalloc
        cudaMalloc((void**)&d_1[cpu_thread_id], size/NGPU);
        cudaMalloc((void**)&d_2[cpu_thread_id], size/NGPU);
        cudaMalloc((void**)&d_C[cpu_thread_id], sb/NGPU);
  
        // Copy vectors from the host memory to the device memory
        // Copy the 2D array in 1D each by each
        for (int i=0; i < Ly; i++) {
          float *h, *d;
          // pointer calculus
          h = h_1 + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_1[cpu_thread_id] + i*Lx;
          cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
        }
        for (int i=0; i < Ly; i++) {
          float *h, *d;
          h = h_2 + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_2[cpu_thread_id] + i*Lx;
          cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
        }
  
        #pragma omp barrier
  
        // stop the timer
        if (cpu_thread_id == 0) {
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Intime, start, stop);
            printf("  Data input time for GPU: %f (ms) \n",Intime);
        }
      } // OpenMP

      //  Compute GPU solution.
      // start the timer
      cudaEventRecord(start,0);
      printf("\n* Compute GPU solution ....\n");
      fflush(stdout);

      iter = 0;        // counter for iterations
      while ((error > eps) && (iter < MAX)) {
          #pragma omp parallel private(cpu_thread_id)
          {
            int cpuid_x, cpuid_y;
            cpu_thread_id = omp_get_thread_num();
            cpuid_x       = cpu_thread_id % NGx;
            cpuid_y       = cpu_thread_id / NGx;
            cudaSetDevice(Dev[cpu_thread_id]);

            float **d_old, **d_new;
            float *dL_old, *dR_old, *dT_old, *dB_old, *d0_old, *d0_new;
            d_old  = (flag == true) ? d_1 : d_2;
            d_new  = (flag == true) ? d_2 : d_1;
            d0_old = d_old[cpu_thread_id];           
            d0_new = d_new[cpu_thread_id];

            // If the GPU contains the boundary, then return NULL, 
            // otherwise return the corresponding GPU
            dL_old = (cpuid_x == 0)     ? NULL : d_old[cpuid_x-1+cpuid_y*NGx];
            dR_old = (cpuid_x == NGx-1) ? NULL : d_old[cpuid_x+1+cpuid_y*NGx];
            dB_old = (cpuid_y == 0    ) ? NULL : d_old[cpuid_x+(cpuid_y-1)*NGx];
            dT_old = (cpuid_y == NGy-1) ? NULL : d_old[cpuid_x+(cpuid_y+1)*NGx];

            laplacian<<<blocks,threads,sm>>>(d0_old, dL_old, dR_old, dB_old,
                                             dT_old, d0_new, d_C[cpu_thread_id], omega);
            cudaDeviceSynchronize();

            cudaMemcpy(h_C+bx*by/NGPU*cpu_thread_id, d_C[cpu_thread_id], sb/NGPU,
                       cudaMemcpyDeviceToHost);
          } // OpenMP

          error = 0.0;
          for(int i=0; i<bx*by; i++)
            error = error + h_C[i];

          error = sqrt(error);

          iter++;
          flag = !flag;
      }

      printf("  error (GPU) = %.15e\n",error);
      printf("  total iterations (GPU) = %d\n",iter);
      fflush(stdout);
      
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &gputime, start, stop);
      flops = 7.0*(Nx-2)*(Ny-2)*iter;
      printf("  Processing time for GPU: %f (ms) \n",gputime);
      printf("  GPU Gflops: %f\n",flops/(1000000.0*gputime));
      fflush(stdout);
      
      //  Copy result from device memory to host memory
      // start the timer
      cudaEventRecord(start,0);
      printf("\n* Copy result from device memory to host memory ....\n");
      fflush(stdout);
      
      #pragma omp parallel private(cpu_thread_id)
      {
        int cpuid_x, cpuid_y;
        cpu_thread_id = omp_get_thread_num();
        cpuid_x       = cpu_thread_id % NGx;
        cpuid_y       = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);
  
        float* d_new = (flag == true) ? d_2[cpu_thread_id] : d_1[cpu_thread_id];
        for (int i=0; i < Ly; i++) {
          float *g, *d;
          g = g_new + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_new + i*Lx;
          cudaMemcpy(g, d, Lx*sizeof(float), cudaMemcpyDeviceToHost);
        }
        cudaFree(d_1[cpu_thread_id]);
        cudaFree(d_2[cpu_thread_id]);
        cudaFree(d_C[cpu_thread_id]);
      } // OpenMP
  
      // stop the timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime( &Outime, start, stop);
      gputime_tot = Intime + gputime + Outime;
      printf("  Data output time for GPU: %f (ms) \n",Outime);
      printf("  Total time for GPU: %f (ms) \n",gputime_tot);
      fflush(stdout);

      // Save GPU solution in phi_GPU.dat
      FILE *outg;
      if ((outg = fopen("phi_GPU.dat","w")) == NULL) {
        printf("!!! Cannot open file: phi_GPU.dat\n");
        exit(1);
      }
      fprintf(outg, "GPU field configuration:\n");
      for(int j=Ny-1;j>-1;j--) {
        for(int i=0; i<Nx; i++) {
          fprintf(outg,"%.2e ",g_new[i+j*Nx]);
        }
        fprintf(outg,"\n");
      }
      fclose(outg);      
    
      free(h_new);
      free(h_old);
      free(g_new);
      free(d_1);
      free(d_2);
      free(d_C);


      output = fopen("HeatDiffusion_result.txt", "a");
      fprintf(output, "%d %f %f %.15e\n", tx, gputime, gputime_tot, error);
      fclose(output); 
    }

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    #pragma omp parallel private(cpu_thread_id)
    {
      cpu_thread_id = omp_get_thread_num();
      cudaSetDevice(Dev[cpu_thread_id]);
      cudaDeviceReset();
    } // OpenMP

    return 0;
}
