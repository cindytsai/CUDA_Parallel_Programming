/*
Monte Carlo simulation of Ising model on 2D lattice with 2 GPU
using Metropolis algorithm
using checkerboard (even-odd) update 
Compile

nvcc -arch=compute_52 -code=sm_52 -O3 -m64 --compiler-options -fopenmp ising2d_Ngpu_gmem_v2.cu -lgsl -lgslcblas -lcurand -lcudadevrt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>

gsl_rng *rng=NULL;    // pointer to gsl_rng random number generator

void exact_2d(double, double, double*, double*);
void rng_MT(float*, int);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

__global__ void metro_gmem_odd(int* spin, int* spinL, int* spinR, int* spinB, int* spinT,
                               float *ranf, const float B, const float T)
{
    int    x, y, parity;
    int    i, io;
    int    old_spin, new_spin, spins;
    int    l, r, t, b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction of the sliced lattice
    int nx = 2*blockDim.x*gridDim.x;   // total number of sites in x-axis of the sliced lattice 
    int ny = blockDim.y * gridDim.y;   // total number of sites in y-axis of the sliced lattice 

    // Find the odd chestbox index 
    io = threadIdx.x + threadIdx.y*blockDim.x;   
    x = (2*io)%Nx;
    y = ((2*io)/Nx)%Nx;
    parity=(x+y+1)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice
    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;

    // Boundary of the sliced lattice at traverse
    if( x == 0 ){
      l = spinL[(nx-1) + y * nx];
      r = spin[i+1];
    }
    else if( x == nx - 1 ){
      l = spin[i-1];
      r = spinR[y * nx];
    }
    else{
      l = spin[i-1];
      r = spin[i+1];
    }

    // Boundary of the sliced lattice at longnitude
    if( y == 0 ){
      t = spin[i + nx];
      b = spinB[x + (ny - 1) * nx];
    }
    else if( y == ny - 1){
      t = spinT[x];
      b = spin[i - nx];
    }
    else{
      t = spin[i + nx];
      b = spin[i - nx];
    }

    spins = l + r + t + b;
    de = -((float)new_spin - (float)old_spin)*((float)spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }

    __syncthreads();

}


__global__ void metro_gmem_even(int* spin, int* spinL, int* spinR, int* spinB, int* spinT,
                               float *ranf, const float B, const float T)
{
    int    x, y, parity;
    int    i, ie;
    int    old_spin, new_spin, spins;
    int    l, r, t, b;
    float  de; 

    // thread index in a block of size (tx,ty) corresponds to 
    // the index ie/io of the lattice with size (2*tx,ty)=(Nx,Ny).
    // tid = threadIdx.x + threadIdx.y*blockDim.x = ie or io  
  
    int Nx = 2*blockDim.x;             // block size before even-odd reduction
    int nx = 2*blockDim.x*gridDim.x;   // number of sites in x-axis of the entire lattice 
    int ny = blockDim.y * gridDim.y;   // number of sites in y-axis of the sliced lattice 

    // first, go over the even sites 
    ie = threadIdx.x + threadIdx.y*blockDim.x;  
    x = (2*ie)%Nx;
    y = ((2*ie)/Nx)%Nx;
    parity=(x+y)%2;
    x = x + parity;  

    // add the offsets to get its position in the full lattice
    x += Nx*blockIdx.x;    
    y += blockDim.y*blockIdx.y;  

    i = x + y*nx;
    old_spin = spin[i];
    new_spin = -old_spin;

    // Boundary of the sliced lattice at traverse
    if( x == 0 ){
      l = spinL[(nx-1) + y * nx];
      r = spin[i+1];
    }
    else if( x == nx - 1 ){
      l = spin[i-1];
      r = spinR[y * nx];
    }
    else{
      l = spin[i-1];
      r = spin[i+1];
    }

    // Boundary of the sliced lattice at longnitude
    if( y == 0 ){
      t = spin[i + nx];
      b = spinB[x + (ny - 1) * nx];
    }
    else if( y == ny - 1){
      t = spinT[x];
      b = spin[i - nx];
    }
    else{
      t = spin[i + nx];
      b = spin[i - nx];
    }

    spins = l + r + t + b;
    de = -((float)new_spin - (float)old_spin)*((float)spins + B);
    if((de <= 0.0) || (ranf[i] < exp(-de/T))) {
      spin[i] = new_spin;       // accept the new spin;
    }
    
    __syncthreads();
 
}   

int main(void) {
  int NGPU;         // Number of GPU to use
  int NGx,NGy;      // The partition of the lattice (NGx*NGy=NGPU).
  int *Dev;         // GPU id array
  int cpu_thread_id;
  int cpuid_x, cpuid_y;
  int Lx, Ly;       // Lattice size in each GPU

  int *ffw;       // forward index
  int *bbw;       // backward index

  int nx,ny; 		// # of sites in x and y directions respectively
  int ns; 		// ns = nx*ny, total # of sites

  int *spin;
  float *h_rng;

  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2;           // right, top
  int istart; 		// istart = (0: cold start/1: hot start)
  double T; 		// temperature
  double B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >

  float gputime;
  float flops;

  cudaEvent_t start, stop;

  /*
  Get the GPU number to use and their ids.
   */
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

  /*
  Set up the environment
   */
  printf("Ising Model on 2D Square Lattice with p.b.c.\n");
  printf("============================================\n");
  printf("Initialize the RNG...\n");
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  printf("Enter the seed: ");
  long seed;
  scanf("%ld",&seed);
  printf("%ld\n",seed); 
  gsl_rng_set(rng,seed);
  printf("The RNG has been initialized.\n");  

  // Input numbers of site in each dim, and check the validity
  printf("Enter the number of sites in one dimension (<= 1000): ");
  scanf("%d",&nx);
  printf("%d\n",nx);
  ny = nx;
  ns = nx * ny;
  if (nx % NGx != 0) {
    printf("!!! Invalid partition of lattice: nx %% NGx != 0\n");
    exit(1);
  }
  if (ny % NGy != 0) {
    printf("!!! Invalid partition of lattice: ny %% NGy != 0\n");
    exit(1);
  }
  Lx = nx / NGx;
  Ly = ny / NGy;

  ffw = (int*)malloc(nx*sizeof(int));
  bbw = (int*)malloc(nx*sizeof(int));
  for(int i=0; i<nx; i++) {
    ffw[i]=(i+1)%nx;
    bbw[i]=(i-1+nx)%nx;
  }

  spin = (int*)malloc(ns*sizeof(int));          // host spin variables
  h_rng = (float*)malloc(ns*sizeof(float));     // host random numbers

  printf("Enter the # of sweeps for thermalization\n");
  scanf("%d",&nt);
  printf("%d\n",nt);
  printf("Enter the # of measurements\n");
  scanf("%d",&nm);
  printf("%d\n",nm);
  printf("Enter the interval between successive measurements\n");
  scanf("%d",&im);
  printf("%d\n",im);
  printf("Enter the display interval\n");
  scanf("%d",&nd);
  printf("%d\n",nd);
  printf("Enter the temperature (in units of J/k)\n");
  scanf("%lf",&T);
  printf("%lf\n",T);
  printf("Enter the external magnetization\n");
  scanf("%lf",&B);
  printf("%lf\n",B);
  printf("Initialize spins configurations :\n");
  printf(" 0: cold start \n");
  printf(" 1: hot start \n");
  scanf("%d",&istart);
  printf("%d\n",istart);
 
  // Set the number of threads (tx,ty) per block
  int tx,ty;
  printf("Enter the number of threads (tx,ty) per block: \n");
  printf("For even/odd updating, tx=ty/2 is assumed, (tx, ty): ");
  scanf("%d %d",&tx, &ty);
  printf("%d %d\n",tx, ty);
  if(2*tx != ty) {
    printf("Wrong tx, ty, since for even/odd updating, tx=ty/2 is assumed.\n");
    exit(0);
  }
  if(tx*ty > 1024) {
    printf("The number of threads per block must be less than 1024 ! \n");
    exit(0);
  }
  dim3 threads(tx,ty);

  // The total number of threads in the grid is equal to (nx/2)*ny = ns/2 
  int bx = nx/tx/2;
  if(bx*tx*2 != nx && (bx / NGx) * NGx != bx) {
    printf("The block size in x is incorrect\n");
    exit(0);
  }
  int by = ny/ty;
  if(by*ty != ny && (by / NGy) * NGy != by) {
    printf("The block size in y is incorrect\n");
    exit(0);
  }
  if((bx > 65535)||(by > 65535)) {
    printf("The grid size exceeds the limit ! \n");
    exit(0);
  }
  // Distributed to NGPU
  dim3 blocks(bx / NGx, by / NGy);
  printf("The dimension of the grid is (%d, %d)\n",bx,by);

  if(istart == 0) {
    for(int j=0; j<ns; j++) {       // cold start
      spin[j] = 1;
    }
  }
  else {
    for(int j=0; j<ns; j++) {     // hot start
      if(gsl_rng_uniform(rng) > 0.5) { 
        spin[j] = 1;
      }
      else {
        spin[j] = -1;
      }
    }
  }

  FILE *output;            
  output = fopen("ising2d_Ngpu_gmem.dat","w");

  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }
  fprintf(output,"     E           M        \n");
  fprintf(output,"--------------------------\n");

  printf("Thermalizing\n");
  printf("sweeps   < E >     < M >\n");
  printf("---------------------------------\n");
  fflush(stdout);

  te=0.0;                          //  initialize the accumulators
  tm=0.0;
  count=0.0;
  sweeps=nt+nm*im;                 //  total # of sweeps

  // Saving an array of GPU memory pointer that points to GPU memory, 
  // since there are many GPU
  int   **d_1;
  float **d_2;
  d_1 = (int **) malloc(NGPU * sizeof(int *));
  d_2 = (float **) malloc(NGPU * sizeof(float *));

  // One thread control one GPU
  omp_set_num_threads(NGPU);

  # pragma omp parallel private(cpu_thread_id, cpuid_x, cpuid_y)
  {
    // Set the GPU
    cpu_thread_id = omp_get_thread_num();
    cpuid_x = cpu_thread_id % NGx;
    cpuid_y = cpu_thread_id / NGx;
    cudaSetDevice(Dev[cpu_thread_id]);

    // Create the clock, and start the clock
    if(cpu_thread_id == 0){
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
    }

    // In order to activiate the P2P access of all the GPUs
    int cpuid_r = ((cpuid_x+1)%NGx) + cpuid_y*NGx;         // GPU on the right
    cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
    int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;     // GPU on the left
    cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
    int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;         // GPU on the top
    cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
    int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;     // GPU on the bottom
    cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);
    
    // Allocate spins vector, and rngs in device memory
    // d_1 -> spins
    // d_2 -> rngs
    cudaMalloc((void**)&d_1[cpu_thread_id], ns * sizeof(int) / NGPU);
    cudaMalloc((void**)&d_2[cpu_thread_id], ns * sizeof(float) / NGPU);

    // Copy spins from host memory to device memory
    // And since we allocate the device memory in 1D array,
    // we have to copy then line by line, even though they are seperate as blocks.
    for(int j = 0; j < Ly; j = j+1){
      int *h_temp, *d_temp;
      h_temp = spin + cpuid_x * Lx + (cpuid_y * Ly + j) * nx;
      d_temp = d_1[cpu_thread_id] + j * Lx;
      cudaMemcpy(d_temp, h_temp, Lx * sizeof(int), cudaMemcpyHostToDevice);
    }

    # pragma omp barrier
  }

  cudaSetDevice(Dev[0]);
  cudaEventRecord(start, 0);

  // Thermalization for "nt" steps
  for(int swp = 0; swp < nt; swp = swp+1){
    rng_MT(h_rng, ns);

    # pragma omp parallel private(cpu_thread_id, cpuid_x, cpuid_y)
    {
      // Set GPU
      cpu_thread_id = omp_get_thread_num();
      cpuid_x = cpu_thread_id % NGx;
      cpuid_y = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      // Find the pointer to the neighboring block
      // left, right, top, bottom
      int *dL, *dR, *dT, *dB;
      dL = d_1[(cpuid_x - 1 + NGx) % NGx + cpuid_y * NGx];
      dR = d_1[(cpuid_x + 1) % NGx + cpuid_y * NGx];
      dB = d_1[cpuid_x + ((cpuid_y - 1 + NGy) % NGy) * NGx];
      dT = d_1[cpuid_x + ((cpuid_y + 1) % NGy) * NGx];

      // Copy newly generate rng to device memory d_2
      cudaMemcpy(d_2[cpu_thread_id], h_rng + (ns / NGPU) * cpu_thread_id, (ns / NGPU) * sizeof(float), cudaMemcpyHostToDevice);

      // Update
      metro_gmem_even <<< blocks, threads >>> (d_1[cpu_thread_id], dL, dR, dB, dT, d_2[cpu_thread_id], B, T);
      
      # pragma omp barrier

      metro_gmem_odd <<< blocks, threads >>> (d_1[cpu_thread_id], dL, dR, dB, dT, d_2[cpu_thread_id], B, T);
      
      # pragma omp barrier
    }

  }

  // Measurements
  for(int swp = nt; swp < sweeps; swp = swp+1){
    
    rng_MT(h_rng, ns);

    /*
    Update the lattice
     */
    # pragma omp parallel private(cpu_thread_id, cpuid_x, cpuid_y)
    {
      // Set GPU
      cpu_thread_id = omp_get_thread_num();
      cpuid_x = cpu_thread_id % NGx;
      cpuid_y = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      // Find the neighbering device memory deivce pointer
      int *dL, *dR, *dT, *dB;
      dL = d_1[(cpuid_x - 1 + NGx) % NGx + cpuid_y * NGx];
      dR = d_1[(cpuid_x + 1) % NGx + cpuid_y * NGx];
      dB = d_1[cpuid_x + ((cpuid_y - 1 + NGy) % NGy) * NGx];
      dT = d_1[cpuid_x + ((cpuid_y + 1) % NGy) * NGx];

      // Copy newly generate rng to device memory
      cudaMemcpy(d_2[cpu_thread_id], h_rng + (ns / NGPU) * cpu_thread_id, (ns / NGPU) * sizeof(float), cudaMemcpyHostToDevice);

      // Update
      metro_gmem_even <<< blocks, threads >>> (d_1[cpu_thread_id], dL, dR, dB, dT, d_2[cpu_thread_id], B, T);
      
      # pragma omp barrier

      metro_gmem_odd <<< blocks, threads >>> (d_1[cpu_thread_id], dL, dR, dB, dT, d_2[cpu_thread_id], B, T);

      # pragma omp barrier
    }

    /*
    Do the measurements
     */
    if(swp % im == 0){

      // Copy spin lattice back to host memory      
      # pragma omp parallel private(cpu_thread_id, cpuid_x, cpuid_y)
      {
        cpu_thread_id = omp_get_thread_num();
        cpuid_x = cpu_thread_id % NGx;
        cpuid_y = cpu_thread_id / NGx;
        cudaSetDevice(Dev[cpu_thread_id]);

        int *h_temp, *d_temp;
        for(int j = 0; j < Ly; j = j+1){
          h_temp = spin + cpuid_x * Lx + (cpuid_y * Ly + j) * nx;
          d_temp = d_1[cpu_thread_id] + j * Lx;
          cudaMemcpy(h_temp, d_temp, Lx * sizeof(int), cudaMemcpyDeviceToHost);
        }

        # pragma omp barrier
      }

      // Measurement
      int k;
      mag = 0.0;
      energy = 0.0;
      for(int j = 0; j < ny; j = j+1){
        for(int i = 0; i < nx; i = i+1){
          k = i + j * nx;
          k1 = ffw[i] + j*nx;
          k2 = i + ffw[j]*nx;
          mag = mag + spin[k]; // total magnetization;
          energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;        }
        }
      }
      energy = energy - B*mag;
      te = te + energy;
      tm = tm + mag;
      count = count + 1.0;
      fprintf(output, "%.5e  %.5e\n", energy/(double)ns, mag/(double)ns);  // save the raw data 
    }

    /*
    Print the measurement on screen
     */
    if(swp % nd == 0){
      E = te/(count*(double)(ns));
      M = tm/(count*(double)(ns));
      printf("%d  %.5e  %.5e\n", swp, E, M);
    }

  }
  fclose(output);
  printf("---------------------------------\n");
  if(B == 0.0) {
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }

  // stop the timer, we set the timer on Dev[0]
  cudaSetDevice(Dev[0]);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&gputime, start, stop);
  printf("Processing time for GPU: %f (ms) \n",gputime);
  flops = 7.0*nx*nx*sweeps;
  printf("GPU Gflops: %lf\n",flops/(1000000.0*gputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  gsl_rng_free(rng);

  for(int i = 0; i < NGPU; i = i+1){
    cudaFree(d_1[i]);
    cudaFree(d_2[i]);

    cudaSetDevice(Dev[i]);
    cudaDeviceReset();
  }

  free(spin);
  free(h_rng);

  return 0;
}
          
          
// Exact solution of 2d Ising model on the infinite lattice

void exact_2d(double T, double B, double *E, double *M)
{
  double x, y;
  double z, Tc, K, K1;
  const double pi = acos(-1.0);
    
  K = 2.0/T;
  if(B == 0.0) {
    Tc = -2.0/log(sqrt(2.0) - 1.0); // critical temperature;
    if(T > Tc) {
      *M = 0.0;
    }
    else if(T < Tc) {
      z = exp(-K);
      *M = pow(1.0 + z*z,0.25)*pow(1.0 - 6.0*z*z + pow(z,4),0.125)/sqrt(1.0 - z*z);
    }
    x = 0.5*pi;
    y = 2.0*sinh(K)/pow(cosh(K),2);
    K1 = ellf(x, y);
    *E = -1.0/tanh(K)*(1. + 2.0/pi*K1*(2.0*pow(tanh(K),2) - 1.0));
  }
  else
    printf("Exact solution is only known for B=0 !\n");
    
  return;
}


/*******
* ellf *      Elliptic integral of the 1st kind 
*******/

double ellf(double phi, double ak)
{
  double ellf;
  double s;

  s=sin(phi);
  ellf=s*rf(pow(cos(phi),2),(1.0-s*ak)*(1.0+s*ak),1.0);

  return ellf;
}

double rf(double x, double y, double z)
{
  double rf,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4;
  ERRTOL=0.08; 
  TINY=1.5e-38; 
  BIG=3.0e37; 
  THIRD=1.0/3.0;
  C1=1.0/24.0; 
  C2=0.1; 
  C3=3.0/44.0; 
  C4=1.0/14.0;
  double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    
  if(min(x,y,z) < 0 || min(x+y,x+z,y+z) < TINY || max(x,y,z) > BIG) {
    printf("invalid arguments in rf\n");
    exit(1);
  }

  xt=x;
  yt=y;
  zt=z;

  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=THIRD*(xt+yt+zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } 
  while (max(abs(delx),abs(dely),abs(delz)) > ERRTOL);

  e2=delx*dely-pow(delz,2);
  e3=delx*dely*delz;
  rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
    
  return rf;
}

double min(double x, double y, double z)
{
  double m;

  m = (x < y) ? x : y;
  m = (m < z) ? m : z;

  return m;
}

double max(double x, double y, double z)
{
  double m;

  m = (x > y) ? x : y;
  m = (m > z) ? m : z;

  return m;
}

void rng_MT(float* data, int n)   // RNG with uniform distribution in (0,1)
{
    for(int i = 0; i < n; i++)
      data[i] = (float) gsl_rng_uniform(rng); 
}

