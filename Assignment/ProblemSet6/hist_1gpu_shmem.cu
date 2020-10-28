// To compute histogram with atomic operations */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


// Variables
float* data_h;          // host vectors
unsigned int* hist_h;   // GPU solution back to the CPU 
float* data_d;          // device vectors
unsigned int* hist_d;
unsigned int* hist_c;   // CPU solution


// Functions
void RandomUniform(float*, long);
void RandomNormal(float*, long);
void RandomExpDecay(float*, long);

__global__ void hist_shmem(float *data, const long N, unsigned int *hist, 
                           const int bins, const float Rmin, const float binsize) 
{

    // use shared memory and atomic addition

    extern __shared__  unsigned int temp[];     // assume the blocksize is equal to the total # bins
    temp[threadIdx.x] = 0;
    __syncthreads();

    long i = threadIdx.x + blockIdx.x * blockDim.x;
    long stride = blockDim.x * gridDim.x;

//    if( (index > bins-1) || (index < 0)) {
//      printf("data[%d]=%f, index=%d\n",i,data[i],index);
//    }

    while (i < N) {
        int index = (int)((data[i] - Rmin) / binsize);
        atomicAdd(&temp[index], 1);
        i += stride;
    }

    __syncthreads();
    atomicAdd( &(hist[threadIdx.x]), temp[threadIdx.x] );

}


int main(void)
{

    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    scanf("%d",&gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    cudaSetDevice(gid);

    printf("To find the histogram of a data set (with real numbers): \n");
    long N; 
    int bins,index;
    float Rmin, Rmax, binsize;

    printf("Enter the size of the data vector: ");
    scanf("%ld",&N);
    printf("%ld\n",N);
    long size = N * sizeof(float);

    printf("Enter the data range [Rmin, Rmax] for the histogram: ");
    scanf("%f %f",&Rmin, &Rmax);
    printf("%f %f\n",Rmin, Rmax);

    printf("Enter the number of bins of the histogram: ");
    scanf("%d",&bins);
    printf("%d\n",bins);
    if(bins > 1024) {
        printf("The number of bins is set to # of threads per block < 1024 ! \n");
        exit(0);
    }
    int bsize = bins*sizeof(int);
    binsize = (Rmax - Rmin)/(float)bins;
     
    data_h = (float*)malloc(size);
    hist_h = (unsigned int*)malloc(bsize);

    // Check memory allocations
    if(data_h == NULL || hist_h == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<bins; i++){
        hist_h[i]=0;
    }

    // initialize the data_h vector
    // srand(time(NULL));  // initialize the seed with the current time
    srand(12345);  

//    RandomUniform(data_h, N);      // uniform deviate in (0,1)
    RandomExpDecay(data_h, N);       

    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d",&threadsPerBlock);
    printf("%d\n",threadsPerBlock);
    if( threadsPerBlock != bins ) {
        printf("The number of threads per block must be equal to the number of bins ! \n");
        exit(0);
    }
    fflush(stdout);

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("%d\n",blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }
    printf("The number of blocks is %d\n", blocksPerGrid);
    fflush(stdout);

    int CPU;
    printf("To compute the histogram with CPU (1/0) ? ");
    scanf("%d",&CPU);
    printf("%d\n",CPU);
    fflush(stdout);


    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate vectors in device memory

    cudaMalloc((void**)&hist_d, bsize);
    cudaMalloc((void**)&data_d, size);

    // Copy vectors from host memory to device memory

    cudaMemcpy(data_d, data_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(hist_d, hist_h, bsize, cudaMemcpyHostToDevice);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime( &Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n",Intime);

   // start the timer
    cudaEventRecord(start,0);
    
    int sm = threadsPerBlock * sizeof(int);

    hist_shmem <<< blocksPerGrid, threadsPerBlock, sm >>> (data_d, N, hist_d, bins, Rmin, binsize);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime( &gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n",gputime);
    printf("GPU Gflops: %f\n",2*N/(1000000.0*gputime));

    // Copy result from device memory to host memory
    // hist_h contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(hist_h, hist_d, bsize, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(hist_d);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n", Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    // Save histogram in file
    FILE *out;
    out = fopen("hist_shmem.dat","w");

    fprintf(out, "Histogram (GPU):\n");
    for(int i=0; i<bins; i++) {
        float x = Rmin + (i + 0.5) * binsize;         // the center of each bin
        fprintf(out,"%f %d \n",x,hist_h[i]);
    }
    fclose(out);

    // Print the histogram on screen
    printf("Histogram (GPU):\n");
    for(int i = 0; i < bins; i = i+1) {
      float x = Rmin + (i + 0.5) * binsize;         // the center of each bin
      printf("%f %d\n", x, hist_h[i]);
    }

    if(CPU==0) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaDeviceReset();
      free(data_h);
      free(hist_h);
      return 0;
    }

    // To compute the CPU reference solution 

    hist_c = (unsigned int*)malloc(bsize);
    for(int i = 0; i < bins; i = i+1){
        hist_c[i] = 0;
    }

    // start the timer
    cudaEventRecord(start,0);

    for(int i = 0; i < N; i = i+1) {
        index = (int)((data_h[i] - Rmin) / binsize);
        if( (index > bins - 1) || (index < 0)) {
            printf("data[%d]=%f, index=%d\n",i,data_h[i],index);
            exit(0);
        } 
        hist_c[index]++;
    }

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",2*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check histogram sum equal to the total number of data 

    int sum = 0;
    for(int i = 0; i < bins; i = i+1) {
        sum += hist_c[i];
    }
    if(sum != N) {
        printf("Error, sum = %d\n",sum);
        exit(0);
    }

    // compare histograms from CPU and GPU

    for (int i = 0; i < bins; i++) {
        if(hist_h[i] != hist_c[i]) {
            printf("i=%d, hist_h=%d, hist_c=%d \n", i, hist_h[i], hist_c[i]);
        }
    }

    // save histogram in file
    FILE *out1;            
    out1 = fopen("hist_cpu.dat","w");

    fprintf(out1, "Histogram (CPU):\n");
    for(int i=0; i<bins; i++) {
        float x=Rmin+(i+0.5)*binsize;         // the center of each bin
        fprintf(out1,"%f %d \n",x,hist_c[i]);
    }
    fclose(out1);

    printf("Histogram (CPU):\n");
    for(int i=0; i<bins; i++) {
      float x=Rmin+(i+0.5)*binsize;         // the center of each bin
      printf("%f %d \n",x,hist_c[i]);
    }

    cudaDeviceReset();

    free(data_h);
    free(hist_h);
    free(hist_c);

    return 0;
}


void RandomUniform(float* data, long n)   // RNG with uniform distribution in (0,1)
{
    for(long i = 0; i < n; i++){
        data[i] = rand()/(float)RAND_MAX;
    }
}

void RandomNormal(float* data, long n)   // RNG with normal distribution, mu=0, sigma=1
{
    const float Pi = acos(-1.0);

    for(long i = 0; i < n; i++) {
        double y = (double) rand() / (float)RAND_MAX;
        double x = -log(1.0-y);
        double z = (double) rand() / (float)RAND_MAX;
        double theta = 2*Pi*z;
        data[i] = (float) (sqrt(2.0*x)*cos(theta));   
    }
}


void RandomExpDecay(float* data, long n)   // RNG with Exponential Decay
{
    for(long i = 0; i < n; i = i+1){
        double y = (double) rand() / (float) RAND_MAX;
        data[i] = (float) -log(1.0 - y);
    }
}
