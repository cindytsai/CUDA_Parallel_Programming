/*
    Vector Dot Product, using multiple GPUs with OpenMP
    C = A.B
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* h_G;   // to store partial sum by process in each GPU, to prevent race condition

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecDot(const float* A, const float* B, float* C, int N)
{
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp += A[i]*B[i];
        i += blockDim.x*gridDim.x;  
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m

    int ib = blockDim.x/2;
    while (ib != 0) {
      if(cacheIndex < ib)
        cache[cacheIndex] += cache[cacheIndex + ib]; 

      __syncthreads();

      ib /=2;
    }
    
    if(cacheIndex == 0)
      C[blockIdx.x] = cache[0];
}
// Host code

int main(void)
{
    printf("Vector Dot Product with multiple GPUs \n");
    int N, NGPU, cpu_thread_id=0;
    int *Dev; 
    long mem = 1024*1024*1024;     // 4 Giga for float data type.

    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int)*NGPU);

    int numDev = 0;
    printf("GPU device number: ");
    for(int i = 0; i < NGPU; i++) {
      scanf("%d", &Dev[i]);
      printf("%d ",Dev[i]);
      numDev++;
      if(getchar() == '\n') break;
    }
    printf("\n");
    if(numDev != NGPU) {
      fprintf(stderr,"Should input %d GPU device numbers\n", NGPU);
      exit(1);
    }

    printf("Enter the size of the vectors: ");
    scanf("%d", &N);        
    printf("%d\n", N);        
    if (3*N > mem) {
        printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
        exit(1);
    }
    

    // Set the sizes of threads and blocks
    int threadsPerBlock, m;
    printf("Enter the number of threads per block (2^m), m : ");
    scanf("%d", &m);
    threadsPerBlock = pow(2, m);
    printf("Block Size = %d\n", threadsPerBlock);
   
    if(threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(1);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("Grid size = %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(1);
    }

    long size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);  // Output array from GPU
    int sm = threadsPerBlock*sizeof(float);  // GPU Shared Memory Size

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(sb * NGPU);
    h_G = (float*)malloc(sizeof(float) * NGPU);
    if (! h_A || ! h_B || ! h_C) {
        printf("!!! Not enough memory.\n");
        exit(1);
    }
    
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);
    for(int i = 0; i < NGPU; i = i+1){
        h_G[i] = 0.0;
    }

    // declare cuda event for timer
    cudaEvent_t start, stop;
//    cudaEventCreate(&start);    // events must be created after devices are set 
//    cudaEventCreate(&stop);

    float Intime,gputime,Outime;
    
    // Set numbers of threads = numbers of GPU
    omp_set_num_threads(NGPU);

    // So that "cpu_thread_id" is declared under each threads, and they are independent.
    // All omp thread do the same code in this block.
    #pragma omp parallel private(cpu_thread_id)
    {
    	float *d_A, *d_B, *d_C;
    	cpu_thread_id = omp_get_thread_num();
    	cudaSetDevice(Dev[cpu_thread_id]);


        // start the timer
        // And maybe since OpenMP thread id = 0 , start the first (?)
        // Start the clock here, to see how much time it takes to input array.
        // And also, we use a thread (here '0') to track the clock.
        if(cpu_thread_id == 0) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start,0);
        }

    	// Allocate vectors in device memory
        // N / NGPU should be divisible.
    	cudaMalloc((void**)&d_A, size/NGPU);
    	cudaMalloc((void**)&d_B, size/NGPU);
        // Since one threads handles one GPU
    	cudaMalloc((void**)&d_C, sb);

        // Copy vectors from host memory to device memory
    	cudaMemcpy(d_A, h_A+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_B, h_B+N/NGPU*cpu_thread_id, size/NGPU, cudaMemcpyHostToDevice);

        // Wait until all threads come to this step, synchronizes all threads on OpenMP
    	#pragma omp barrier

        // stop the timer
    	if(cpu_thread_id == 0) {
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime( &Intime, start, stop);
            printf("Data input time for GPU: %f (ms) \n",Intime);
    	}

        // start the timer
        if(cpu_thread_id == 0) cudaEventRecord(start,0);

        VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, d_C, N/NGPU);
        // Blocks until the device has completed all the preceding requested task.
    	cudaDeviceSynchronize();

        // stop the timer
    	if(cpu_thread_id == 0) {
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime( &gputime, start, stop);
            printf("Processing time for GPU: %f (ms) \n",gputime);
            printf("GPU Gflops: %f\n",3*N/(1000000.0*gputime));
    	}

        // Copy result from device memory to host memory
        // h_C contains the result in host memory
        // start the timer
        if(cpu_thread_id == 0){
            cudaEventRecord(start,0);
        } 
            
        cudaMemcpy(h_C+blocksPerGrid*cpu_thread_id, d_C, sb, cudaMemcpyDeviceToHost);
    	cudaFree(d_A);
    	cudaFree(d_B);
    	cudaFree(d_C);

        for(int i = blocksPerGrid * cpu_thread_id; i < (blocksPerGrid * cpu_thread_id) + blocksPerGrid; i = i+1) {
            h_G[cpu_thread_id] = h_G[cpu_thread_id] + h_C[i];
        }

        // Wait till OpenMP threads are finish!
        #pragma omp barrier
    }

    // Calculate the final result
    float DotGPU = 0.0;
    for(int i = 0; i < NGPU; i = i+1){
        DotGPU = DotGPU + h_G[i];
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &Outime, start, stop);
    printf("Data output time for GPU: %f (ms) \n",Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // Compute Dot Product by CPU

    // start the timer
    cudaEventRecord(start,0);

    double DotCPU = 0.0;   // compute the reference solution
    for (int i = 0; i < N; i = i+1) {
        DotCPU = DotCPU + (double)(h_A[i] * h_B[i]);
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/gputime_tot);

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("Check result:\n");
    printf("DotGPU = %f\n", DotGPU);
    printf("DotCPU = %f\n", DotCPU);
    double diff;
    diff = abs(DotCPU - (double)DotGPU);
    printf("abs(DotCPU - DotGPU)=%20.15e\n",diff);
    printf("error = abs(DotCPU - DotGPU) / DotCPU  = %20.15e\n", diff / DotCPU);

    for (int i=0; i < NGPU; i++) {
        cudaSetDevice(Dev[i]);
        cudaDeviceReset();
    }

    // Free all the vectors
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_G);

    return 0;
}


// Allocates an array with random float entries.
// From (0, 1)
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
