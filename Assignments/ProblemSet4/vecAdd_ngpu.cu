// Vector addition: C = 1/A + 1/B 
// using multiple GPUs with OpenMP

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>          // header for OpenMP
#include <cuda_runtime.h>

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_C;
float* h_D;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = 1.0/A[i] + 1.0/B[i];
    __syncthreads();
}

// Host code

int main(void)
{
    printf("\n");
    printf("Vector Addition with multiple GPUs \n");
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
    long size = N*sizeof(float);

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if(threadsPerBlock > 1024) {
      printf("The number of threads per block must be less than 1024 ! \n");
      exit(1);
    }
    int blocksPerGrid = (N + threadsPerBlock*NGPU - 1) / (threadsPerBlock*NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if(blocksPerGrid > 2147483647) {
      printf("The number of blocks must be less than 2147483647 ! \n");
      exit(1);
    }

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    if (! h_A || ! h_B || ! h_C) {
	printf("!!! Not enough memory.\n");
	exit(1);
    }
    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // declare cuda event for timer
    cudaEvent_t start, stop;
//    cudaEventCreate(&start);    // events must be created after devices are set 
//    cudaEventCreate(&stop);

    float Intime,gputime,Outime;

    // Set numbers of threads = numbers of GPU
    omp_set_num_threads(NGPU);

    // So that "cpu_thread_id" is declared under each threads, and they are independent.
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
        // size should be devisiable by Number of GPU 
    	cudaMalloc((void**)&d_A, size/NGPU);
    	cudaMalloc((void**)&d_B, size/NGPU);
    	cudaMalloc((void**)&d_C, size/NGPU);

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

        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N/NGPU);
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
        if(cpu_thread_id == 0) cudaEventRecord(start,0);
        cudaMemcpy(h_C+N/NGPU*cpu_thread_id, d_C, size/NGPU, cudaMemcpyDeviceToHost);
    	cudaFree(d_A);
    	cudaFree(d_B);
    	cudaFree(d_C);

        // stop the timer
    	if(cpu_thread_id == 0) {
            cudaEventRecord(stop,0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime( &Outime, start, stop);
            printf("Data output time for GPU: %f (ms) \n",Outime);
    	}
    }

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n",gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    h_D = (float*)malloc(size);     // compute the reference solution
    for (int i = 0; i < N; ++i) {
        h_D[i] = 1.0/h_A[i] + 1.0/h_B[i];
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
    double sum=0; 
    double diff;
    for (int i = 0; i < N; ++i) {
        diff = abs(h_D[i] - h_C[i]);
        sum += diff*diff; 
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n",sum);

    for (int i=0; i < NGPU; i++) {
	cudaSetDevice(i);
	cudaDeviceReset();
    }

    return 0;
}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
