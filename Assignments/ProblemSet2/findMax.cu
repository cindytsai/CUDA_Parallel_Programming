#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;   // host vectors
float* h_C;
float* d_A;   // device vectors
float* d_C;

// Functions
void RandomInit(float*, int);
__global__ void FindMax(const float*, float*, int);

// Host Code
int main(){
    // Settings
    // gid      -> GPU device id (0, 1, ...)
    // err      -> error message get from CUDA calls
    // N        -> Length of an array
    // size     -> memory size of the allocate array
    // sb       -> memory size after handle by GPU
    // sm       -> size of shared memory in each individual block
    // m        -> the power of threadsPerBlock
    // threadsPerBlock, blocksPerGrid -> For launching kernel
    // 
    // start, stop      -> CUDA event timer
    // Intime           -> Calculate the input time, allocate and move data in device memory
    // gputime          -> Time spent in GPU only
    // Outime           -> Time used to handle the rest of finding maximum
    // gputime_tot      -> Time total spent
    // 
    // max_value        -> Maximum value inside this array, find by GPU
    // max_value_CPU    -> Maximum value inside this array, find by CPU
    int gid;
    cudaError_t err;
    int N;
    int size, sb;
    int sm;
    int m, threadsPerBlock, blocksPerGrid;
    cudaEvent_t start, stop;
    float Intime, gputime, Outime, gputime_tot, cputime;
    float max_value = -2.0;
    float max_value_CPU = -2.0;


    // Select GPU device
    printf("Select the GPU with device ID: "); 
    scanf("%d", &gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    // Set the size of the vector
    printf("Find maximum value within one array:\n");
    printf("Enter the length of an array: ");
    scanf("%d", &N);
    printf("Array length = %d\n", N);

    // Set blocksize and grid size
    printf("Enter the power (m) of threads per block (2^m): ");
    scanf("%d", &m);
    threadsPerBlock = pow(2, m);
    printf("Threads per block = %d\n", threadsPerBlock);
    if(threadsPerBlock > 1024){
        printf("The number of threads per block must be less than 1024 (2^m , m <=10) ! \n");
        exit(0);
    }

    printf("Enter the number of blocks per grid: ");
    scanf("%d",&blocksPerGrid);
    printf("Blocks per grid = %d\n", blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(0);
    }

    // Allocate input array
    size = N * sizeof(float);
    sb = blocksPerGrid * sizeof(float);
    h_A = (float*)malloc(size);
    h_C = (float*)malloc(sb);

    // Initialize input vectors
    RandomInit(h_A, N);

    // Create the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Start the timer: Record allocate memory and move data, from host to device
    cudaEventRecord(start, 0);

    // Allocate the array in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_C, sb);

    // Copy array from host to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Stop the timer: Record allocate memory and move data, from host to device
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate spend time: Record allocate memory and move data, from host to device
    cudaEventElapsedTime(&Intime, start, stop);
    printf("=================================\n");
    printf("Allocate memory and move data from host to device time spent for GPU: %f (ms) \n", Intime);

    // start the timer
    cudaEventRecord(start, 0);

    // Called the kernel
    sm = threadsPerBlock * sizeof(float);
    FindMax <<< blocksPerGrid, threadsPerBlock, sm >>>(d_A, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate spend time: Matrix Addition calculation time
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Time used for GPU: %f (ms) \n", gputime);
    
    
    
    // start the timer
    cudaEventRecord(start,0);
    
    // Copy result from device memory to host memory
    // h_C contains the result of each block in host memory
    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);

    for(int i = 0; i < blocksPerGrid; i = i+1){
        if(h_C[i] > max_value){
            max_value = h_C[i];
        }
    }

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime( &Outime, start, stop);
    printf("Time used to handle the rest of finding maximum: %f (ms) \n", Outime);

    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    // Compute the reference solution
    for(int i = 0; i < N; i = i+1){
        if(h_A[i] > max_value_CPU){
            max_value_CPU = h_A[i];
        }
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&cputime, start, stop);
    printf("---------------------------------\n");
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("=================================\n");
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check the result
    printf("=================================\n");
    printf("Check the result:\n");
    printf("Maximum find by GPU = %.23f\n", max_value);
    printf("Maximum find by CPU = %.23f\n", max_value_CPU);

    free(h_A);
    free(h_C);

    cudaDeviceReset();

    return 0;
}

__global__ void FindMax(const float* A, float* C, int N){
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float max = -2.0;

    while (i < N) {
        if(A[i] > max){
            max = A[i];
        }
        i = i + blockDim.x * gridDim.x;  
    }
   
    cache[cacheIndex] = max;

    __syncthreads();

    // Perform parallel reduction, threadsPerBlock must be 2^m
    int ib = blockDim.x/2;
    while (ib != 0) {
        if(cacheIndex < ib){
            if(cache[cacheIndex] < cache[cacheIndex + ib]){
                cache[cacheIndex] = cache[cacheIndex + ib];
            }
        }

        __syncthreads();

        ib = ib / 2;
    }
    
    if(cacheIndex == 0){
        C[blockIdx.x] = cache[0];
    }

}

// Allocates an array with random float entries in (-1,1)
void RandomInit(float* data, int n){
    for (int i = 0; i < n; ++i)
        data[i] = 2.0*rand()/(float)RAND_MAX - 1.0;
}