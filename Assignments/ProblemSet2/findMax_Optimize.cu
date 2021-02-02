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
	// gid		-> GPU device id (0, 1, ...)
	// err		-> error message get from CUDA calls
	// N 		-> Length of an array
	// size 	-> memory size of the allocate array
	// sb 		-> memory size after handle by GPU
	// sm 		-> size of shared memory in each individual block
	// m  		-> the power of threadsPerBlock
	// threadsPerBlock, blocksPerGrid -> For launching kernel
	// 
	// start, stop		-> CUDA event timer
	// Intime			-> Calculate the input time, allocate and move data in device memory
	// gputime			-> Time spent in GPU only
	// Outime			-> Time used to handle the rest of finding maximum
	// gputime_tot		-> Time total spent
	// 
	// max_value		-> Maximum value inside this array, find by GPU
	int gid;
	cudaError_t err;
	int N;
	int size, sb;
	int sm;
	int threadsPerBlock, blocksPerGrid;
	cudaEvent_t start, stop;
	float Intime, gputime, Outime, gputime_tot, cputime;
    float max_value;
    float max_value_CPU;
    FILE *output;

    // Optimize block size and grid size , with array length N.
    N = 81920007;
    size = N * sizeof(float);

	// Select GPU device
    printf("Select the GPU with device ID: "); 
    scanf("%d", &gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
		printf("!!! Cannot select GPU with device ID = %d\n", gid);
	  	exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    // Create the timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    output = fopen("optimize_result.txt", "a");
    fprintf(output, "BlockSize GridSize GPUonly TotalGPU TotalCPU SpeedUp Check\n");
    fclose(output);

    // m -> the power of the block size
    // g -> the power of the grid size
    for(int m = 1; m <= 10; m = m+1){
        for(int g = 1; g <= 4; g = g+1){

            Intime = 0.0;
            gputime = 0.0;
            Outime = 0.0;
            gputime_tot = 0.0;
            cputime = 0.0;
            max_value = -2.0;
            max_value_CPU = -2.0;

            threadsPerBlock = pow(2, m);
            blocksPerGrid = pow(10, g);
            printf("%4d    %7d\n", threadsPerBlock, blocksPerGrid);

            // Allocate input array
            sb = blocksPerGrid * sizeof(float);
            h_A = (float*)malloc(size);
            h_C = (float*)malloc(sb);
            
            // Initialize input vectors
            RandomInit(h_A, N);

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
            gputime_tot = Intime + gputime + Outime;

            // start the timer
            cudaEventRecord(start, 0);

            // Compute the reference solution
            for(int i = 0; i < N; i = i+1){
                if(h_A[i] > max_value_CPU){
                    max_value_CPU = h_A[i];
                }
            }
            
            // stop the timer
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&cputime, start, stop);

            free(h_A);
            free(h_C);

            // Write to file
            output = fopen("optimize_result.txt", "a");
            fprintf(output, "%d %d %f %f %f %f ", threadsPerBlock, blocksPerGrid, gputime, gputime_tot, cputime, cputime/gputime_tot);
            fprintf(output, "%.23f\n", max_value - max_value_CPU);
            fclose(output);
            

        }
    }


    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Reset the device
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