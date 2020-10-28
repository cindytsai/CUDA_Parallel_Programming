#include <stdio.h>
#include <stdlib.h>

// Variables
float** h_A;   // host vectors
float** h_B;
float** h_C;
float** h_D;
float** d_A;   // device vectors
float** d_B;
float** d_C;

// Functions
void RandomInit(float*, int, int);
__global__ void MatAdd(const float*, size_t, const float*, size_t, float*, size_t, int, int);

// Host code
int main(){
	// Settings
	// gid		-> GPU device id (0, 1, ...)
	// err		-> error message get from CUDA calls
	// N, M     -> Matrix size N x M
	// mem		-> to calculate maximum memory size
	// 
	// dimBlock, dimGrid -> Launching the kernel
	// block_size		 -> For temperory store the data
	// 
	// start, stop		-> CUDA event timer
	// Intime			-> Calculate the input time, allocate and move data in device memory
	// gputime			-> Time spent for doing the calculation
	// Outime			-> Copy result from device to host memory
	// gputime_tot		-> Time total spent
	// 
	// cputime  		-> Time spent for only using CPU
	// 
	// sum, diff		-> For checking the result from GPU
	int gid;
	cudaError_t err;
	int N, M;
	int mem = 1024 * 1024 * 1024;
	int block_size;
	cudaEvent_t start, stop;
	float Intime, gputime, Outime, gputime_tot;
	float cputime;
	double sum = 0, diff;

	// Select GPU device
    printf("Select the GPU with device ID: "); 
    scanf("%d", &gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
		printf("!!! Cannot select GPU with device ID = %d\n", gid);
	  	exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    // Input the size of Matrix A, B, size = N x M
    printf("Matrix Addition of N x M Matrix: C = A + B\n");
    printf("Enter the width (col) of the matrix: ");
    scanf("%d", &M);
    printf("Matrix width = %d\n", M);
    printf("Enter the height (row) of the matrix: ");
    scanf("%d", &N);
    printf("Matrix height = %d\n", N);
    // Check if the memory can fit the A,B,C matrix
    if(3 * N * M > mem){
    	printf("The size of these 3 matrices cannot be fitted into 4 Gbyte\n");
    	exit(2);
    }

    // Allocate 2D array for matrix A, B, C on host memory
    h_A = (float**) malloc(N * sizeof(float*));
    h_B = (float**) malloc(N * sizeof(float*));
    h_C = (float**) malloc(N * sizeof(float*));
    for(int i = 0; i < N; i = i+1){
    	h_A[i] = (float*) malloc(M * sizeof(float));
    	h_B[i] = (float*) malloc(M * sizeof(float));
    	h_C[i] = (float*) malloc(M * sizeof(float));
    }

    // Initialize the matrices
    RandomInit(h_A, N, M);
    RandomInit(h_B, N, M);

    // Input the block size, and compute the grid size
	// Enter the block size (threadsPerBlock)
	printf("Enter the number of square block size: ");
	scanf("%d", &block_size);
	printf("Block size = %d x %d\n", block_size, block_size);
	if(block_size * block_size > 1024){
		printf("The number of threads per block must be less than 1024 ! \n");
        exit(1);
	}
	dim3 dimBlock(block_size, block_size);

	// Calculate the grid size, so that every threads calculate one element in Matrix C
	dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
	printf("Grid size = %d x %d\n", dimGrid.y, dimGrid.x);
	if((dimGrid.x > 2147483647) || (dimGrid.y > 65535)){
		printf("The number of blocks per grid is too much ! \n");
		exit(1);
	}

    // Create the timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


    // Start the timer: Record allocate memory and move data, from host to device
	cudaEventRecord(start, 0);

    // Allocate 2D array of the matrices in device memory
    size_t* pitch_A;
    cudaMallocPitch((void**)&d_A, &pitch_A, N * sizeof(float), M);
    size_t* pitch_B;
    cudaMallocPitch((void**)&d_B, &pitch_B, N * sizeof(float), M);
    size_t* pitch_C;
    cudaMallocPitch((void**)&d_C, &pitch_C, N * sizeof(float), M);

    // Copy matrices from host to device memory
	cudaMemcpy2D(d_A, pitch_A, h_A, N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_B, pitch_B, h_B, N * sizeof(float), N * sizeof(float), M, cudaMemcpyHostToDevice);

	// Stop the timer: Record allocate memory and move data, from host to device
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate spend time: Record allocate memory and move data, from host to device
	cudaEventElapsedTime(&Intime, start, stop);
	printf("=================================\n");
	printf("Allocate memory and move data from host to device time spent for GPU: %f (ms) \n", Intime);


	// Start the timer: Matrix Addition calculation time
	cudaEventRecord(start, 0);

	// Call the kernel MatAdd
	MatAdd<<<dimGrid, dimBlock>>>(d_A, pitch_A, d_B, pitch_B, d_C, pitch_C, N, M);

	// Stop the timer: Matrix Addition calculation time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate spend time: Matrix Addition calculation time
	cudaEventElapsedTime(&gputime, start, stop);
	printf("Matrix Addition calculation time for GPU: %f (ms) \n", gputime);
	printf("GPU Gflops: %f\n", 3 * N / (1000000.0 * gputime));


	// Start the timer: Copy result from device memory to host memory
	cudaEventRecord(start, 0);

	// Copy result from device memory to host memory, and free the device memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// Stop the timer: Copy result from device memory to host memory
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate spend time: Copy result from device memory to host memory
	cudaEventElapsedTime( &Outime, start, stop);
    printf("Copy result from device memory to host memory time spent for GPU: %f (ms) \n", Outime);


    // Total time spent for GPU
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);


    // Start the timer: Time spent for just using CPU
    cudaEventRecord(start, 0);

    h_D = (float*) malloc(size);
    for(int i = 0; i < N * M; i = i+1){
    	h_D[i] = h_A[i] + h_B[i];
    }

    // Stop the timer: Time spent for just using CPU
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	// Calculate spend time: Time spent for just using CPU
	cudaEventElapsedTime(&cputime, start, stop);
	printf("---------------------------------\n");
    printf("Time spent for just using CPU: %f (ms) \n", cputime);
    printf("CPU Gflops: %f\n", 3 * N / (1000000.0 * cputime));
	printf("=================================\n");
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check the result
    printf("=================================\n");
    printf("Check the result:\n");
    for(int i = 0; i < N * M; i = i+1){
    	diff = abs(h_D[i] - h_C[i]);
    	sum = sum + diff * diff;
    	if(diff > 1.0e-15){
    		printf("row=%d, col=%d, h_D=%15.10e, h_C=%15.10e \n", i / M, i % M, h_D[i], h_C[i]);
    	}
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n", sum);

    cudaDeviceReset();
    return 0;
}

// Create a random array of size n, with elements between 0~1
void RandomInit(float** data, int n, int m){
	for(int i = 0; i < n; i = i+1){
		for(int j = 0; j < m; j = j+1){
			data[i][j] = rand() / (float)RAND_MAX;
		}
	}
}

// Device code
// Add matrix A and B, with size N x M together.
__global__ void MatAdd(const float* A, size_t pitch_A, const float* B, size_t pitch_B, float* C, size_t pitch_C, int N, int M){

	int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * M + col;
    if ((index < N * N) && (col < M) && (row < N)){
        C[index] = A[index] + B[index];
    }
    
    __syncthreads();

}



