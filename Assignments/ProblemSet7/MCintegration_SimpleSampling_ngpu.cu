/*
Compile
 nvcc -arch=compute_52 -code=sm_52,sm_52 -m64 --compiler-options -fopenmp -o q2-SS.exe MCintegration_SimpleSampling_ngpu.cu
 */

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<unistd.h>



float* h_mean;		// Store vector from d_mean
float* h_sigma;		// Store vector from d_sigma
float* hh_mean;		// to store the sum up h_mean per thread
float* hh_sigma;	// to store the sum up h_sigma per thread

// Host code
double f(double*);

// Device code
__global__ void SimpleSampling(int N, float* mean, float* sigma, unsigned int seed);


int main(void){
	// Setting up random number generator
	srand(101);

	// Settings for CPU
	// Sampling N points
	// N = pow(2, n);
	// n = 1, 2 , ..., 16
	// array x --> holds the random number for coordinate. 
	// Store temperary random number in r, y.
	double mean, sigma;
	int N;
	double x[10];
	
	FILE *output;
	float gputime_tot, cputime;

	// Settings for omp
	int cpu_thread_id = 0;

	// Settings for GPU
	int NGPU;	// numbers of GPU
	int *Dev;	// Store the GPU id
	int m, threadsPerBlock, blocksPerGrid;
	int sb;		// Size of output array from individual GPU
	int sm;		// Size of the shared memory per block
	cudaEvent_t start, stop;


	// Get the settings of GPU
	printf("\n* Initial parameters for GPU:\n");
    printf("  Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int) * NGPU);
    for (int i = 0; i < NGPU; i = i+1) {
      	printf("  Enter the GPU ID (0/1/...): ");
      	scanf("%d",&(Dev[i]));
      	printf("%d\n", Dev[i]);
    }

    printf("\n* Solve Monte Carlo Integration in 10-dim:\n");
    printf("  Enter the number of sample points: ");
    scanf("%d",&N);        
    printf("%d\n",N);

    // Check can N be divided by NGPU, for saving my time on how to distribute workload!!!
    if (N % NGPU != 0) {
      	printf("!!! Invalid partition of lattice: N %% NGPU != 0\n");
      	exit(1);
    }

    // Set the number of threads per block
    // Since I would use parallel reduction , threads per block should be 2^m
    printf("  Enter the power (m) of threads per block (2^m): ");
    scanf("%d", &m);
    printf("%d\n", m);
    threadsPerBlock = pow(2, m);
    if( threadsPerBlock > 1024 ) {
      	printf("!!! The number of threads per block must be less than 1024.\n");
      	exit(0);
    }
	printf("threads per block = %d\n", threadsPerBlock);

    printf("  Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);
    if( blocksPerGrid > 2147483647 ) {
      	printf("!!! The number of blocks per grid must be less than 2147483647.\n");
      	exit(0);
    }

    if( (N / NGPU) < blocksPerGrid * threadsPerBlock){
        printf("!!! The number of threads per grid must be less than number of sample points N / NGPU.\n");
        exit(0);
    }

	// Output to a file
	output = fopen("integration_result_SS.txt", "a");
	fprintf(output, "N BlockSize GridSize SSGPU SSGPUerror SSCPU SSCPUerror SpeedUp\n");
	fclose(output);    

	/*
	Monte Carlo integration with GPU
	 */
	
	sb = blocksPerGrid * sizeof(float);		
	sm = threadsPerBlock * sizeof(float);
	h_mean 	 = (float*)malloc(sb * NGPU);
	h_sigma  = (float*)malloc(sb * NGPU);
	hh_mean  = (float*)malloc(sizeof(float) * NGPU);
	hh_sigma = (float*)malloc(sizeof(float) * NGPU);

    for(int i = 0; i < NGPU; i = i+1){
        hh_mean[i]  = 0.0;
        hh_sigma[i] = 0.0;
    }

    // Set numbers of threads = numbers of GPU
    omp_set_num_threads(NGPU);

    #pragma omp parallel private(cpu_thread_id)
    {
    	// Declare private pointer
    	float* d_mean;
    	float* d_sigma;
    	// Get thread num, and set the GPU accordingly
    	cpu_thread_id = omp_get_thread_num();
    	cudaSetDevice(Dev[cpu_thread_id]);

    	// Create the timer and start it at thread = 0 only
        if(cpu_thread_id == 0) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        // Allocate device memory
        cudaMalloc((void**)&d_mean, sb);
        cudaMalloc((void**)&d_sigma, sb);

        // Since I need to stored mean and sigma seperately,
        // so shared memory must x2
        // set the seed as time(NULL) + cpu_thread_id * 10.0 so that different gpu has different seed
        SimpleSampling<<<blocksPerGrid, threadsPerBlock, 2 * sm>>>(N / NGPU, d_mean, d_sigma, time(NULL) + cpu_thread_id * 10.0);

        // Copy d_mean and d_sigma from device to host
        cudaMemcpy(h_mean + blocksPerGrid * cpu_thread_id, d_mean, sb, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sigma + blocksPerGrid * cpu_thread_id, d_sigma, sb, cudaMemcpyDeviceToHost);

       	cudaFree(d_mean);
       	cudaFree(d_sigma);

        for(int i = blocksPerGrid * cpu_thread_id; i < (blocksPerGrid * cpu_thread_id) + blocksPerGrid; i = i+1) {
            hh_mean[cpu_thread_id] = hh_mean[cpu_thread_id] + h_mean[i];
            hh_sigma[cpu_thread_id] = hh_sigma[cpu_thread_id] + h_sigma[i];
        }

        // Wait till OpenMP threads are finish!
        #pragma omp barrier
    }

    // Calculate the final result
    mean = 0.0;
    sigma = 0.0;

    for(int i = 0; i < NGPU; i = i+1){
    	mean = mean + (double)hh_mean[i];
    	sigma = sigma + (double)hh_sigma[i];
    }

    mean = mean / (double) N;
    sigma = sqrt(((1.0 / (double) N) * sigma + pow(mean, 2)) / (double) N);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime_tot, start, stop);

    // Write result to file
	output = fopen("integration_result_SS.txt", "a");
	fprintf(output, "%d %d %d %.5e %.5e ", N, threadsPerBlock, blocksPerGrid, mean, sigma);
	fclose(output);

    // Print the result from GPU
    printf("------GPU result------\n");
    printf("Mean  = %lf\n", mean);
    printf("Sigma = %lf\n", sigma);
    printf("GPU total time used = %.3lf (ms)\n\n", gputime_tot);

	/*
	Monte Carlo integration with CPU
	 */
	/*-----Simple Sampling-----*/
	mean = 0.0;
	sigma = 0.0;

	cudaEventRecord(start, 0);

	for(int i = 0; i < N; i = i+1){
		// Get random x coordinates
		for(int j = 0; j < 10; j = j+1){
			x[j] = (double) rand() / (double) RAND_MAX;
		}
		mean = mean + f(x);
		sigma = sigma + pow(f(x), 2);
	}
	mean = mean / (double) N;
	sigma = sqrt(((1.0 / (double) N) * sigma + pow(mean, 2)) / (double) N);

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&cputime, start, stop);

	// Write result to file
	output = fopen("integration_result_SS.txt", "a");
	fprintf(output, "%.5e +- %.5e %.3f\n", mean, sigma, cputime / gputime_tot);
	fclose(output);

    // Print the result from CPU
    printf("------CPU result------\n");
    printf("Mean  = %lf\n", mean);
    printf("Sigma = %lf\n", sigma);
    printf("GPU total time used = %.3lf (ms)\n\n", cputime);


	// All done , reset and free source
	free(h_mean);
    free(h_sigma);
    free(hh_mean);
    free(hh_sigma);

    // Destroy timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Reset GPU
    for(int i = 0; i < NGPU; i = i+1){
      	cudaSetDevice(Dev[i]);
      	cudaDeviceReset();
    }

	return 0;
}



/*
Host Code
 */
// function to be integrated
double f(double *x){
	double result = 1.0;
	for(int i = 0; i <= 9; i = i+1){
		result = result + pow(x[i], 2);
	}
	result = 1.0 / result;
	return result;
}

/*
Device Code
 */

__global__ void SimpleSampling(int N, float* mean, float* sigma, unsigned int seed)
{
	/*
	The mean and sigma here just sum them up with x and x^2
	Divided them and further calculate the real mean and sigma at host.
	 */

	extern __shared__ float cache[];
	float *meanCache = &cache[0];	
	float *sigmaCache = &cache[blockDim.x];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float f;
    float temp_mean = 0.0;
    float temp_sigma = 0.0;
    float temp_rand;

    // initialize cuRAND    
    curandState_t state;
    seed = seed + i;
    curand_init(seed, i, 0, &state);

    while (i < N){
		
		// Get random x[i] coordinate , and calculate f directly
    	f = 0.0;
    	for(int j = 0; j < 10; j = j+1){
            temp_rand = curand_uniform(&state);
    		f = f + powf(temp_rand, 2);
    	}
    	f = f + 1.0;
    	f = 1.0 / f;

    	// Simple Sampling
    	temp_mean  = temp_mean + f;
    	temp_sigma = temp_sigma + powf(f, 2);

    	// Go to next round
        i = i + blockDim.x * gridDim.x;
    }

    meanCache[cacheIndex] = temp_mean;
    sigmaCache[cacheIndex] = temp_sigma;
	
	__syncthreads();
    
    // perform parallel reduction, threadsPerBlock must be 2^m
    int ib = blockDim.x / 2;
    while (ib != 0) {
      	if(cacheIndex < ib){
      		meanCache[cacheIndex]  += meanCache[cacheIndex + ib];
      		sigmaCache[cacheIndex] += sigmaCache[cacheIndex + ib];
      	}
      	
      	__syncthreads();
	
      	ib /=2;
    }

    if(cacheIndex == 0){
    	mean[blockIdx.x]  = meanCache[0];
    	sigma[blockIdx.x] = sigmaCache[0];
    }
    
}