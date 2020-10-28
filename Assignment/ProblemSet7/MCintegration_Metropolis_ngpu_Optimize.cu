#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<unistd.h>


float* h_mean;      // Store vector from d_mean
float* h_sigma;     // Store vector from d_sigma
float* hh_mean;     // to store the sum up h_mean per thread
float* hh_sigma;    // to store the sum up h_sigma per thread

// Host code
double f(double*);
double W(double*);

// weight function parameter
double a = 0.25;

// Device code
__global__ void Metropolis(int N, float a, float* mean, float* sigma, unsigned int seed);

int main(void){
    // Setting up random number generator
    srand(101);

    // Settings
    // Sampling N points
    // N = pow(2, n);
    // n = 1, 2 , ..., 16
    // array x --> holds the random number for coordinate. 
    // Store temperary random number in r, y.
    double mean, sigma;
    int N;
    double x[10], x_old[10];
    double r;

    FILE *output;
    float gputime_tot, cputime;

    // Settings for omp
    int cpu_thread_id = 0;

    // Settings for GPU
    int NGPU;   // numbers of GPU
    int *Dev;   // Store the GPU id
    int threadsPerBlock, blocksPerGrid;
    int sb;     // Size of output array from individual GPU
    int sm;     // Size of the shared memory per block
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

    // Output to a file
    output = fopen("integration_result_MA.txt", "a");
    fprintf(output, "N BlockSize GridSize MAGPU MAGPUerror MACPU MACPUerror SpeedUp\n");
    fclose(output);

    /*
    Start the loop to optimize the GPU
     */
    
    for(int m = 1; m <= 10; m = m+1){
        for(int n = 1; n <= 5; n = n+1){

            threadsPerBlock = pow(2, m);
            blocksPerGrid = pow(10, n);

            printf("threads per block = %d\n", threadsPerBlock);
            printf("blocks per grid = %d\n", blocksPerGrid);

            // Since in GPU, I calculate blockDim.x * GridDim.x of x first ,
            // then go to next round , so N > blockDim.x * GridDim.x 
            // The most ideal number of threads per grid for the integral to be more accurate,
            // is to make it run more turns.
            if( (N / NGPU) < blocksPerGrid * threadsPerBlock){
                output = fopen("integration_result_MA.txt", "a");
                fprintf(output, "%d %d %d %s %s %s %s %s\n", N, threadsPerBlock, blocksPerGrid, "nan", "nan", "nan", "nan", "nan");
                fclose(output);
                continue;
            }
        
            /*
            Monte Carlo integration with GPU
             */
            
            sb = blocksPerGrid * sizeof(float);     
            sm = threadsPerBlock * sizeof(float);
            h_mean   = (float*)malloc(sb * NGPU);
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
                Metropolis<<<blocksPerGrid, threadsPerBlock, 2 * sm>>>(N / NGPU, a, d_mean, d_sigma, time(NULL) + cpu_thread_id * 10.0);
        
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
            output = fopen("integration_result_MA.txt", "a");
            fprintf(output, "%d %d %d %.5e %.5e ", N, threadsPerBlock, blocksPerGrid, mean, sigma);
            fclose(output);

            // All done , reset and free source
            free(h_mean);
            free(h_sigma);
            free(hh_mean);
            free(hh_sigma);
        
            /*
            Monte Carlo integration with CPU
             */
            /*-----Metropolis Algorithm-----*/
            mean = 0.0;
            sigma = 0.0;
        
            cudaEventRecord(start, 0);  
        
            // Get initial x --> x_old (N = 1) 
            for(int j = 0; j < 10; j = j+1){
                x_old[j] = (double) rand() / (double) RAND_MAX;
            }
            mean = mean + f(x_old) / W(x_old);
            sigma = sigma + pow(f(x_old) / W(x_old), 2);
            
            // Get the other (N-1) sample points
            for(int i = 2; i <= N; i = i+1){
                // Get new x --> x
                for(int j = 0; j < 10; j = j+1){
                    x[j] = (double) rand() / (double) RAND_MAX;
                }
                
                // Check acceptance
                if(W(x) >= W(x_old)){ 
                    // Accept x, and to avoid overflow
                    memcpy(x_old, x, sizeof(x_old));
                }
                else{
                    r = (double) rand() / (double) RAND_MAX;
                    if(r < (W(x) / W(x_old))){
                        // Accept x, and to avoid overflow
                        memcpy(x_old, x, sizeof(x_old));
                    }
                }
                mean = mean + f(x_old) / W(x_old);
                sigma = sigma + pow(f(x_old) / W(x_old), 2);
            }
            mean = mean / (double) N;
            sigma = sqrt(((1.0 / (double) N) * sigma + pow(mean, 2)) / (double) N);
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
        
            cudaEventElapsedTime(&cputime, start, stop);    
        
            // Write result to file
            output = fopen("integration_result_MA.txt", "a");
            fprintf(output, "%.5e +- %.5e %.3f\n", mean, sigma, cputime / gputime_tot);
            fclose(output);
        
        }
    }

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

// function to be integrated
double f(double *x){
    double result = 1.0;
    for(int i = 0; i <= 9; i = i+1){
        result = result + pow(x[i], 2);
    }
    result = 1.0 / result;
    return result;
}

// weight function
double W(double *x){
    double weight = 1.0;
    double c;
    
    // Find c, so that integral c*exp(-ax) between [0,1] = 1
    c = a / (1.0 - exp(-a));

    // Calculate the weight function
    for(int i = 0; i <= 9; i = i+1){
        weight = weight * c * exp(-a * x[i]);
    }
    return weight;
}

__global__ void Metropolis(int N, float a, float* mean, float* sigma, unsigned int seed)
{
    /*
    The mean and sigma here just sum them up with x and x^2
    Divided them and further calculate the real mean and sigma at host.
    
    For Metropolis x series, each threads generate their own series.
     */

    extern __shared__ float cache[];
    float *meanCache = &cache[0];   
    float *sigmaCache = &cache[blockDim.x];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float f_old, W_old;
    float f_new, W_new;
    float c = a / (1.0 - expf(-a));
    float temp_mean  = 0.0;
    float temp_sigma = 0.0;
    float temp_rand;

    // initialize cuRAND    
    curandState_t state;
    seed = seed + i;
    curand_init(seed, i, 0, &state);

    // For each threads, generate the first {x}.
    // Get initial x-->x_old (N = 1), and calculate f and W directly.
    f_old = 0.0;
    W_old = 1.0;
    for(int j = 0; j < 10; j = j+1){
        temp_rand = curand_uniform(&state);
        f_old = f_old + powf(temp_rand, 2);
        W_old = W_old * c * expf(-a * temp_rand);
    }
    f_old = f_old + 1.0;
    f_old = 1.0 / f_old;

    temp_mean  = temp_mean + f_old / W_old;
    temp_sigma = temp_sigma + powf(f_old / W_old, 2);

    // Get the other (N-1) sample points
    while (i < (N - blockDim.x * gridDim.x)){
        
        // Get new x --> x , and calculate f and W directly
        // so that I don't have to store addition x[10] array.
        f_new = 0.0;
        W_new = 1.0;
        for(int j = 0; j < 10; j = j+1){
            temp_rand = curand_uniform(&state);
            f_new = f_new + powf(temp_rand, 2);
            W_new = W_new * c * expf(-a * temp_rand);
        }
        f_new = f_new + 1.0;
        f_new = 1.0 / f_new;

        // Check acceptance
        if(W_new > W_old){
            // Accept x, and record it.
            f_old = f_new;
            W_old = W_new;
        }
        else{
            temp_rand = curand_uniform(&state);
            if(temp_rand < (W_new / W_old)){
                // Accept x, and record it.
                f_old = f_new;
                W_old = W_new;
            }
        }

        temp_mean  = temp_mean + f_old / W_old;
        temp_sigma = temp_sigma + powf(f_old / W_old, 2);
        
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