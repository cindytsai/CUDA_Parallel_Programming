#include<unistd.h>
#include<stdio.h>
#include<curand.h>
#include<curand_kernel.h>

__global__ void random(unsigned int seed, float* result, int N)
{
	extern __shared__ float cache[];
	float *a = &cache[0];
	float *b = &cache[N];

	int id = threadIdx.x + blockIdx.x * blockDim.x;

//	curandState_t state;
//	curand_init(seed, id, 0, &state);
//	seed = curand(&state);

//	*result = curand_uniform(&state);
//	float ingpu_result;
//	for(int i = 0; i < N; i = i+1){
//		ingpu_result = curand_uniform(&state);
//		//printf("%lf\n", ingpu_result);	// the output series of this is not the same
//		//printf("%lf\n", powf(ingpu_result, 2) );
//	}
	curandState_t state;
	seed = seed + id;
	printf("seed:%d", seed);
	curand_init(seed, id, 0, &state);

//	for(int i = 0; i < N; i = i+1){
		

//		a[i] = curand_uniform(&state);
//		b[i] = powf(curand_uniform(&state),2);
//	}
	float temp_rand;
	for(int i = 0; i < N; i = i+1){
		temp_rand=curand_uniform(&state);
		printf("i=%d threadId=%d blockId=%d 	%lf	\n", i, threadIdx.x, blockIdx.x, temp_rand);
		//printf("i=%d threadId=%d blockId=%d 	%lf	%lf\n", i, threadIdx.x, blockIdx.x, cache[i], cache[i+N]);
	}
}

int main(){
	int gid;
	scanf("%d", &gid);
	cudaSetDevice(gid);
	
	float* gpu_x;
	int N = 5;
	cudaMalloc((void**) &gpu_x, sizeof(float));
	
//	for (int i = 0; i < N; i = i+1){
//		random<<<1,1>>>(time(NULL), gpu_x, N);
//		cudaMemcpy(&x, gpu_x, sizeof(float), cudaMemcpyDeviceToHost);
	
//		printf("%lf\n", x);	// the output of this is the same

//	}

	random<<<2,10, sizeof(float)*N*2>>>(time(NULL), gpu_x, N);

	cudaFree(gpu_x);

	cudaDeviceReset();
	return 0;
}
