#include<unistd.h>
#include<stdio.h>
#include<curand.h>
#include<curand_kernel.h>

__global__ void random(unsigned int seed, float* result, int N)
{
	curandState_t state;
	curand_init(seed, 0, 0, &state);

	*result = curand_uniform(&state);
	float ingpu_result;
	for(int i = 0; i < N; i = i+1){
		ingpu_result = curand_uniform(&state);
		printf("%lf\n", ingpu_result);	// the output series of this is not the same
		//printf("%lf\n", powf(ingpu_result, 2) );
	}
}

int main(){
	int gid;
	scanf("%d", &gid);
	cudaSetDevice(gid);
	
	float x;
	float* gpu_x;
	int N = 10;
	cudaMalloc((void**) &gpu_x, sizeof(float));
	
//	for (int i = 0; i < N; i = i+1){
//		random<<<1,1>>>(time(NULL), gpu_x, N);
//		cudaMemcpy(&x, gpu_x, sizeof(float), cudaMemcpyDeviceToHost);
	
//		printf("%lf\n", x);	// the output of this is the same

//	}

	random<<<1,1>>>(time(NULL), gpu_x, N);



	cudaFree(gpu_x);

	cudaDeviceReset();
	return 0;
}
