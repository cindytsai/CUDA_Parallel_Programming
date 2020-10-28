/*This version synchronize the N_GPU calculation. */
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <omp.h>
# include <cuda_runtime.h>
# include <curand_kernel.h>
# include <gsl/gsl_sf_ellint.h>
# include <gsl/gsl_rng.h>
# define _USE_MATH_DEFINES

__constant__ double exp_update0_dev[5];
__constant__ double exp_update1_dev[2];
__device__ int UPDATE_SPIN_GPU (int, int, int, double);

void EXACT_EM (double*, double*);
void UPDATE_MATRIX(double*, double*);
void UPDATE_SPIN(int, int, int, int, int, int*, double*, gsl_rng*);
void ISING_CPU (void);

__global__ void SET_UP_KERNEL (long seed, curandState *state)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init (seed, index, 0, &state[index]);
}

__global__ void INITIALIZE_COLD (int *d_spin)
{
	int index_x = threadIdx.x + blockIdx.x*blockDim.x;
	int index_y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = index_x + index_y*blockDim.x*gridDim.x;

	d_spin[index] = 1;
}

__global__ void INITIALIZE_HOT (int *d_spin, curandState *state)
{
	int index_x = threadIdx.x + blockIdx.x*blockDim.x;
	int index_y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = index_x + index_y*blockDim.x*gridDim.x;
	curandState local = state[index];
	double x = curand_uniform(&local);

	if (x<0.5)
		d_spin[index] = 1;
	else
		d_spin[index] = -1;
//	printf("%d\t%.4f\n", index, x);
	state[index] = local;
}

__global__ void DOUBLE_CHECKERBOARD_SETUP(int *oTiB, int *eTiB)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Tx = blockDim.x;
	int thread_index = tx + ty*Tx;
	int cache_index = (tx+1) + (ty+1)*(Tx+2);
	int cache_index_r = (tx+2) + (ty+1)*(Tx+2);
	int cache_index_l = tx + (ty+1)*(Tx+2);
	int cache_index_u = (tx+1) + (ty+2)*(Tx+2);
	int cache_index_d = (tx+1) + ty*(Tx+2);
	// odd threads
	if (threadIdx.x%2!=threadIdx.y%2)
	{
		oTiB[5*(thread_index/2)] = cache_index;
		oTiB[5*(thread_index/2)+1] = cache_index_r;
		oTiB[5*(thread_index/2)+2] = cache_index_l;
		oTiB[5*(thread_index/2)+3] = cache_index_u;
		oTiB[5*(thread_index/2)+4] = cache_index_d;
	}
	//
	// even threads
	else if (threadIdx.x%2==threadIdx.y%2)
	{
		eTiB[5*(thread_index/2)] = cache_index;
		eTiB[5*(thread_index/2)+1] = cache_index_r;
		eTiB[5*(thread_index/2)+2] = cache_index_l;
		eTiB[5*(thread_index/2)+3] = cache_index_u;
		eTiB[5*(thread_index/2)+4] = cache_index_d;
//			printf("%d\t%d\t%d\t\n", block_index, cache_index_d, eTioB[block_index/2][5*(thread_index/2)+4]);
	}
	//
}

__global__ void ISING_ODD_BLOCK(int interval_measure, double B, int *d_spin, int *d_spin_l, int *d_spin_r, int *d_spin_d, int *d_spin_u, curandState *state, int *oTiB, int *eTiB)
{
	extern __shared__ int cache[];
	int lattice_center_odd, lattice_center_even;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Tx = blockDim.x;
	int Ty = blockDim.y;
	int thread_index = tx + ty*Tx;
	int block_index = blockIdx.x + gridDim.x*blockIdx.y;
	int Bx = 2*gridDim.x;
	int By = gridDim.y;
	int Nx = 2*Tx*Bx;
	int Ny = Ty*By;
	int bx = (2*block_index)%Bx;
	int by = (2*block_index)/Bx;
	//for odd block
	int parity = (bx+by+1)%2;
	bx += parity;
	// mount odd site in each odd block to shared
	int x = (2*thread_index)%(2*Tx);
	int y = (2*thread_index)/(2*Tx);
	parity = (x+y+1)%2;
	x += parity;
	int cache_index = (x+1) + (y+1)*(2*Tx+2);
	x += 2*Tx*bx;
	y += Ty*by;
	int lattice_index = x + y*Nx;
	lattice_center_odd = lattice_index;  // to copy the updated spin from shared to d_spin
	cache[cache_index] = d_spin[lattice_index];
//	if (block_index==0)
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
	//
	// mount even site in each block to shared
	x = (2*thread_index)%(2*Tx);
	y = (2*thread_index)/(2*Tx);
	parity = (x+y)%2;
	x += parity;
	cache_index = (x+1) + (y+1)*(2*Tx+2);
	x += 2*Tx*bx;
	y += Ty*by;
	lattice_index = x + y*Nx;
	lattice_center_even = lattice_index;  // to copy the updated spin from shared to d_spin
	cache[cache_index] = d_spin[lattice_index];
//	if (block_index==2047)
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
	//
	// mount the boundary to shared
	if (Tx==1)
	{
		// left
		x = 0;
		y = ty; 
		cache_index = (y+1)*(2*Tx+2);
		x += 2*Tx*bx;
		y += Ty*by;
		lattice_index = (x-1+Nx)%Nx + y*Nx;
		if (x==0)
			cache[cache_index] = d_spin_l[lattice_index];
		else
			cache[cache_index] = d_spin[lattice_index];
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		// right
		x = 1;
		y = ty;
		cache_index = (x+2) + (y+1)*(2*Tx+2);
		x += 2*Tx*bx;
		y += Ty*by;
		lattice_index = (x+1)%Nx + y*Nx;
		if (x==Nx-1)
			cache[cache_index] = d_spin_r[lattice_index];
		else
			cache[cache_index] = d_spin[lattice_index];
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		// down
		if (ty==0)
		{
			for (int i=0; i<2; i++)
			{
				x = i;
				y = 0;
				cache_index = x+1;
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y-1+Ny)%Ny*Nx;
				if (y==0)
					cache[cache_index] = d_spin_d[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
		// up
		else if (ty==Ty-1)
		{
			for (int i=0; i<2; i++)
			{
				x = i;
				y = Ty-1;
				cache_index = (x+1) + (y+2)*(2*Tx+2);
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y+1)%Ny*Nx;
				if (y==Ny-1)
					cache[cache_index] = d_spin_u[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
	}
	else
	{
		// left
		if (tx==0)
		{
			x = 0;
			y = ty;
			cache_index = (y+1)*(2*Tx+2);
			x += 2*Tx*bx;
			y += Ty*by;
			lattice_index = (x-1+Nx)%Nx + y*Nx;
			if (x==0)
				cache[cache_index] = d_spin_l[lattice_index];
			else
				cache[cache_index] = d_spin[lattice_index];
//			printf("%d\t%d\n", bx*Bx, by*By);
//			printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		}
		// right
		else if (tx==Tx-1)
		{
			x = 2*Tx-1;
			y = ty;
			cache_index = (x+2) + (y+1)*(2*Tx+2);
			x += 2*Tx*bx;
			y += Ty*by;
			lattice_index = (x+1)%Nx + y*Nx;
			if (x==Nx-1)
				cache[cache_index] = d_spin_r[lattice_index];
			else
				cache[cache_index] = d_spin[lattice_index];
//			printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		}
		// down
		if (ty==0)
		{
			for (int i=0; i<2; i++)
			{
				x = 2*tx+i;
				y = 0;
				cache_index = x+1;
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y-1+Ny)%Ny*Nx;
				if (y==0)
					cache[cache_index] = d_spin_d[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
		// up
		else if (ty==Ty-1)
		{
			for (int i=0; i<2; i++)
			{
				x = 2*tx+i;
				y = Ty-1;
				cache_index = (x+1) + (y+2)*(2*Tx+2);
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y+1)%Ny*Nx;
				if (y==Ny-1)
					cache[cache_index] = d_spin_u[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
	}
	__syncthreads();
	//
	curandState local = state[threadIdx.x+blockDim.x*blockIdx.x + blockDim.x*gridDim.x*(threadIdx.y+blockDim.y*blockIdx.y)];
	int center, right, left, up, down;
	int spin_r, spin_l, spin_u, spin_d;
	int old_spin, new_spin, spin_around;
	int delta_E;
	// spin update
		// odd threads
	center = oTiB[5*thread_index];
	right = oTiB[5*thread_index+1];
	left = oTiB[5*thread_index+2];
	up = oTiB[5*thread_index+3];
	down = oTiB[5*thread_index+4];

	for (int iter=0; iter<interval_measure; iter++)
	{
		spin_r = cache[right];
		spin_l = cache[left];
		spin_u = cache[up];
		spin_d = cache[down];
		old_spin = cache[center];
		new_spin = -old_spin;
		spin_around = spin_r + spin_l + spin_u  + spin_d;
		if (spin_around%2!=0)
			printf("Odd thread Odd block: %d\t%d\t%d\t%d\t%d\n", center, spin_r, spin_l, spin_u, spin_d);
		delta_E = (old_spin-new_spin)*(spin_around+B);
		if (delta_E<=0)
			cache[center] = new_spin;
		else
		{
			double x = curand_uniform(&local);
			cache[center] = UPDATE_SPIN_GPU(old_spin, new_spin, spin_around, x);
//			if (thread_index==0)
//				printf("%.4f\n", x);
		}
	}
	d_spin[lattice_center_odd] = cache[center];
	__syncthreads();
		//
		// even threads;
	center = eTiB[5*thread_index];
	right = eTiB[5*thread_index+1];
	left = eTiB[5*thread_index+2];
	up = eTiB[5*thread_index+3];
	down = eTiB[5*thread_index+4];

	for (int iter=0; iter<interval_measure; iter++)
	{
		spin_r = cache[right];
		spin_l = cache[left];
		spin_u = cache[up];
		spin_d = cache[down];
		old_spin = cache[center];
		new_spin = -old_spin;
		spin_around = spin_r + spin_l + spin_u  + spin_d;
		if (spin_around%2!=0)
			printf("Even thread Odd block: %d\t%d\t%d\t%d\t%d\n", center, spin_r, spin_l, spin_u, spin_d);
		delta_E = (old_spin-new_spin)*(spin_around+B);
		if (delta_E<=0)
			cache[center] = new_spin;
		else
		{
			double x = curand_uniform(&local);
			cache[center] = UPDATE_SPIN_GPU(old_spin, new_spin, spin_around, x);
		}
	}
	d_spin[lattice_center_even] = cache[center];
	state[threadIdx.x+blockDim.x*blockIdx.x + blockDim.x*gridDim.x*(threadIdx.y+blockDim.y*blockIdx.y)]=local;
	__syncthreads();
		//
}

__global__ void ISING_EVEN_BLOCK(int interval_measure, double B, int *d_spin, int *d_spin_l, int *d_spin_r, int *d_spin_d, int *d_spin_u, curandState *state, int *oTiB, int *eTiB)
{
	extern __shared__ int cache[];
	int lattice_center_odd, lattice_center_even;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Tx = blockDim.x;
	int Ty = blockDim.y;
	int thread_index = tx + ty*Tx;
	int block_index = blockIdx.x + gridDim.x*blockIdx.y;
	int Bx = 2*gridDim.x;
	int By = gridDim.y;
	int Nx = 2*Tx*Bx;
	int Ny = Ty*By;
	int bx = (2*block_index)%Bx;
	int by = (2*block_index)/Bx;
	// for even block
	int parity = (bx+by)%2;
	bx += parity;
	// mount odd site in each odd block to shared
	int x = (2*thread_index)%(2*Tx);
	int y = (2*thread_index)/(2*Tx);
	parity = (x+y+1)%2;
	x += parity;
	int cache_index = (x+1) + (y+1)*(2*Tx+2);
	x += 2*Tx*bx;
	y += Ty*by;
	int lattice_index = x + y*Nx;
	lattice_center_odd = lattice_index; // to copy the updated spin from shared to d_spin
	cache[cache_index] = d_spin[lattice_index];
//	if (block_index==3)
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
	//
	// mount even site in each block to shared
	x = (2*thread_index)%(2*Tx);
	y = (2*thread_index)/(2*Tx);
	parity = (x+y)%2;
	x += parity;
	cache_index = (x+1) + (y+1)*(2*Tx+2);
	x += 2*Tx*bx;
	y += Ty*by;
	lattice_index = x + y*Nx;
	lattice_center_even = lattice_index; // to copy the updated spin from shared to d_spin
	cache[cache_index] = d_spin[lattice_index];
//	if (block_index==2)
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
	//
	// mount the boundary to shared
	if (Tx==1)
	{
		// left
		x = 0;
		y = ty; 
		cache_index = (y+1)*(2*Tx+2);
		x += 2*Tx*bx;
		y += Ty*by;
		lattice_index = (x-1+Nx)%Nx + y*Nx;
		if (x==0)
			cache[cache_index] = d_spin_l[lattice_index];
		else
			cache[cache_index] = d_spin[lattice_index];
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		// right
		x = 1;
		y = ty;
		cache_index = (x+2) + (y+1)*(2*Tx+2);
		x += 2*Tx*bx;
		y += Ty*by;
		lattice_index = (x+1)%Nx + y*Nx;
		if (x==Nx-1)
			cache[cache_index] = d_spin_r[lattice_index];
		else
			cache[cache_index] = d_spin[lattice_index];
//		printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		// down
		if (ty==0)
		{
			for (int i=0; i<2; i++)
			{
				x = i;
				y = 0;
				cache_index = x+1;
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y-1+Ny)%Ny*Nx;
				if (y==0)
					cache[cache_index] = d_spin_d[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
		// up
		else if (ty==Ty-1)
		{
			for (int i=0; i<2; i++)
			{
				x = i;
				y = Ty-1;
				cache_index = (x+1) + (y+2)*(2*Tx+2);
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y+1)%Ny*Nx;
				if (y==Ny-1)
					cache[cache_index] = d_spin_u[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
	}
	else
	{
		// left
		if (tx==0)
		{
			x = 0;
			y = ty;
			cache_index = (y+1)*(2*Tx+2);
			x += 2*Tx*bx;
			y += Ty*by;
			lattice_index = (x-1+Nx)%Nx + y*Nx;
			if (x==0)
				cache[cache_index] = d_spin_l[lattice_index];
			else
				cache[cache_index] = d_spin[lattice_index];
//			printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		}
		// right
		else if (tx==Tx-1)
		{
			x = 2*Tx-1;
			y = ty;
			cache_index = (x+2) + (y+1)*(2*Tx+2);
			x += 2*Tx*bx;
			y += Ty*by;
			lattice_index = (x+1)%Nx + y*Nx;
			if (x==Nx-1)
				cache[cache_index] = d_spin_r[lattice_index];
			else
				cache[cache_index] = d_spin[lattice_index];
//			printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
		}
		// down
		if (ty==0)
		{
			for (int i=0; i<2; i++)
			{
				x = 2*tx+i;
				y = 0;
				cache_index = x+1;
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y-1+Ny)%Ny*Nx;
				if (y==0)
					cache[cache_index] = d_spin_d[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
		// up
		else if (ty==Ty-1)
		{
			for (int i=0; i<2; i++)
			{
				x = 2*tx+i;
				y = Ty-1;
				cache_index = (x+1) + (y+2)*(2*Tx+2);
				x += 2*Tx*bx;
				y += Ty*by;
				lattice_index = x + (y+1)%Ny*Nx;
				if (y==Ny-1)
					cache[cache_index] = d_spin_u[lattice_index];
				else
					cache[cache_index] = d_spin[lattice_index];
//				printf("%d\t%d\t%d\t%d\n", lattice_index, cache_index, d_spin[lattice_index], cache[cache_index]);
			}
		}
	}
	__syncthreads();
	//
	curandState local = state[threadIdx.x+blockDim.x*blockIdx.x + blockDim.x*gridDim.x*(threadIdx.y+blockDim.y*blockIdx.y)];
	int center, right, left, up, down;
	int spin_r, spin_l, spin_u, spin_d;
	int old_spin, new_spin, spin_around;
	int delta_E;
	// spin update
		// odd threads
	center = oTiB[5*thread_index];
	right = oTiB[5*thread_index+1];
	left = oTiB[5*thread_index+2];
	up = oTiB[5*thread_index+3];
	down = oTiB[5*thread_index+4];

	for (int iter=0; iter<interval_measure; iter++)
	{
		spin_r = cache[right];
		spin_l = cache[left];
		spin_u = cache[up];
		spin_d = cache[down];
		old_spin = cache[center];
		new_spin = -old_spin;
		spin_around = spin_r + spin_l + spin_u  + spin_d;
		if (spin_around%2!=0)
			printf("Odd thread Even block: %d\t%d\t%d\t%d\t%d\n", center, spin_r, spin_l, spin_u, spin_d);
		delta_E = (old_spin-new_spin)*(spin_around+B);
		if (delta_E<=0)
			cache[center] = new_spin;
		else
		{
			double x = curand_uniform(&local);
			cache[center] = UPDATE_SPIN_GPU(old_spin, new_spin, spin_around, x);
		}
	}
	d_spin[lattice_center_odd] = cache[center];
	__syncthreads();
		//
		// even threads;
	center = eTiB[5*thread_index];
	right = eTiB[5*thread_index+1];
	left = eTiB[5*thread_index+2];
	up = eTiB[5*thread_index+3];
	down = eTiB[5*thread_index+4];

	for (int iter=0; iter<interval_measure; iter++)
	{
		spin_r = cache[right];
		spin_l = cache[left];
		spin_u = cache[up];
		spin_d = cache[down];
		old_spin = cache[center];
		new_spin = -old_spin;
		spin_around = spin_r + spin_l + spin_u  + spin_d;
		if (spin_around%2!=0)
		{
			printf("Even threads Even block: %d\t%d\t%d\t%d\t%d\t%d\n", block_index, center, spin_r, spin_l, spin_u, spin_d);
//			printf("%d\n", eTieB[block_index][5*thread_index+4]);
		}
		delta_E = (old_spin-new_spin)*(spin_around+B);
		if (delta_E<=0)
			cache[center] = new_spin;
		else
		{
			double x = curand_uniform(&local);
			cache[center] = UPDATE_SPIN_GPU(old_spin, new_spin, spin_around, x);
		}
	}
	d_spin[lattice_center_even] = cache[center];
	state[threadIdx.x+blockDim.x*blockIdx.x + blockDim.x*gridDim.x*(threadIdx.y+blockDim.y*blockIdx.y)]=local;
	__syncthreads();
		//
}

__global__ void  ISING_MEASUREMENT (double B, int *d_spin, int *d_spin_r, int *d_spin_u, int *d_M, int *d_E)
{
	extern __shared__ int cache[];
	int layer = blockDim.x*blockDim.y/2;
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int Nx = blockDim.x*gridDim.x;
	int Ny = blockDim.y*gridDim.y;
	int index = x + Nx*y;
	int index_r = (x+1)%Nx + Nx*y;
	int index_u = x + Nx*((y+1)%Ny);
	int cache_index = threadIdx.x + blockDim.x*threadIdx.y;
	int block_index = blockIdx.x + gridDim.x*blockIdx.y;
	int shift = blockDim.x*blockDim.y;
	int spin_r, spin_u;
	if (x==Nx-1)
		spin_r = d_spin_r[index_r];
	else
		spin_r = d_spin[index_r];
	if (y==Ny-1)
		spin_u = d_spin_u[index_u];
	else
		spin_u = d_spin[index_u];
	cache[cache_index] = d_spin[index];
	cache[cache_index+shift] = -d_spin[index]*(spin_r+spin_u+B);
	__syncthreads();

	while (layer>0)
	{
		if (cache_index<layer)
		{
			cache[cache_index] += cache[cache_index+layer];
			cache[cache_index+shift] += cache[cache_index+shift+layer];
		}
		layer /= 2;
		__syncthreads();
	}	
	if (cache_index==0)
	{
		d_M[block_index] = cache[0];
		d_E[block_index] = cache[shift];
	}
}

__global__ void CONSTANT_MEMORY_TEST ()
{
//	printf("Do constant memory test.\n");
	for (int i=0; i<5; i++)
		printf("%.4f\t", exp_update0_dev[i]);
	printf("\n");
	for (int i=0; i<2; i++)
		printf("%.4f\t", exp_update1_dev[i]);
	printf("\n");
}

__global__ void RANDOM_NUMBER_TEST (int iter, curandState *state)
{
	int index_x = threadIdx.x + blockIdx.x*blockDim.x;
	int index_y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = index_x + index_y*blockDim.x*gridDim.x;
	curandState local = state[index];
	
	int N = 1;
	for (int i=0; i<N; i++)
		printf("%d\t%.4f\n", iter, curand_uniform(&local));
	state[threadIdx.x+blockDim.x*blockIdx.x] = local;
}

char ini;
int N_thermal, N_measure, interval_measure, interval_save, interval_display;
int Nx, Ny, N_site;
double T, B;
double Tc, E_exact, M_exact;
int *h_spin;
long seed;
gsl_rng *rng;

int main(void)
{
	char mode;
	int size_site, size_site_GPU, m, n;
	int N_GPU_x, N_GPU_y, N_GPU, tx, ty, bx, by, N_block, N_thread, cpu_thread_id;
	float gpu_time, total_time, cpu_time;
	int *gid;
	long *seed_GPU;
	cudaEvent_t start, stop;
	
	puts("Simulate the 2D Ising model by N GPU.\n");
	puts("Set the number of GPU in x any direction ( N_GPU_x,N_GPU_y ).");
	scanf("%d %d", &N_GPU_x, &N_GPU_y);
	printf("The number of GPU in x and y direction is is ( %d,%d ) .\n", N_GPU_x, N_GPU_y);
	N_GPU = N_GPU_x*N_GPU_y;
	puts("Set the lattice size ( Nx,Ny ) for the 2D lattice (Nx/N_GPU_x and Ny/N_GPU_y must be even to apply checkboard scheme with periodic boundary condition).");
	scanf("%d %d", &Nx, &Ny);
	if ((Nx/N_GPU_x)%2!=0 || (Ny/N_GPU_y)%2!=0)
	{
		puts("Nx/N_GPU_x and Ny/N_GPU_y must be even! Exit!");
		exit(1);
	}
	N_site = Nx*Ny;
	printf("The lattice size is ( %d,%d ) .\n", Nx, Ny);

	size_site = N_site*sizeof(int);
	h_spin = (int*)malloc(size_site);

	puts("Set the temperature for the simulation in unit of J.");
	scanf("%lf", &T);
	if (T<0)
	{
		puts("Temperature must be positive! Exit!");
		exit(1);
	}
	printf("The temperature is %.6f J .\n", T);
	puts("Set the magnetic field for the simulation in unit of J.");
	scanf("%lf", &B);
	if (B<0)
	{
		puts("Field strength must be positive! Exit!");
		exit(1);
	}
	printf("The magnetic field is %.6f J.\n", B);
	puts("Choose cold/hot start (c/h) .");
	scanf("%c", &ini);  // absorb the \n
	scanf("%c", &ini);
	if ( ini=='c' )
		puts("Cold start is chosen.");
	else if ( ini=='h' )
		puts("Hot start is chosen");
	else
	{
		puts("Must enter c or h! Exit!");
		exit(1);
	}
	printf("\n");
	puts("Enter the number of measurement(s)");
	scanf("%d", &N_measure);
	printf("The number of measurement(s) is %d .\n", N_measure);
	puts("Enter the interval for measurement.");
	scanf("%d", &interval_measure);
	printf("The interval for measurement is %d .\n", interval_measure);
	puts("Enter the number of iterations for thermalization (must be divisibe by interval_measure).");
	scanf("%d", &N_thermal);
	if ( N_thermal%interval_measure!=0 )
	{
		puts("N_thermal is indivisible by interval_measure. Exit!");
		exit(1);
	}
	printf("The number of iterations for thermalization is %d .\n", N_thermal);
	puts("Enter the interval for display (must be divisible by interval_measure).");
	scanf("%d", &interval_display);
	if ( interval_display%interval_measure!=0 )
	{
		puts("interval_display is indivisible by interval_measure. Exit!");
		exit(1);
	}
	printf("The interval for display is %d .\n", interval_display);
	puts("Enter the interval for saving (must be divisible by interval_measure).");
	scanf("%d", &interval_save);
	if ( interval_save%interval_measure!=0 )
	{
		puts("N_thermal is indivisible by interval_measure. Exit!");
		exit(1);
	}
	printf("The interval for saving is %d .\n", interval_save);

	puts("Select the mode for simulation GPU/CPU/Both (g/c/b) .");
	scanf("%c", &mode);  // absorb the \n
	scanf("%c", &mode);
	if ( mode=='g' )
		puts("Do GPU simulation.");
	else if ( mode=='c' )
		puts("Do CPU simulation.");
	else if ( mode=='b' )
		puts("Do both GPU and CPU simulation.");
	else
	{
		puts("Must enter g, c or b! Exit!");
		exit(1);
	}
	if (mode=='c')
	{
		puts("Set the CPU seed for random number generator.");
		scanf("%ld", &seed);
		printf("The CPU seed is set to be %ld .\n", seed);
	}
	else
	{
		printf("\n");
		seed_GPU = (long *)malloc(N_GPU*sizeof(long));
		gid = (int*)malloc(N_GPU*sizeof(int));

		for (int i=0; i<N_GPU; i++)
		{
			printf("Set the seed for #%d GPU.\n", i+1);
			scanf("%ld", &seed_GPU[i]);
			printf("The seed for #%d GPU is %d .\n", i+1, seed_GPU[i]);
		}

		for (int i=0; i<N_GPU; i++)
		{
			printf("Set the GPU ID for #%d GPU.", i+1);
			scanf("%d", &gid[i]);
			printf("The GPU ID for #%d GPU is %d .\n", i+1, gid[i]);
		}

		puts("Set the value m, n such that the theads per block is ( tx,ty ) = (2^m,2^n) (Nx/N_GPU must be divisible by tx and Ny/N_GPU must be divisible by ty).");
		scanf("%d %d", &m, &n);
		if ( m<1 )
		{
			puts("m must be positive! Exit!");
			exit(1);
		}
		else if ( n<1 )
		{
			puts("n must be positive! Exit!");
			exit(1);
		}
		tx = (int)pow(2.0, m);
		ty = (int)pow(2.0, n);
		if ( (Nx/N_GPU_x)%tx!=0 )
		{
			puts("Nx/N_GPU_x must be divisible by threads per block tx! Exit!");
			printf("%d\t%d\n", tx, (Nx/N_GPU_x)%tx);
			exit(1);
		}
		if ( (Ny/N_GPU_y)%ty!=0 )
		{
			puts("Ny/N_GPU_y must be divisible by threads per block ty! Exit!");
			exit(1);
		}
		printf("Threads per block is ( %d,%d ) .\n", tx, ty);
		puts("Use only one grids to accomdate the lattice so blocks per grid is auto-matically set.");
		bx = Nx/N_GPU_x/tx;
		if (bx%2!=0)
		{
			puts("bx must be even to apply double checkboard scheme for periodic boundary condition!");
			exit(1);
		}
		by = Ny/N_GPU_y/ty;
		if (by%2!=0)
		{
			puts("by must be even to apply double checkboard scheme for periodic boundary condition!");
			exit(1);
		}
		printf("Blocks per grid is ( %d,%d ) .\n", bx, by);
		printf("\n");

		int size_block;
		int size_shared;
		int size_shared_measure;
		int to_save = interval_save/interval_measure;
		int to_display = interval_display/interval_measure;
		double M_mean, E_mean, M_sigma, E_sigma;
		int *M_GPU, *E_GPU;
		int *h_M;
		int *h_E;
		double *M_save;
		double *E_save;
		int **gid_P2P;
		int **d_M_GPU;
		int **d_E_GPU;
		int **d_spin_GPU;
		curandState **devstate_GPU;
		int **odd_T_in_B_GPU, **even_T_in_B_GPU;
		FILE *output = fopen("spin_config_gpu.txt", "w");
		FILE *output2 = fopen("M_and_E_gpu.txt", "w");
		dim3 TpB (tx, ty);
		dim3 BpG (bx, by);
		N_block = bx*by;
		N_thread = tx*ty;
		size_block = N_block*sizeof(int);
		size_shared = (tx+2)*(ty+2)*sizeof(int);
		size_shared_measure = 2*N_thread*sizeof(int); // to calculate both M and E;
		size_site_GPU = size_site/N_GPU;
		M_GPU = (int *)malloc(N_GPU*sizeof(int));
		E_GPU = (int *)malloc(N_GPU*sizeof(int));
		h_M = (int *)malloc(N_GPU*size_block);
		h_E = (int *)malloc(N_GPU*size_block);
		M_save = (double *)malloc(N_measure*sizeof(double));
		E_save = (double *)malloc(N_measure*sizeof(double));
		gid_P2P = (int **)malloc(N_GPU*sizeof(int *));
		d_M_GPU = (int **)malloc(N_GPU*sizeof(int *));
		d_E_GPU = (int **)malloc(N_GPU*sizeof(int *));
		d_spin_GPU = (int **)malloc(N_GPU*sizeof(int *));
		devstate_GPU = (curandState **)malloc(N_GPU*sizeof(curandState *));
		odd_T_in_B_GPU = (int **)malloc(N_GPU*sizeof(int *));
		even_T_in_B_GPU = (int **)malloc(N_GPU*sizeof(int *));

		omp_set_num_threads(N_GPU);
		#pragma omp parallel private(cpu_thread_id)
		// OpenMP starts
		{
			int gpu_id_x, gpu_id_y;
			cpu_thread_id = omp_get_thread_num();
			cudaSetDevice(gid[cpu_thread_id]);

			gid_P2P[cpu_thread_id] = (int *)malloc(5*sizeof(int));
			gpu_id_x = cpu_thread_id%N_GPU_x;
			gpu_id_y = cpu_thread_id/N_GPU_x;
			gid_P2P[cpu_thread_id][0] = (gpu_id_x-1+N_GPU_x)%N_GPU_x + gpu_id_y*N_GPU_x;  // gid at left
			gid_P2P[cpu_thread_id][1] = (gpu_id_x+1)%N_GPU_x + gpu_id_y*N_GPU_x;  // gid at right
			gid_P2P[cpu_thread_id][2] = gpu_id_x + (gpu_id_y-1+N_GPU_y)%N_GPU_y*N_GPU_x;  // gid at down
			gid_P2P[cpu_thread_id][3] = gpu_id_x + (gpu_id_y+1)%N_GPU_y*N_GPU_x;  // gid at up
			gid_P2P[cpu_thread_id][4] = gpu_id_x*(Nx/N_GPU_x) + Nx*gpu_id_y*(Ny/N_GPU_y);  // shift of sites for each gpu, necessary for spin copy

			#pragma omp master
			{
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				total_time = 0.0;
				cudaEventRecord(start, 0);
			}
			cudaDeviceEnablePeerAccess(gid[gid_P2P[cpu_thread_id][0]],0);
			cudaDeviceEnablePeerAccess(gid[gid_P2P[cpu_thread_id][1]],0);
			cudaDeviceEnablePeerAccess(gid[gid_P2P[cpu_thread_id][2]],0);
			cudaDeviceEnablePeerAccess(gid[gid_P2P[cpu_thread_id][3]],0);
			#pragma omp master
			{
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				total_time += gpu_time;
				printf("GPU time for enabling peer to peer is %.4f ms.\n", gpu_time);		
			}

			cudaMalloc((void **)&d_spin_GPU[cpu_thread_id], size_site_GPU);
			
			#pragma omp master
			{
				cudaEventRecord(start, 0);
			}
			// initialize the spin
			if (ini=='c')
				INITIALIZE_COLD<<<BpG, TpB>>> (d_spin_GPU[cpu_thread_id]);
			else
			{
				cudaMalloc((void**)&devstate_GPU[cpu_thread_id], N_site/N_GPU*sizeof(curandState));
				SET_UP_KERNEL<<<N_block, N_thread>>> ( seed_GPU[cpu_thread_id], devstate_GPU[cpu_thread_id]);
				INITIALIZE_HOT<<<BpG, TpB>>> (d_spin_GPU[cpu_thread_id], devstate_GPU[cpu_thread_id]);
			}
			// spin initialize check
//			for (int i=0; i<Ny/N_GPU_y; i++)
//				cudaMemcpy(h_spin+gid_P2P[cpu_thread_id][4]+i*Nx/N_GPU_x, d_spin_GPU[cpu_thread_id]+i*Nx/N_GPU_x, Nx/N_GPU_x*sizeof(int), cudaMemcpyDeviceToHost);
//			#pragma omp barrier
//			#pragma omp single
//			{
//				FILE *check = fopen("initial_spin_check.txt", "w");
//				for (int j=0; j<Ny; j++)
//				{
//					for (int i=0; i<Nx; i++)	
//						fprintf(check, "%d\t", h_spin[i+j*Nx]);
//					fprintf(check, "\n");
//				}
//				fprintf(check, "\n");
//			}
				//
	
			if (mode=='b')
			{
				for (int i=0; i<Ny/N_GPU_y; i++)
					cudaMemcpy(h_spin+gid_P2P[cpu_thread_id][4]+i*Nx/N_GPU_x, d_spin_GPU[cpu_thread_id]+i*Nx/N_GPU_x, Nx/N_GPU_x*sizeof(int), cudaMemcpyDeviceToHost);
			}
			//
		
			#pragma omp master
			{
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				total_time += gpu_time;
				printf("GPU time for initialize the spin lattice on device is %.4f ms.\n", gpu_time);		
			}
	
//			int block_o, block_e;
			int thread_o, thread_e;
//			block_o = N_block/2;
//			block_e = (N_block+1)/2;
//			block_e = block_o;
			thread_o = N_thread/2;
//			thread_e = (N_thread+1)/2;
			thread_e = thread_o;
//			int *pointer_of_threads;
			cudaMalloc((void**)&odd_T_in_B_GPU[cpu_thread_id], thread_o*5*sizeof(int));
			cudaMalloc((void**)&even_T_in_B_GPU[cpu_thread_id], thread_e*5*sizeof(int));

			#pragma omp master			
			{
				puts("Initialize the mapping from shared memory indices to thread indices.");
				cudaEventRecord(start, 0);
			}
			DOUBLE_CHECKERBOARD_SETUP <<<1, TpB>>> (odd_T_in_B_GPU[cpu_thread_id], even_T_in_B_GPU[cpu_thread_id]);
			#pragma omp master
			{
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				total_time += gpu_time;
				printf("GPU time for initializing the mapping from shared memory indices to thread indices is %.4f ms.\n", gpu_time);		
			}
			
			double **exp_update = (double **)malloc(2*sizeof(double *));
			exp_update[0] = (double *)malloc(5*sizeof(double));
			exp_update[1] = (double *)malloc(2*sizeof(double));
			#pragma omp master
			{
				puts("Calculate the Boltzmann factor and store it in the constant memroy.");
				cudaEventRecord(start, 0);
			}
			UPDATE_MATRIX (exp_update[0], exp_update[1]);
//			for (int i=0; i<5; i++)
//				printf("%.4f\n", exp_update[0][i]);
//			for (int i=0; i<2; i++)
//				printf("%.4f\n", exp_update[1][i]);
			cudaMemcpyToSymbol(exp_update0_dev, exp_update[0], 5*sizeof(double));
			cudaMemcpyToSymbol(exp_update1_dev, exp_update[1], 2*sizeof(double));
			#pragma omp master
			{
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&gpu_time, start, stop);
				total_time += gpu_time;
				printf("GPU time for calculating the Boltzmann factor and storing it in the constant memory is %.4f ms.\n", gpu_time);		

				puts("index of Meas.\t<M>\t\t<E>\n===============================================================");
			}
//			cudaFree(pointer_of_threads);
			free(exp_update);
			
			cudaMalloc((void **)&d_M_GPU[cpu_thread_id], size_block);
			cudaMalloc((void **)&d_E_GPU[cpu_thread_id], size_block);
//			cudaMemset(d_M_GPU[cpu_thread_id], 0, size_block);
			cudaMalloc((void**)&devstate_GPU[cpu_thread_id], N_site/N_GPU/2/2*sizeof(curandState));
			SET_UP_KERNEL<<<N_block/2, N_thread/2>>> (seed_GPU[cpu_thread_id], devstate_GPU[cpu_thread_id]);

			//constant memory test
//			cudaEvent_t start_test, stop_test;
//			cudaEventCreate(&start_test);
//			cudaEventCreate(&stop_test);
//			cudaEventRecord(start_test, 0);
//			CONSTANT_MEMORY_TEST<<<1,1>>> ();
//			cudaEventRecord(stop_test, 0);
//			cudaEventSynchronize(stop_test);
//			cudaEventElapsedTime(&gpu_time, start_test, stop_test);
//			printf("GPU time for constant memory test is %.4f ms.\n", gpu_time);		
		  	//
		  	//curand device API seed test
//			cudaMalloc((void**)&devstate_GPU[cpu_thread_id], sizeof(curandState));
//			SET_UP_KERNEL<<<N_block, N_thread>>> (seed, devstate_GPU[cpu_thread_id]);
//			for (int i=0; i<1; i++)
//				RANDOM_NUMBER_TEST<<<dim3(bx,by), dim3(tx,ty)>>> (i, devstate_GPU[cpu_thread_id]);
			//
			cudaDeviceSynchronize();
			#pragma omp barrier
		}  // end of openMP

		// simulation
		M_mean = 0;
		E_mean = 0;
		cudaEventRecord(start, 0);
			// thermalizatoin
		for (int N=0; N<N_thermal/interval_measure; N++)
		{
			#pragma omp parallel private (cpu_thread_id)
			{
				cpu_thread_id = omp_get_thread_num();

				ISING_ODD_BLOCK<<<dim3(bx/2,by), dim3(tx/2,ty), size_shared>>> (interval_measure, B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][0]], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][2]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], devstate_GPU[cpu_thread_id], odd_T_in_B_GPU[cpu_thread_id], even_T_in_B_GPU[cpu_thread_id]);
				cudaDeviceSynchronize();
				#pragma omp barrier

				ISING_EVEN_BLOCK<<<dim3(bx/2,by), dim3(tx/2,ty), size_shared>>> (interval_measure, B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][0]], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][2]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], devstate_GPU[cpu_thread_id], odd_T_in_B_GPU[cpu_thread_id], even_T_in_B_GPU[cpu_thread_id]);
				cudaDeviceSynchronize();
				#pragma omp barrier
			}	
		}
			//
			// measurements and updating
		for (int N=0; N<N_measure; N++)
		{
			#pragma omp parallel private(cpu_thread_id)
			{
				// measurement
				cpu_thread_id = omp_get_thread_num();
				ISING_MEASUREMENT<<<BpG, TpB, size_shared_measure>>> (B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], d_M_GPU[cpu_thread_id], d_E_GPU[cpu_thread_id]);
				cudaDeviceSynchronize();
				//
				cudaMemcpy(h_M+N_block*cpu_thread_id, d_M_GPU[cpu_thread_id], size_block, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_E+N_block*cpu_thread_id, d_E_GPU[cpu_thread_id], size_block, cudaMemcpyDeviceToHost);
				M_GPU[cpu_thread_id] = 0;
				E_GPU[cpu_thread_id] = 0;
				for (int i=0; i<N_block; i++)
				{
					M_GPU[cpu_thread_id] += h_M[i+N_block*cpu_thread_id];
					E_GPU[cpu_thread_id] += h_E[i+N_block*cpu_thread_id];
				}			
  		}
			M_save[N] = 0;
			E_save[N] = 0;
			for (int i=0; i<N_GPU; i++)
			{
//				M_save[N] += M_GPU[i];
				M_save[N] += abs(M_GPU[i]);
				E_save[N] += E_GPU[i];
			}
			M_mean += M_save[N];
			E_mean += E_save[N];
				//
				//save
			if ((N+1)%to_save==0)
			{
				#pragma omp parallel private(cpu_thread_id)
				{
					cpu_thread_id = omp_get_thread_num();
					for (int i=0; i<Ny/N_GPU_y; i++)
						cudaMemcpy(h_spin+gid_P2P[cpu_thread_id][4]+i*Nx/N_GPU_x, d_spin_GPU[cpu_thread_id]+i*Nx/N_GPU_x, Nx/N_GPU_x*sizeof(int), cudaMemcpyDeviceToHost);
					#pragma omp barrier
				}
				fprintf(output, "#%d Spin Configuration\n===============================================================\n", (N+1)/to_save);
				for (int j=0; j<Ny; j++)
				{
					for (int i=0; i<Nx; i++)	
						fprintf(output, "%d\t", h_spin[i+j*Nx]);
					fprintf(output, "\n");
				}
				fprintf(output, "\n");
			}
				//
			if ((N+1)%to_display==0)
				printf("%d\t\t%.6f\t\t%.6f\n", N+1, M_mean/(N+1)/N_site, E_mean/(N+1)/N_site ); 
				//updating
			#pragma omp parallel private(cpu_thread_id)
			{
				cpu_thread_id = omp_get_thread_num();

				ISING_ODD_BLOCK<<<dim3(bx/2,by), dim3(tx/2,ty), size_shared>>> (interval_measure, B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][0]], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][2]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], devstate_GPU[cpu_thread_id], odd_T_in_B_GPU[cpu_thread_id], even_T_in_B_GPU[cpu_thread_id]);
				cudaDeviceSynchronize();
				#pragma omp barrier
				ISING_EVEN_BLOCK<<<dim3(bx/2,by), dim3(tx/2,ty), size_shared>>> (interval_measure, B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][0]], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][2]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], devstate_GPU[cpu_thread_id], odd_T_in_B_GPU[cpu_thread_id], even_T_in_B_GPU[cpu_thread_id]);
				cudaDeviceSynchronize();
				#pragma omp barrier
			}
		}  // end of measurement and updating

//		#pragma omp parallel private(cpu_thread_id)
//		{
//			cpu_thread_id = omp_get_thread_num();
//			ISING_MEASUREMENT<<<BpG, TpB, size_shared_measure>>> (B, d_spin_GPU[cpu_thread_id], d_spin_GPU[gid_P2P[cpu_thread_id][1]], d_spin_GPU[gid_P2P[cpu_thread_id][3]], d_M_GPU[cpu_thread_id], d_E_GPU[cpu_thread_id]);
//			cudaDeviceSynchronize();
////			printf("%d\t%d\t%d\n", gid_P2P[cpu_thread_id][1], cpu_thread_id, gid_P2P[cpu_thread_id][3]);
//			cudaMemcpy(h_M+N_block*cpu_thread_id, d_M_GPU[cpu_thread_id], size_block, cudaMemcpyDeviceToHost);
//			cudaMemcpy(h_E+N_block*cpu_thread_id, d_E_GPU[cpu_thread_id], size_block, cudaMemcpyDeviceToHost);
//			M_GPU[cpu_thread_id] = 0;
//			E_GPU[cpu_thread_id] = 0;
//			for (int i=0; i<N_block; i++)
//			{
//				M_GPU[cpu_thread_id] += h_M[i+N_block*cpu_thread_id];
//				E_GPU[cpu_thread_id] += h_E[i+N_block*cpu_thread_id];
//			}			
//		}
//		double M_test = 0;
//		double E_test = 0;
//		for (int i=0; i<N_GPU; i++)
//		{
//			M_test += abs(M_GPU[i]);
//			E_test += E_GPU[i];
//		}
//		printf("%.6f\t%.6f\n", M_test/N_site, E_test/N_site);

			//analysis
		puts("===============================================================");
		fprintf(output2, "<M>\t<E>\n===============================================================\n");
		M_mean /= (double)(N_site)*(double)(N_measure);
		E_mean /= (double)(N_site)*(double)(N_measure);
		for (int i=0; i<N_measure; i++)
		{
//			M_sigma += pow(M_save[i]/N_site-M_mean, 2.0);
			M_sigma += pow(abs(M_save[i])/N_site-M_mean, 2.0);
			E_sigma += pow(E_save[i]/N_site-E_mean, 2.0);
			fprintf(output2, "%.6f\t%.6f\n", M_save[i]/N_site, E_save[i]/N_site);
		}
		M_sigma = sqrt(M_sigma/(N_measure-1));
		E_sigma = sqrt(E_sigma/(N_measure-1));
		puts("GPU Simulation result:");
		printf("\tMagnetization: Mean = %.6f\t Stand Deviation = %.6f\n", M_mean, M_sigma);
		printf("\tEnergy: Mean = %.6f\t Stand Deviation = %.6f\n", E_mean, E_sigma);
		// measurement and updating end
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_time, start, stop);
		total_time += gpu_time;
		printf("GPU time for simulation is %.4f ms.\n", gpu_time);		
		printf("Total GPU time is %.4f ms.\n", total_time);		
			//
		//  simulation ends

		#pragma omp parallel private(cpu_thread_id)
		{
			cpu_thread_id = omp_get_thread_num();
			cudaFree(d_spin_GPU[cpu_thread_id]);
			cudaFree(d_M_GPU[cpu_thread_id]);
			cudaFree(d_E_GPU[cpu_thread_id]);
			cudaFree(devstate_GPU[cpu_thread_id]);
			cudaDeviceDisablePeerAccess(gid[gid_P2P[cpu_thread_id][0]]);
			cudaDeviceDisablePeerAccess(gid[gid_P2P[cpu_thread_id][1]]);
			cudaDeviceDisablePeerAccess(gid[gid_P2P[cpu_thread_id][2]]);
			cudaDeviceDisablePeerAccess(gid[gid_P2P[cpu_thread_id][3]]);
			free(gid_P2P[cpu_thread_id]);
			if (cpu_thread_id!=0)
				cudaDeviceReset();
		}
		free(odd_T_in_B_GPU);
		free(even_T_in_B_GPU);
		free(devstate_GPU);
		free(d_spin_GPU);
		free(d_M_GPU);
		free(d_E_GPU);
		free(h_M);
		free(h_E);
		free(M_GPU);
		free(E_GPU);
		free(M_save);
		free(E_save);
		free(gid_P2P);
		fclose(output);
		fclose(output2);
	}

	if (mode!='g')
	{
		rng = gsl_rng_alloc(gsl_rng_mt19937);
		gsl_rng_set(rng, seed);
		if (mode=='c')
		{
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
		}
		// initialize the spin (for mode 'b', the spin is already initialized by GPU)
		if (ini=='c')
			for (int i=0; i<N_site; i++)
				h_spin[i] = 1;
		else
			for (int i=0; i<N_site; i++)
			{
				if (gsl_rng_uniform(rng)<=0.5)
					h_spin[i] = 1;
				else
					h_spin[i] = -1;
			}
		//
		cudaEventRecord(start, 0);
		ISING_CPU();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&cpu_time, start, stop);
		printf("CPU time for simulation is %.4f ms.\n", cpu_time);		
		if (mode=='b')
			printf("Speed-up by GPU is: %.2f\n", cpu_time/total_time);
		cudaEventDestroy(stop);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();
	free(h_spin);

	if ( B==0.0 )
	{
		EXACT_EM(&E_exact, &M_exact);
		printf("Exact solution: Tc = %.6f , M_exact = %.6f, E_exact = %.6f .\n", Tc, M_exact, E_exact);
	}
	else
		puts("Exact solution when B is non-zero is waiting to be found!");
}

__device__ int UPDATE_SPIN_GPU(int old_spin, int new_spin, int spin_around, double x)
{
	if (old_spin==1)
	{
		if (spin_around==4)
		{
			if (x<exp_update0_dev[0])
				return new_spin;
			else
				return old_spin;
		}
		else if (spin_around==2)
		{
			if (x<exp_update0_dev[1])
				return new_spin;
			else
				return old_spin;
		}
		else if (spin_around==0)
		{
			if (x<exp_update0_dev[2])
				return new_spin;
			else
				return old_spin;
		}
		else if (spin_around==-2)
		{
			if (x<exp_update0_dev[3])
				return new_spin;
			else
				return old_spin;
		}
		else if (spin_around==-4)
		{
			if (x<exp_update0_dev[4])
				return new_spin;
			else
				return old_spin;
		}
		else
		{
			printf("Error happens for old_spin parallel to B! Exit!\n");
			printf("Spin aournd = %d\n", spin_around);
			return 0;
		}
	}
	else if (old_spin==-1)
	{
		if (spin_around==-4)
		{
			if (x<exp_update1_dev[0])
				return new_spin;
			else
				return old_spin;
		}
		else if (spin_around==-2)
		{
			if (x<exp_update1_dev[1])
				return new_spin;
			else
				return old_spin;
		}
		else
		{
			printf("Error happens for old spin anti-parallel to B! Exit!\n");
			printf("Spin aournd = %d\n", spin_around);
			return 0;
		}
	}
	else
	{
		printf("Error for old spin! Exit!\n");
		printf("Old spin = %d\n", old_spin);
		return 0;
	}
}

void EXACT_EM (double *E_exact, double *M_exact)
{
	double k = 1.0/sinh(2.0/T)/sinh(2.0/T);
	Tc =  2.0/log(1.0+sqrt(2.0));

	*E_exact = -1.0/tanh(2.0/T)*( 1.0 + 2.0/M_PI*(2.0*tanh(2.0/T)*tanh(2.0/T)-1.0)*gsl_sf_ellint_Kcomp(2.0*sqrt(k)/(1.0+k), GSL_PREC_DOUBLE) );

	if ( T<=Tc )
		*M_exact = pow( (1.0-pow(sinh(2.0/T), -4.0)), 0.125);
	else
		*M_exact = 0.0;
}

void UPDATE_MATRIX (double *A1, double *A2)
{
	A1[0] = exp(-2.0*(4.0+B)/T);
	A1[1] = exp(-2.0*(2.0+B)/T);
	A1[2] = exp(-2.0*B/T);
	A1[3] = exp(-2.0*(-2.0+B)/T);
	A1[4] = exp(-2.0*(-4.0+B)/T);

	A2[0] = exp(-2.0*(4.0-B)/T);
	A2[1] = exp(-2.0*(2.0-B)/T);
}

//void UPDATE_SPIN(int center, int old_spin, int new_spin, int spin_around, double delta_E, int *h_spin, double **exp_update, gsl_rng *rng)
void UPDATE_SPIN(int center, int old_spin, int new_spin, int spin_around, int delta_E, double **exp_update)
{
	if (delta_E<=0)
		h_spin[center] = new_spin;
	else if (delta_E>0)
	{
		if (old_spin==1)
		{
			if (spin_around==4)
			{
				if (gsl_rng_uniform(rng)<exp_update[0][0])
					h_spin[center] = new_spin;
			}
			else if (spin_around==2)
			{
				if (gsl_rng_uniform(rng)<exp_update[0][1])
					h_spin[center] = new_spin;
			}
			else if (spin_around==0)
			{
				if (gsl_rng_uniform(rng)<exp_update[0][2])
					h_spin[center] = new_spin;
			}
			else if (spin_around==-2)
			{
				if (gsl_rng_uniform(rng)<exp_update[0][3])
					h_spin[center] = new_spin;
			}
			else if (spin_around==-4)
			{
				if (gsl_rng_uniform(rng)<exp_update[0][4])
					h_spin[center] = new_spin;
			}
			else
			{
				puts("Error happens for old_spin parallel to B! Exit!");
				printf("Spin aournd = %d\n", spin_around);
				exit(1);
			}
		}
		else if (old_spin==-1)
		{
			if (spin_around==-4)
			{
				if (gsl_rng_uniform(rng)<exp_update[1][0])
					h_spin[center] = new_spin;
			}
			else if (spin_around==-2)
			{
				if (gsl_rng_uniform(rng)<exp_update[1][1])
					h_spin[center] = new_spin;
			}
			else
			{
				puts("Error happens for old spin anti-parallel to B! Exit!");
				printf("Spin aournd = %d\n", spin_around);
				exit(1);
			}
		}
		else
		{
			puts("Error for old spin! Exit!");
			printf("Old spin = %d\n", old_spin);
			exit(1);
		}
	} // end of delta_E>0 case
}

void ISING_CPU (void)
{
	int x, y, parity;
	int center, fw_x, bw_x, fw_y, bw_y, spin_around;
	int old_spin, new_spin, delta_E;
	int i_o = (int)(N_site/2);
	int i_e = (int)((N_site+1)/2);
	int *h_spin_odd = (int*)malloc(i_o*sizeof(int));
	int *h_spin_even = (int*)malloc(i_e*sizeof(int));
	double **exp_update = (double**)malloc(2*sizeof(double*));
	exp_update[0] = (double*)malloc(5*sizeof(double));
	exp_update[1] = (double*)malloc(2*sizeof(double));
//	gsl_rng rng;
//	rng = gsl_rng_alloc(gsl_rng_mt19937);
//	gsl_rng_set(rng, seed);
	UPDATE_MATRIX(exp_update[0], exp_update[1]);
	
	//odd site
	for (int i=0; i<i_o; i++)
	{
		x = (2*i)%Nx;
		y = (2*i)/Nx;
		parity = (x+y+1)%2;
		x = x+parity;
		h_spin_odd[i] = x+Nx*y;
	}
	//
	//even site
	for (int i=0; i<i_e; i++)
	{
		x = (2*i)%Nx;
		y = (2*i)/Nx;
		parity = (x+y)%2;
		x = x+parity;
		h_spin_even[i] = x+Nx*y;
	}
	//
	for (int N=0; N<N_thermal; N++)
	{
		for (int i=0; i<i_o; i++)
		{
			center = h_spin_odd[i];
			x = center%Nx;
			y = center/Nx;
			fw_x = (x+1)%Nx+y*Nx;
			bw_x = (x-1+Nx)%Nx+y*Nx;
			fw_y = x+((y+1)%Ny)*Nx;
			bw_y = x+((y-1+Ny)%Ny)*Nx;
			spin_around = h_spin[fw_x]+h_spin[bw_x]+h_spin[fw_y]+h_spin[bw_y];
			old_spin = h_spin[center];
			new_spin = -old_spin;
			delta_E = (old_spin-new_spin)*(spin_around+B); // delta_E = E_new-E_ole = -(spin_new[center]-spin_old[center])*(summation_of_neighboring_spin+B)
//			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, h_spin, exp_update, rng);
			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, exp_update);
		} // end of odd site
		for (int i=0; i<i_e; i++)
		{
			center = h_spin_even[i];
			x = center%Nx;
			y = center/Nx;
			fw_x = (x+1)%Nx+y*Nx;
			bw_x = (x-1+Nx)%Nx+y*Nx;
			fw_y = x+((y+1)%Ny)*Nx;
			bw_y = x+((y-1+Ny)%Ny)*Nx;
			spin_around = h_spin[fw_x]+h_spin[bw_x]+h_spin[fw_y]+h_spin[bw_y];
			old_spin = h_spin[center];
			new_spin = -old_spin;
			delta_E = (old_spin-new_spin)*(spin_around+B); // delta_E = E_new-E_ole = -(spin_new[center]-spin_old[center])*(summation_of_neighboring_spin+B)
//			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, h_spin, exp_update, rng);
			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, exp_update);
		} // end even_site
	} // end of thermalization

	int N = 0;
	int count = 1;
	double M_mean = 0, E_mean = 0;
	double M_sigma = 0, E_sigma = 0;
	double *M_save = (double*)malloc(N_measure*sizeof(double));
	double *E_save = (double*)malloc(N_measure*sizeof(double));
	FILE *output = fopen("spin_config_cpu.txt", "w");
	FILE *output2 = fopen("M_and_E_cpu.txt", "w");
	puts("index of Meas.\t<M>\t\t<E>\n===============================================================");
	while (N<N_measure)
	{
		if (count%interval_measure==0)
		{
			int M = 0, E = 0;
			for (int i=0; i<N_site; i++)
			{
				x = i%Nx;
				y = i/Nx;
				fw_x = (x+1)%Nx+y*Nx;
				fw_y = x+((y+1)%Ny)*Nx;
				spin_around = h_spin[fw_x]+h_spin[fw_y];
				M += h_spin[i];
				E += -h_spin[i]*(spin_around+B);
			}
//			M_mean += M;
			M_mean += abs(M);
			E_mean += E;
			M_save[N] = M;
			E_save[N] = E;
			N++;
		}

		if (count%interval_save==0)
		{
			fprintf(output, "#%d Spin Configuration\n===============================================================\n", (int)(count/interval_save));
			for (int j=0; j<Ny; j++)
			{
				for (int i=0; i<Nx; i++)	
					fprintf(output, "%d\t", h_spin[i+j*Nx]);
				fprintf(output, "\n");
			}
			fprintf(output, "\n");
		}
		
		if (count%interval_display==0)
			printf("%d\t\t%.6f\t\t%.6f\n", N, M_mean/N/N_site, E_mean/N/N_site ); 

		for (int i=0; i<i_o; i++)
		{
			center = h_spin_odd[i];
			x = center%Nx;
			y = center/Nx;
			fw_x = (x+1)%Nx+y*Nx;
			bw_x = (x-1+Nx)%Nx+y*Nx;
			fw_y = x+((y+1)%Ny)*Nx;
			bw_y = x+((y-1+Ny)%Ny)*Nx;
			spin_around = h_spin[fw_x]+h_spin[bw_x]+h_spin[fw_y]+h_spin[bw_y];
			old_spin = h_spin[center];
			new_spin = -old_spin;
			delta_E = (old_spin-new_spin)*(spin_around+B); // delta_E = E_new-E_ole = -(spin_new[center]-spin_old[center])*(summation_of_neighboring_spin+B)
//			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, h_spin, exp_update, rng);
			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, exp_update);
		} // end of odd site
		for (int i=0; i<i_e; i++)
		{
			center = h_spin_even[i];
			x = center%Nx;
			y = center/Nx;
			fw_x = (x+1)%Nx+y*Nx;
			bw_x = (x-1+Nx)%Nx+y*Nx;
			fw_y = x+((y+1)%Ny)*Nx;
			bw_y = x+((y-1+Ny)%Ny)*Nx;
			spin_around = h_spin[fw_x]+h_spin[bw_x]+h_spin[fw_y]+h_spin[bw_y];
			old_spin = h_spin[center];
			new_spin = -old_spin;
			delta_E = (old_spin-new_spin)*(spin_around+B); // delta_E = E_new-E_ole = -(spin_new[center]-spin_old[center])*(summation_of_neighboring_spin+B)
//			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, h_spin, exp_update, rng);
			UPDATE_SPIN(center, old_spin, new_spin, spin_around, delta_E, exp_update);
		} // end even_site
		count++;
	} // end of measurements
	puts("===============================================================");
	fprintf(output2, "<M>\t<E>\n===============================================================\n");
	M_mean /= (N_site*N_measure);
	E_mean /= (N_site*N_measure);
	for (int i=0; i<N_measure; i++)
	{
//		M_sigma += pow(M_save[i]/N_site-M_mean, 2.0);
		M_sigma += pow(abs(M_save[i])/N_site-M_mean, 2.0);
		E_sigma += pow(E_save[i]/N_site-E_mean, 2.0);
		fprintf(output2, "%.6f\t%.6f\n", M_save[i]/N_site, E_save[i]/N_site);
	}
	M_sigma = sqrt(M_sigma/(N_measure-1));
	E_sigma = sqrt(E_sigma/(N_measure-1));
	puts("CPU Simulation result:");
	printf("\tMagnetization: Mean = %.6f\t Stand Deviation = %.6f\n", M_mean, M_sigma);
	printf("\tEnergy: Mean = %.6f\t Stand Deviation = %.6f\n", E_mean, E_sigma);

	free(h_spin_odd);
	free(h_spin_even);
	free(exp_update);
	free(M_save);
	free(E_save);
	fclose(output);
	fclose(output2);
} // end of ISING_CPU


