#include <complex>
#include <cstdio>
#include <cufft.h>
#include <math.h>
#include <string.h>

using namespace std;

__global__ void getPhi(double *d_phi_k, int Nx, int Ny, int Nz_half, double L){
    // Change made inside d_phi_k
    
    int N = Nx * Ny * Nz_half;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int nx, ny; 
    double kx2, ky2, kz2;
    
    while(index < N) {
        int i = index / (Ny * Nz_half);
        int j = (index / Nz_half) % Ny;
        int k = index % Nz_half;

        if(2 * i < Nx){
            nx = i;
        }
        else{
            nx = Nx - i;
        }
        if(2 * j < Ny){
            ny = j;
        }
        else{
            ny = Ny - j;
        }

        kx2 = pow(2.0 * M_PI * (double)nx / L, 2);
        ky2 = pow(2.0 * M_PI * (double)ny / L, 2);
        kz2 = pow(2.0 * M_PI * (double)k  / L, 2);

        if(index != 0){
            // d_phi_k[2*index]   = 4.0 * M_PI * d_phi_k[2*index] / (kx2 + ky2 + kz2);
            // d_phi_k[2*index+1] = 4.0 * M_PI * d_phi_k[2*index+1] / (kx2 + ky2 + kz2);
            d_phi_k[2*index]   = d_phi_k[2*index] / (kx2 + ky2 + kz2);
            d_phi_k[2*index+1] = d_phi_k[2*index+1] / (kx2 + ky2 + kz2);
        }

        index = index + blockDim.x * gridDim.x;

    }

}

int main ()
{
    // Set GPU Device
    int gid;
    printf("Enter the GPU ID (0/1): ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    cudaSetDevice(gid);

    int Nx, Ny, Nz, N;
    printf("Enter the sample points of the cube in each side: ");
    scanf("%d", &Nx);
    printf("Each side sample points = %d\n", Nx);
    Ny = Nx;
    Nz = Nx;
    N = pow(Nx, 3);


    double dx = 1.0;    // First fixed dx. TODO
    double L = dx * (double)Nx;

    // Do not fix dx
    // printf("Enter the length of the cube: ");
    // scanf("%lf", &L);
    // printf("Length = %.2lf\n", L);
    // dx = L / (double) Nx;
    // printf("dx = %.2lf\n", dx);

    int io;
    printf("Print the data (0/1) ? ");
    scanf("%d",&io);
    printf("%d\n", io);

    /*
    Initialize
     */
    double *lo;
    complex<double> *lo_k;

    lo   = (double*) malloc(sizeof(double) * N);
    lo_k = (complex<double> *) malloc(sizeof(complex<double>) * Nx * Ny * (Nz/2+1));
    memset(lo, 0.0, sizeof(double) * N);

    // point charge at the origin
    lo[0] = 1.0;

    /*
    Poisson Eq with FFT method
     */ 
    // FFT lo -> lo_k
    cufftHandle plan;
    cufftDoubleReal *dataIn;
    cufftDoubleComplex *dataOut;

    cudaMalloc((void**)&dataIn, sizeof(cufftDoubleReal) * N);
    cudaMalloc((void**)&dataOut, sizeof(cufftDoubleComplex) * N);   
    cudaMemcpy(dataIn, lo, sizeof(cufftDoubleReal) * N, cudaMemcpyHostToDevice);

    if(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_D2Z) != CUFFT_SUCCESS){
        printf("CUFFT error: cufftPlan3d creation failed.\n");
        exit(1);
    }

    if(cufftExecD2Z(plan, dataIn, dataOut) != CUFFT_SUCCESS){
        printf("CUFFT error: cufftExecD2Z forward failed.\n");
        exit(1);
    }

    // Copy only the non redundant data
    cudaMemcpy(lo_k, dataOut, sizeof(cufftDoubleComplex) * Nx * Ny * (Nz/2+1), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cufftDestroy(plan);
    cudaFree(dataIn);
    cudaFree(dataOut);
    free(lo);

    // Print the data of lo_k
    // for(int i = 0; i < Nx * Ny * (Nz/2+1); i = i+1){
    //     printf("%.3lf + i * %.3lf\n", real(lo_k[i]), imag(lo_k[i]));
    // }   
    
    // Calculate lo_k / k**2 = phi_k
    complex<double> *phi_k;
    double *d_phi_k;

    phi_k = (complex<double> *)malloc(sizeof(complex<double>) * Nx * Ny * (Nz/2+1));
    cudaMalloc((void**)&d_phi_k, sizeof(double) * 2 * Nx * Ny * (Nz/2+1));
    cudaMemcpy(d_phi_k, lo_k, sizeof(double) * 2 * Nx * Ny * (Nz/2+ 1), cudaMemcpyHostToDevice);

    getPhi <<<64, 64>>> (d_phi_k, Nx, Ny, Nz/2+1, L);

    cudaMemcpy(phi_k, d_phi_k, sizeof(double) * 2 * Nx * Ny * (Nz/2+1), cudaMemcpyDeviceToHost);

    cudaFree(d_phi_k);
    free(lo_k);

    // IFFT phi_k -> phi
    double *phi;

    phi = (double*) malloc(sizeof(double) * N);
    cudaMalloc((void**)&dataIn, sizeof(cufftDoubleReal) * N);
    cudaMalloc((void**)&dataOut, sizeof(cufftDoubleComplex) * Nx * Ny * (Nz/2+1));   

    cudaMemcpy(dataOut, phi_k, sizeof(cufftDoubleComplex) * Nx * Ny * (Nz/2+1), cudaMemcpyHostToDevice);

    if(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_Z2D) != CUFFT_SUCCESS){
        printf("CUFFT error: cufftPlan3d creation failed.\n");
        exit(1);
    }
    if(cufftExecZ2D(plan, dataOut, dataIn) != CUFFT_SUCCESS){
        printf("CUFFT error: cufftExecZ2D forward failed.\n");
        exit(1);
    }

    cudaMemcpy(phi, dataIn, sizeof(cufftDoubleReal) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(dataIn);
    cudaFree(dataOut);
    free(phi_k);

    // Print out on screen
    if(io == 1){
        printf("phi-X r phi-D r\n");
        for(int i = 0; i < Nx; i =  i+1){
            printf("%.5lf %.5lf ", (phi[i] - phi[1]) / (double)N, (double)i * dx);
            printf("%.5lf %.5lf\n", (phi[i*Ny*Nz + i*Ny + i] - phi[1]) / (double)N, sqrt(3.0 * pow((double)i * dx,2)));
        }
    }

    // Print to file

    cufftDestroy(plan);
    cudaDeviceReset();

    return 0;
}

// eof
