//  Monte Carlo simulation of Ising model on 2D lattice
//  using Metropolis algorithm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>

void exact_2d(double, double, double*, double*);

double ellf(double phi, double ak);
double rf(double x, double y, double z);
double min(double x, double y, double z);
double max(double x, double y, double z);

int main(void) {
  int nx,ny; 		// # of sites in x and y directions respectively
  int ns; 		// ns = nx*ny, total # of sites
  int old_spin;
  int new_spin;       	// old and new spin values at the site
  int spins; 		// sum of neighbouring spins
  int *spin;	        // spin variables
  int *fw;      	// forward index
  int *bw; 	        // backward index
  int nt; 		// # of sweeps for thermalization
  int nm; 		// # of measurements
  int im; 		// interval between successive measurements
  int nd; 		// # of sweeps between displaying results
  int nb; 		// # of sweeps before saving spin configurations
  int sweeps; 		// total # of sweeps at each temperature
  int k1, k2, k3, k4;   // right, top, left, bottom
  int i1, j1;
  int istart; 		// istart = (0: cold start/1: hot start)
  double T; 		// temperature
  double B; 		// external magnetic field
  double energy; 	// total energy of the system
  double mag; 		// total magnetization of the system
  double de; 		// the change of energy due to a spin flip
  double te; 		// accumulator for energy
  double tm; 		// accumulator for mag
  double count; 	// counter for # of measurements
  double M; 		// magnetization per site, < M >
  double E; 		// energy per site, < E >
  double E_ex; 		// exact solution of < E >
  double M_ex; 		// exact solution of < M >
  gsl_rng *rng=NULL;    // pointer to gsl_rng random number generator

  printf("Ising Model on 2D Square Lattice with p.b.c.\n");
  printf("============================================\n");
  printf("Initialize the RNG\n");
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  printf("Enter the seed:\n");
  long seed;
  scanf("%ld",&seed);
  printf("%ld\n",seed); 
  gsl_rng_set(rng,seed);
  printf("The RNG has been initialized\n");
  printf("Enter the number of sites in each dimension\n");
  scanf("%d",&nx);
  printf("%d\n",nx);
  ny=nx;
  ns=nx*ny;
  fw = (int*)malloc(nx*sizeof(int));
  bw = (int*)malloc(nx*sizeof(int));
  for(int i=0; i<nx; i++) {
    fw[i]=(i+1)%nx;
    bw[i]=(i-1+nx)%nx;
  }
  spin = (int*)malloc(ns*sizeof(int));
  printf("Enter the # of sweeps for thermalization\n");
  scanf("%d",&nt);
  printf("%d\n",nt);
  printf("Enter the # of measurements\n");
  scanf("%d",&nm);
  printf("%d\n",nm);
  printf("Enter the interval between successive measurements\n");
  scanf("%d",&im);
  printf("%d\n",im);
  printf("Enter the display interval\n");
  scanf("%d",&nd);
  printf("%d\n",nd);
  printf("Enter the temperature (in units of J/k)\n");
  scanf("%lf",&T);
  printf("%lf\n",T);
  printf("Enter the external magnetization\n");
  scanf("%lf",&B);
  printf("%lf\n",B);
  printf("Initialize spins configurations :\n");
  printf(" 0: cold start \n");
  printf(" 1: hot start \n");
  scanf("%d",&istart);
  printf("%d\n",istart);

  if(istart == 0) {
    for(int j=0; j<ns; j++)       // cold start
      spin[j] = 1;
  }
  else {
    for(int j=0; j<ns; j++) {     // hot start
      if(gsl_rng_uniform(rng) > 0.5) 
        spin[j] = 1;
      else 
        spin[j] = -1;
    }
  }

  FILE *output;            
  output = fopen("ising2d_cpu.dat","w"); 

  if(B == 0.0) {
    exact_2d(T,B,&E_ex,&M_ex);
    fprintf(output,"T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    fprintf(output,"T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }
  fprintf(output,"     E           M        \n");
  fprintf(output,"--------------------------\n");

  printf("Thermalizing\n");
  printf("sweeps   < E >     < M >\n");
  printf("---------------------------------\n");
  fflush(stdout);

  te=0.0;                          //  initialize the accumulators
  tm=0.0;
  count=0.0;
  sweeps=nt+nm*im;                 //  total # of sweeps
  int k;

  // create the timer
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //start the timer
  cudaEventRecord(start,0);

  for(int swp=1; swp<sweeps; swp++) {
    for(j1=0; j1<ny; j1++) {
      for(i1=0; i1<nx; i1++) {
        k = i1 + j1*nx;
        old_spin = spin[k];
        new_spin = -old_spin;
        k1 = fw[i1] + j1*nx;     // right
        k2 = i1 + fw[j1]*nx;     // top
        k3 = bw[i1] + j1*nx;     // left
        k4 = i1 + bw[j1]*nx;     // bottom
        spins = spin[k1] + spin[k2] + spin[k3] + spin[k4];
        de = -(new_spin - old_spin)*(spins + B);
        if((de <= 0.0) || (gsl_rng_uniform(rng) < exp(-de/T))) {
// if (new energy <= old energy) or (r < exp(-dE/kT))  accept the new spin
          spin[k] = new_spin;         // accept the new spin;
        }
      }
    }
    if((swp > nt) && (swp%im == 0)) {
      mag=0.0;
      energy=0.0;
      for(int j=0; j<ny; j++) {
        for(int i=0; i<nx; i++) {
          k = i + j*nx;
          mag = mag + spin[k]; // total magnetization;
          k1 = fw[i] + j*nx;
          k2 = i + fw[j]*nx;
          energy = energy - spin[k]*(spin[k1] + spin[k2]);  // total bond energy;
        }
      }
      energy = energy - B*mag;
      te = te + energy;
      tm = tm + mag;
      count = count + 1.0;
      fprintf(output, "%.5e  %.5e\n", energy/(double)ns, mag/(double)ns);  // save the raw data 
    }
    if((swp > nt) && (swp%nd == 0)) {
      E = te/(count*(double)(ns));
      M = tm/(count*(double)(ns));
      printf("%d  %.5e  %.5e\n", swp, E, M);
    }

  }
  fclose(output);
  printf("---------------------------------\n");
  if(B == 0.0) {
    printf("T=%.5e  B=%.5e  ns=%d  E_exact=%.5e  M_exact=%.5e\n", T, B, ns, E_ex, M_ex);
  }
  else {
    printf("T=%.5e  B=%.5e  ns=%d\n", T, B, ns);
  }

  // stop the timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  float cputime;
  cudaEventElapsedTime( &cputime, start, stop);
  printf("Processing time for CPU: %f (ms) \n",cputime);
  float flops = 7.0*nx*nx*sweeps;
  printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

  // destroy the timer
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  gsl_rng_free(rng);

  return 0;
}
          
          
// Exact solution of 2d Ising model on the infinite lattice

void exact_2d(double T, double B, double *E, double *M)
{
  double x, y;
  double z, Tc, K, K1;
  const double pi = acos(-1.0);
    
  K = 2.0/T;
  if(B == 0.0) {
    Tc = -2.0/log(sqrt(2.0) - 1.0); // critical temperature;
    if(T > Tc) {
      *M = 0.0;
    }
    else if(T < Tc) {
      z = exp(-K);
      *M = pow(1.0 + z*z,0.25)*pow(1.0 - 6.0*z*z + pow(z,4),0.125)/sqrt(1.0 - z*z);
    }
    x = 0.5*pi;
    y = 2.0*sinh(K)/pow(cosh(K),2);
    K1 = ellf(x, y);
    *E = -1.0/tanh(K)*(1. + 2.0/pi*K1*(2.0*pow(tanh(K),2) - 1.0));
  }
  else
    printf("Exact solution is only known for B=0 !\n");
    
  return;
}


/*******
* ellf *      Elliptic integral of the 1st kind 
*******/

double ellf(double phi, double ak)
{
  double ellf;
  double s;

  s=sin(phi);
  ellf=s*rf(pow(cos(phi),2),(1.0-s*ak)*(1.0+s*ak),1.0);

  return ellf;
}

double rf(double x, double y, double z)
{
  double rf,ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4;
  ERRTOL=0.08; 
  TINY=1.5e-38; 
  BIG=3.0e37; 
  THIRD=1.0/3.0;
  C1=1.0/24.0; 
  C2=0.1; 
  C3=3.0/44.0; 
  C4=1.0/14.0;
  double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;
    
  if(min(x,y,z) < 0 || min(x+y,x+z,y+z) < TINY || max(x,y,z) > BIG) {
    printf("invalid arguments in rf\n");
    exit(1);
  }

  xt=x;
  yt=y;
  zt=z;

  do {
    sqrtx=sqrt(xt);
    sqrty=sqrt(yt);
    sqrtz=sqrt(zt);
    alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
    xt=0.25*(xt+alamb);
    yt=0.25*(yt+alamb);
    zt=0.25*(zt+alamb);
    ave=THIRD*(xt+yt+zt);
    delx=(ave-xt)/ave;
    dely=(ave-yt)/ave;
    delz=(ave-zt)/ave;
  } 
  while (max(abs(delx),abs(dely),abs(delz)) > ERRTOL);

  e2=delx*dely-pow(delz,2);
  e3=delx*dely*delz;
  rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
    
  return rf;
}

double min(double x, double y, double z)
{
  double m;

  m = (x < y) ? x : y;
  m = (m < z) ? m : z;

  return m;
}

double max(double x, double y, double z)
{
  double m;

  m = (x > y) ? x : y;
  m = (m > z) ? m : z;

  return m;
}

