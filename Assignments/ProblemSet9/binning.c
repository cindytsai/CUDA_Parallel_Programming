#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

void Binning(double*, int, int, char*);

int main(int argc, char *argv[]){
	
	char filename[100];		// filename
	int N = 1024;			// measurements
	int bmax2 = 10;
	double *E, *M;

	E = (double*) malloc(N * sizeof(double));
	M = (double*) malloc(N * sizeof(double));

	// Read from file
	strcpy(filename, argv[1]);
	FILE *infile;
	infile = fopen(filename,"r");
	for(int i = 1; i <= N; i = i+1) {
		fscanf(infile,"%lf %lf", &E[i], &M[i]);
	}
	fclose(infile);

	Binning(E, bmax2, N, "E.txt");
	Binning(M, bmax2, N, "M.txt");

	return 0;
}

void Binning(double *A, int bmax2, int N, char *file_name){
    // General Settings
    // numbers of blocks m from : 2^1 ~ 2^bmax2
    // numbers of data per block --> nb
    // block average --> B[i]
    // # of array A  --> N
    int m, nb;
    double *B, *Bjk;
    double Bjk_ave, errorJK;

    // Output as file with file name file_name
    FILE *output;
    output = fopen(file_name, "w");

    fprintf(output, "m nb Bjk_ave errorJK\n");

    for(int i = bmax2; i >= 1; i = i-1){
        
        m = pow(2, i);
        nb = N / m;
        B = (double*) malloc(m * sizeof(double));
        Bjk = (double*) malloc(m * sizeof(double));
        Bjk_ave = 0.0;
        errorJK = 0.0;

        // Calculate B_i
        for(int j = 0; j < m; j = j+1){
            B[j] = 0.0;
        }
        for(int j = 0; j < N; j = j+1){
            B[j/nb] = B[j/nb] + (A[j] / (double)nb);
        }

        // Calculate Bjk_s
        for(int s = 0; s < m; s = s+1){
            Bjk[s] = 0.0;
            for(int j = 0; j < m; j = j+1){
                if(j != s){
                    Bjk[s] = Bjk[s] + B[j];
                }
            }
            Bjk[s] = Bjk[s] / (double)(m-1);
        }

        // Calculate Bjk_ave
        for(int j = 0; j < m; j = j+1){
            Bjk_ave = Bjk_ave + Bjk[j];
        }
        Bjk_ave = Bjk_ave / (double)m;

        // Calculate errorJK
        for(int j = 0; j < m; j = j+1){
            errorJK = errorJK + pow((Bjk[j] - Bjk_ave), 2);
        }
        errorJK = errorJK * ((double)(m - 1) / (double)m);
        errorJK = sqrt(errorJK);

        // Print out the result
        fprintf(output, "%d %d %1.5e %1.5e\n", m, nb, Bjk_ave, errorJK);

    }
    // Close file
    fclose(output);
}
