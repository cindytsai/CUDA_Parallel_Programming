CUDIR     := /usr/local/nvidia

SM    := sm_52
CP    := 52

#SM    := sm_61
#CP    := 61

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g --std=c++11
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_$(CP) -code=$(SM) -Xptxas -v -rdc=true --std=c++11 --compiler-options -fopenmp
LIB     := -lgsl -lgslcblas -lcurand -lcudadevrt
#LIB     := -lgsl -lgslcblas -lcudadevrt

BIN1    := ising2d_cpu
BIN2   	:= ising2d_1gpu_gmem_v2 
BIN3 	:= ising2d_Ngpu_gmem_v2

all: $(BIN1) $(BIN2) $(BIN3)

$(BIN1): ising2d_cpu.cu
	$(NVCC) -o $(BIN1) $(NVFLAGS) ising2d_cpu.cu $(LIB)

$(BIN2): ising2d_1gpu_gmem_v2.cu
	$(NVCC) -o $(BIN2) $(NVFLAGS) ising2d_1gpu_gmem_v2.cu $(LIB)

$(BIN3): ising2d_Ngpu_gmem_v2.cu
	$(NVCC) -o $(BIN3) $(NVFLAGS) ising2d_Ngpu_gmem_v2.cu $(LIB)


clean:
	rm -f $(BIN1) $(BIN2) $(BIN3)
