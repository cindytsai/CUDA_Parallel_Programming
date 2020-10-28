################################################################################
#
# Build script for project
#
################################################################################

CXX     := icc
CFLAGS  := -O3
NVCC    := nvcc
#CUFLAGS := -arch=compute_52 -code=sm_52,sm_52 -O3 -m64 --compiler-options -fno-strict-aliasing -DUNIX -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler -fopenmp -rdc=true

CUFLAGS := -arch=compute_52 -code=sm_52,sm_52 -O3 --compiler-options -fopenmp
#CUFLAGS := -O3 --compiler-options -fopenmp

BIN     := ising_model_1gpu_backup
OBJ     := ising_model_1gpu_backup.o
LIB     := -lgsl -lgslcblas -lcudadevrt -lcurand

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN)

$(BIN): $(OBJ)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

ising_model.o:  ising_model_1gpu_backup.cu

clean:
	rm -f $(OBJ) $(BIN)

