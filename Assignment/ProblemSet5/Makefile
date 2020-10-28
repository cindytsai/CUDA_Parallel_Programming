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

BIN     := heat_diffusion
OBJ     := heat_diffusion.o
LIB     :=

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

all: $(BIN)

$(BIN): $(OBJ)
#	$(NVCC) -Xcompiler -fopenmp -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)
	$(NVCC) -o $(BIN) $(CUFLAGS) $(OBJ) $(LIB)

heat_diffusion.o:  heat_diffusion.cu

clean:
	rm -f $(OBJ) $(BIN)

