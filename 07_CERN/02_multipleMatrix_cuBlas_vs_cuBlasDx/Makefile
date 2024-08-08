CUDA_ROOT=/usr/local/cuda

#Needs extra include for cublasdx
MATHDX_INCLUDE_DIR := -I /home/stloufra/cuBLASDx/nvidia/mathdx/24.01/include
CUTLASS_INCLUDE_DIR := -I /home/stloufra/cuBLASDx/nvidia/mathdx/24.01/external/cutlass/include
CUBLASDX_INCLUDE_DIR := -I /home/stloufra/cuBLASDx/nvidia/mathdx/24.01/include/cublasdx/include


CUDA_INCLUDE_DIR := $(MATHDX_INCLUDE_DIR) $(CUTLASS_INCLUDE_DIR) $(CUBLASDX_INCLUDE_DIR)

#Flags for auto included libraries
CUDA_CXX=$(CUDA_ROOT)/bin/nvcc
CUDA_CXX_FLAGS = -lcublas -lcurand

#not known warning
CUDA_SUPRESS_FLAG = -diag-suppress 20013
CUDA_CXX_FLAGS += $(CUDA_SUPRESS_FLAG)

#separate compilation for dynamic parallelism
CUDA_CXX_FLAGS += -rdc=true -lcudadevrt


#Compute capability needs to be higher than 6.0 but docu. mention 7.0
CUDA_ARCH = -arch sm_86
CUDA_CXX_FLAGS += $(CUDA_ARCH)

TARGET = main
SRC = main.cu
run:
	make clean
	make all
	./main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CUDA_CXX) $(SRC) $(CUDA_INCLUDE_DIR) $(CUDA_CXX_FLAGS)  -o $(TARGET)


clean:
	rm -f $(TARGET)

.PHONY: all clean run