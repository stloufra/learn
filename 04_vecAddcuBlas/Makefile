# Makefile for compiling CUDA program with cuBLAS

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -lcublas

# Target executable
TARGET = main

# Source file
SRC = main.cu

# Default rule
all: $(TARGET)

# Rule for compiling the CUDA program
$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCC_FLAGS) -o $(TARGET)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)

.PHONY: all clean