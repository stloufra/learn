CUDA_CXX := nvcc

CUDA_CXXFLAGS := -std=c++17
#CUDA_CXXFLAGS += -DHAVE_CUDA -DNDEBUG
#CUDA_CXXFLAGS +=  -use_fast_math -O3

CUDA_PATH ?= $(abspath $(dir $(shell command -v nvcc))/..)

#$(info Detected CUDA_PATH: $(CUDA_PATH))

# flags for linking CUDA with the host compiler
CUDA_LDFLAGS := -L $(CUDA_PATH)/lib64
CUDA_LDLIBS := -lcudart -ldl -lrt

CUDA_SOURCES := $(wildcard *.cu)
CUDA_TARGETS := $(CUDA_SOURCES:%.cu=%)

PHONY: all cuda
all: cuda
cuda: $(CUDA_TARGETS)

.PHONY: clean
clean:
	rm -f *.d *.o *.cuo $(CUDA_TARGETS)

$(CUDA_TARGETS): % : %.cuo
	$(CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS)

$(CUDA_SOURCES:%.cu=%.cuo): %.cuo : %.cu
	$(CUDA_CXX) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -c -o $@ $<