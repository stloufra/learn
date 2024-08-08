#include <iostream>

#include "func.h"
#include "defs.h"

//#define DEBUG

int main() {

    size_t elements = N * N;
    size_t bytes = elements * sizeof(realType);

    int *d_result;
    realType *d_a, *d_b, *d_cublas, *d_cublasdx, *d_kernel;

    unsigned int NUM_THREADS = 32;
    unsigned int NUM_BLOCKS_1D = (unsigned int) (elements / NUM_THREADS + 1);
    unsigned int NUM_BLOCKS_2D = (unsigned int) (N / NUM_THREADS + 1);

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(NUM_BLOCKS_2D, NUM_BLOCKS_2D);

    realType epsilon = 0.0001;

    //Allocate CUDA memory
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_cublas, bytes));
    CHECK_CUDA(cudaMalloc(&d_cublasdx, bytes));
    CHECK_CUDA(cudaMalloc(&d_kernel, bytes));

    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    //Init matrixes
    funcs::matrixInitRandomDeviceOne(d_a);
    funcs::matrixInitRandomDeviceOne(d_b);

    //MM CUDA
    funcs::matrixMultKernel<<<grid, block>>>(d_a, d_b, d_kernel);

    //MM cuBLAS
    funcs::matrixMultcuBLAS(d_a, d_b , d_cublas);

    //MM cuBLASDx
    funcs::infoPrint<GEMM>(NUM_THREADS,NUM_BLOCKS_1D);

    funcs::matrixMultcuBLASDx<GEMM><<<NUM_BLOCKS_1D,NUM_THREADS>>>(d_a, d_b, d_cublasdx);

    funcs::checkMatrixEqualWithinEpsilon<<<NUM_BLOCKS_1D, NUM_THREADS>>>(d_cublas, d_cublasdx, d_result, elements, epsilon);
    funcs::checkPrint(d_result);
    funcs::printMultiplicationDevice(d_a, d_b, d_cublas);
    funcs::printMultiplicationDevice(d_a, d_b, d_cublasdx);


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_kernel);
    cudaFree(d_cublasdx);
    cudaFree(d_cublas);

    cudaFree(d_result);

    return 0;
}