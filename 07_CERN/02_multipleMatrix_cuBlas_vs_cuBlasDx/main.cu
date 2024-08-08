#include <iostream>

#include "func.h"
#include "defs.h"

#define DEBUG

int main() {

    size_t elements = N * N;
    int numberOfMatrixes = 1000;
    size_t bytes = elements * sizeof(realType) * numberOfMatrixes;

    int *d_result;
    realType *d_a, *d_b, *d_cublas, *d_cublasdx;

    unsigned int NUM_THREADS = 32; // needs to be enough for one matrix for now! TODO: try defining block
    unsigned int NUM_BLOCKS = numberOfMatrixes;
    unsigned int NUM_BLOCKS_CHECK = (int) ceil(elements * numberOfMatrixes / NUM_THREADS + 1);

    realType epsilon = 0.0001;

    //Allocate CUDA memory
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_cublas, bytes));
    CHECK_CUDA(cudaMalloc(&d_cublasdx, bytes));

    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    //Init matrixes
    funcs::matrixInitRandomDeviceMult(d_a, numberOfMatrixes);
    funcs::matrixInitRandomDeviceMult(d_b, numberOfMatrixes);

    //funcs::matrixInitSucDeviceMult(d_a, numberOfMatrixes);
    //funcs::matrixInitSucRevDeviceMult(d_b, numberOfMatrixes);

    funcs::checkMatrixEqualWithinEpsilon<<<NUM_BLOCKS_CHECK, NUM_THREADS>>>(d_a, d_a, d_result, elements*numberOfMatrixes, epsilon);
    funcs::checkPrint(d_result);


    //MM cuBLAS
    funcs::matrixMultcuBLASMult(d_a, d_b, d_cublas, numberOfMatrixes);

    funcs::printMultiplicationDevice(d_a, d_b, d_cublas);

    //MM cuBLASDx
    funcs::infoPrint<GEMM>(NUM_THREADS, NUM_BLOCKS);
    funcs::matrixMultcuBLASDxMult<GEMM><<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_cublasdx);

    funcs::printMultiplicationDevice(d_a, d_b, d_cublasdx);

    funcs::checkMatrixEqualWithinEpsilon<<<NUM_BLOCKS_CHECK, NUM_THREADS>>>(d_cublas, d_cublasdx, d_result, elements *
                                                                                                            numberOfMatrixes, epsilon);
    funcs::checkPrint(d_result);




    cudaFree(d_a);
    cudaFree(d_b);

    cudaFree(d_cublasdx);
    cudaFree(d_cublas);

    cudaFree(d_result);

    return 0;
}