#ifndef INC_01_ONEMATRIX_CUBLAS_VS_CUBLASDX_FUNC_H
#define INC_01_ONEMATRIX_CUBLAS_VS_CUBLASDX_FUNC_H

#include <time.h>
#include "defs.h"

namespace funcs {

    //----------------------------------------------------------------
    //----------------------INIT FUNCTIONS----------------------------
    //----------------------------------------------------------------

    void matrixInitRandomDevice(realType *A) {

        curandGenerator_t randGen;

        CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));

        CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, (unsigned long long) clock()));

        CHECK_CURAND(curandGenerateUniform(randGen, A, N * N));

        curandDestroyGenerator(randGen);


    }

    void matrixInitRandomDeviceMult(realType *A, int numberOfMat) {

        for (int i = 0; i < numberOfMat; i++) {

            realType *currentMatrix = A + i * N * N;
            funcs::matrixInitRandomDevice(currentMatrix);
        }


    }

    void matrixInitNumberDevice(realType *A, int number) {

        auto bytes = sizeof(realType)*N*N;
        realType *A_h;

        A_h = (realType *) malloc(bytes);

        for (int i = 0; i < N*N; i++){
            A_h[i] = (realType)number;
        }

        CHECK_CUDA(cudaMemcpy(A, A_h, bytes, cudaMemcpyHostToDevice));

        free(A_h);

    }

    void matrixInitNumberDeviceMult(realType *A, int numberOfMat, int number) {

        for (int i = 0; i < numberOfMat; i++) {

            realType *currentMatrix = A + i * N * N;
            funcs::matrixInitNumberDevice(currentMatrix, number);
        }


    }

    void matrixInitSucDevice(realType *A) {

        auto bytes = sizeof(realType)*N*N;
        realType *A_h;

        A_h = (realType *) malloc(bytes);

        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++) {
                A_h[i*N + j] = (realType) i*N + j;
            }
        }

        CHECK_CUDA(cudaMemcpy(A, A_h, bytes, cudaMemcpyHostToDevice));

        free(A_h);

    }

    void matrixInitSucDeviceMult(realType *A, int numberOfMat) {

        for (int i = 0; i < numberOfMat; i++) {

            realType *currentMatrix = A + i * N * N;
            funcs::matrixInitSucDevice(currentMatrix);
        }


    }

    void matrixInitSucRevDevice(realType *A) {

        auto bytes = sizeof(realType)*N*N;
        realType *A_h;

        A_h = (realType *) malloc(bytes);

        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++) {
                A_h[i*N + j] = (realType) N*N - (i*N + j) -1;
            }
        }

        CHECK_CUDA(cudaMemcpy(A, A_h, bytes, cudaMemcpyHostToDevice));

        free(A_h);

    }

    void matrixInitSucRevDeviceMult(realType *A, int numberOfMat) {

        for (int i = 0; i < numberOfMat; i++) {

            realType *currentMatrix = A + i * N * N;
            funcs::matrixInitSucRevDevice(currentMatrix);
        }


    }

    //----------------------------------------------------------------
    //----------------------MATMUL FUNCTIONS--------------------------
    //----------------------------------------------------------------

    void matrixMultcuBLAS(realType *A, realType *B, realType *C) {

        cublasHandle_t handle;

        CHECK_CUBLAS(cublasCreate(&handle));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N));

        cublasDestroy(handle);

    }

    void matrixMultcuBLASMult(realType *A, realType *B, realType *C, int numberOfMat) {
        for (int i = 0; i < numberOfMat; i++) {

            realType *currentMatrixA = A + i * N * N;
            realType *currentMatrixB = B + i * N * N;
            realType *currentMatrixC = C + i * N * N;

            matrixMultcuBLAS(currentMatrixA, currentMatrixB, currentMatrixC);

        }
    }

    inline __device__ void shared_copy(realType *dst, const realType *src, int numElem) {
        if (threadIdx.x == 0) {
            for (unsigned int i = 0; i < numElem; ++i) {
                dst[i] = src[i];
            }
        }
    }

    __global__ void matrixMultKernel(realType *a, realType *b, realType *c) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;


        realType tempSum = 0;
        if ((row < N) && (col < N)) {

            for (
                    int k = 0;
                    k < N; k++) {
                tempSum += a[row * N + k] * b[k * N + col];
            }
        }

        c[row * N + col] =
                tempSum;
    }

    template<class GEM>
    __global__ void matrixMultcuBLASDx(realType *A, realType *B, realType *C) {

        __shared__ realType A_shr[N*N*sizeof(realType)];
        __shared__ realType B_shr[N*N*sizeof(realType)];
        __shared__ realType C_shr[N*N*sizeof(realType)];

        funcs::shared_copy(A_shr, A, N*N);
        funcs::shared_copy(B_shr, B, N*N);

        __syncthreads();

        GEM().execute(1.0, B_shr, A_shr, 0.0, C_shr);

        __syncthreads();

        funcs::shared_copy(C, C_shr, N*N);
    }

    __global__ void childKernel(realType *A, realType *C)
    {
        int idx = threadIdx.x;

        __shared__
        realType A_shr[N * N * sizeof(realType)];

        __shared__
        realType C_shr[N * N * sizeof(realType)];

        funcs::shared_copy(A_shr, A, N * N);

        if(idx < N*N){
            C_shr[idx] = A_shr[idx];
        }

        funcs::shared_copy(C, C_shr, N * N);

    }

    template<class GEM>
    __global__ void matrixMultcuBLASDxMult(realType *A, realType *B, realType *C) {

        if (threadIdx.x == 0) {

            realType *A_cur = A + blockIdx.x * N * N;
            realType *B_cur = B + blockIdx.x * N * N;
            realType *C_cur = C + blockIdx.x * N * N;

            matrixMultcuBLASDx<GEM><<<1, blockDim.x>>>(A_cur, B_cur, C_cur);
#ifdef DEBUG
            printf("My block idx = %d\n", blockIdx.x);
#endif
        }

    }


    __global__ void checkMatrixEqualWithinEpsilon(const float *arr1, const float *arr2, int *result, int size, float epsilon) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx < size) {
            if (fabsf(arr1[idx] - arr2[idx]) > epsilon) {
                atomicExch(result, 0);
            } else {
                atomicExch(result, 1);
            }
#ifdef DEBUG
            printf("Res = %d\n", *result);
#endif
        }
    }

//----------------------------------------------------------------
//---------------------PRINT FUNCTIONS----------------------------
//----------------------------------------------------------------

    template<class GEM>
    void infoPrint(int NUM_THREADS, int NUM_BLOCKS) {
        if (cublasdx::is_complete_blas<GEM>::value) {
            std::cout << "Prepared well-defined cuBLASDX GEMM (M x N x K): "
                      << cublasdx::size_of<GEM>::m << " x "
                      << cublasdx::size_of<GEM>::n << " x "
                      << cublasdx::size_of<GEM>::k << std::endl;
        } else {
            std::cout << "cuBLASDX GEMM is ill-defined.\n";
        }

        std::cout << "Running on " << NUM_BLOCKS << " blocks, each consisting of " << NUM_THREADS
                  << " threads. Total of "
                  << NUM_THREADS * NUM_BLOCKS << " threads.\n\n";

    }

    void checkPrint(int *resultDev) {
        int resultHost = 1;

        CHECK_CUDA(cudaMemcpy(&resultHost, resultDev, sizeof(int), cudaMemcpyDeviceToHost));

        if (resultHost == 1) {
            std::cout << "All elements are within epsilon." << std::endl;
        } else {
            std::cout << "Not all elements are within epsilon." << std::endl;
        }
    }

    void printMatrix(realType *A) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (i != N - 1) {
                    printf("%f ", (float) A[j * N + i]);
                } else {
                    printf("%f", (float) A[j * N + i]);
                }
            }
            printf("\n");
        }
    }

    void printMatrixFromDevice(realType *A) {

        realType *A_host;

        A_host = (realType *) malloc(N * N * sizeof(realType));

        CHECK_CUDA(cudaMemcpy(A_host, A, N * N * sizeof(realType), cudaMemcpyDeviceToHost));

        funcs::printMatrix(A_host);

        free(A_host);
    }

    void printMultiplicationDevice(realType *A, realType *B, realType *C) {
        printf("\n---------------------\n");
        funcs::printMatrixFromDevice(A);
        printf("-------- X ----------\n");
        funcs::printMatrixFromDevice(B);
        printf("-------- = ----------\n");
        funcs::printMatrixFromDevice(C);
        printf("---------------------\n\n");
    }

}

#endif //INC_01_ONEMATRIX_CUBLAS_VS_CUBLASDX_FUNC_H
