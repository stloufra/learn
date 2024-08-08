#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>
#include <math.h>

//Threads       - executes the instruction
//Warps         - SIMT (lowest schedulable entity) size 32
//Thread blocks - lowest programmable entity
//              - assigned to shader core
//Grids         - mapping to gpu

using realType = float;

void cudaError(cudaError_t err){
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void vecInit(realType *a, int n){

    for (int i = 0; i < n; i++) {
        a[i] = i;
    }
}

void vectorAddHost(realType *a, realType *b, realType *c, int N) {

    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void checkIfSame(realType *a, realType *b, realType *c, int N){
    for (int i = 0; i < N; i++) {
        if(N < 11) {
            std::cout << "c[" << i << "]= " << c[i] << std::endl;
        }
        assert(c[i] == a[i] + b[i]);
    }

    std::cout<< "IT IS GREAT SUCCESS!" << std::endl;
}
void printVector(realType *vec, int N){
    for (int i = 0; i < N; i++) {
            std::cout << "c[" << i << "]= " << vec[i] << std::endl;
    }
    std::cout<< "\n............\n";
}
__global__ void vectorAdd(realType *a, realType *b, realType *c, int N) {

    int tid = (blockIdx.x * blockDim.x) + blockIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }

}

int main() {

    cublasHandle_t handle;
    cublasStatus_t errorCublas;
    cudaError_t errorCuda;

    int N = 1 << 3;

    realType *h_a, *h_b, *h_c, *h_check;
    realType *d_a, *d_b;

    size_t bytes = sizeof(realType) * N;

    printf("Our total size in bites %d \n", (int)bytes);


    //Alocate host memory
    h_a = (realType *) malloc(bytes);
    h_b = (realType *) malloc(bytes);
    h_c = (realType *) malloc(bytes);\
    h_check = (realType *) malloc(bytes);

    //Allocate CUDA memory
    cudaMalloc(&d_a, bytes);
    cudaError(errorCuda);
    cudaMalloc(&d_b, bytes);
    cudaError(errorCuda);

    vecInit(h_a, N);
    vecInit(h_b, N);

    printVector(h_a, N);
    printVector(h_b, N);



    errorCublas = cublasCreate_v2(&handle);

    if (errorCublas != CUBLAS_STATUS_SUCCESS) {
        printf ("Init fail.");
        return EXIT_FAILURE;
    }

    // instead of MemCpy
    // (num of elements, size of type, pointer on host, step on host, pointer on device, step on device)
    errorCublas = cublasSetVector(N, sizeof(realType), h_a, 1,  d_a, 1);
    if (errorCublas != CUBLAS_STATUS_SUCCESS) {
        printf ("Memcop fail.");
        return EXIT_FAILURE;
    }
    errorCublas = cublasSetVector(N, sizeof(realType), h_b, 1,  d_b, 1);
    if (errorCublas != CUBLAS_STATUS_SUCCESS) {
        printf ("Memcop fail.");
        return EXIT_FAILURE;
    }

    const realType scale = 1.f;

    errorCublas =cublasSaxpy(handle, N, &scale, d_a, 1, d_b, 1);
    if (errorCublas != CUBLAS_STATUS_SUCCESS) {
        printf ("Add fail.");
        return EXIT_FAILURE;
    }

    errorCublas = cublasGetVector(N, sizeof(realType), d_b, 1, h_c, 1);
    if (errorCublas != CUBLAS_STATUS_SUCCESS) {
        printf ("Memcop to host fail.");
        return EXIT_FAILURE;
    }


    vectorAddHost(h_a, h_b, h_check, N);

    printVector(h_c, N);
    printVector(h_check, N);


    checkIfSame( h_a, h_b, h_c, N);


    //destruction
    cublasDestroy(handle);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_check);

    cudaFree(d_a);
    cudaFree(d_b);


    return 0;
}