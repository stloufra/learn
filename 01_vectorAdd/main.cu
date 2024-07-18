#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

//Threads       - executes the instruction
//Warps         - SIMT (lowest schedulable entity) size 32
//Thread blocks - lowest programmable entity
//              - assigned to shader core
//Grids         - mapping to gpu

using realType = float;

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
            std::cout << "c[" << i << "]= " << a[i] << std::endl;
        }
        assert(c[i] == a[i] + b[i]);
    }

    std::cout<< "IT IS GREAT SUCCESS!" << std::endl;
}

__global__ void vectorAdd(realType *a, realType *b, realType *c, int N) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }

}

int main() {


    int N = 1 << 16;

    realType *h_a, *h_b, *h_c;
    realType *d_a, *d_b, *d_c;

    size_t bytes = sizeof(realType) * N;

    int bytesPrint = (int)bytes;

    printf("Our total size in bites %d \n", bytesPrint);

    realType NUM_THREADS = 256;
    realType NUM_BLOCKS = (int) ceil(N / NUM_THREADS); // multiple of 32

    //-----------------
    // managed memory
    //-----------------

    //Alocate host memory
    h_a = (realType *) malloc(bytes);
    h_b = (realType *) malloc(bytes);
    h_c = (realType *) malloc(bytes);

    //Allocate CUDA memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vecInit(h_a, N);
    vecInit(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    checkIfSame( h_a, h_b, h_c, N);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //-----------------
    // unified memory
    //----------------

    int id = cudaGetDevice(&id);

    realType *u_a, *u_b, *u_c;

    cudaMallocManaged(&u_a, bytes);
    cudaMallocManaged(&u_b, bytes);
    cudaMallocManaged(&u_c, bytes);

    vecInit(u_a, N);
    vecInit(u_b, N);

    cudaMemPrefetchAsync(u_a, bytes, id);
    cudaMemPrefetchAsync(u_b, bytes, id);

    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(h_a, h_b, h_c, N);

    // to wait for gpu to finish !!!
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(u_c, bytes, cudaCpuDeviceId);

    checkIfSame( h_a, h_b, h_c, N);

    cudaDeviceSynchronize();


    cudaFree(u_a);
    cudaFree(u_b);
    cudaFree(u_c);

    return 0;
}