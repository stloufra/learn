#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>


using realType = int;

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

__global__ void vectorAdd(realType *a, realType *b, realType *c, int N) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }

}

int main() {

    int N = 1<<20;

    realType *h_a, *h_b, *h_c;
    realType *d_a, *d_b, *d_c;

    size_t bytes = sizeof(realType) * N;

    int bytesPrint = (int)bytes;

    printf("Our total size in bites %d \n", bytesPrint);

    realType NUM_THREADS = 256;
    realType NUM_BLOCKS = (int) ceil(N / NUM_THREADS); // multiple of 32

    h_a = (realType *) malloc(bytes);
    h_b = (realType *) malloc(bytes);
    h_c = (realType *) malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vecInit(h_a, N);
    vecInit(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddHost(h_a, h_b, h_c, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> durationMY = cpu_end - cpu_start;
    std::chrono::duration<double> durationCU = gpu_end - gpu_start;

    std::cout<< "TIME TAKEN BY CPU: "<< durationMY.count() << "\n";
    std::cout<< "TIME TAKEN BY GPU: " << durationCU.count() << "\n";

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}