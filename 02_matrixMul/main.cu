#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

//Threads       - executes the instruction
//Warps         - SIMT (lowest schedulable entity) size 32
//Thread blocks - lowest programmable entity
//              - assignet to shader core
//Grids         - mapping to gpu

//    T(x,y),B(x,y)            0        1    .   2       3
//    0                    0,0;0,0           .
//    1                                      .
//      .....................................................................................
//    2                                      .0,0;1,1
//    3                                      .

//Row = blockIdx.y*blockDim.y + threadIdx.y
//Col = blockIdx.x*blockDim.x + threadIdx.x

//scratch memory | user managed L1 cache | private per-thread block

using realType = int;



void matrixInit(realType *a, int n) {

    for (int i = 0; i < n; i++) { //radky
        for (int j = 0; j < n; j++) { //sloupce
            a[i * n + j] = (realType) j;
        }
    }
}

void matrixMultHost(realType *a, realType *b, realType *ab, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int R = 0; R < N; R++) {
                ab[i * N + j] += a[i * N + R] * b[R * N + j];
            }
        }
    }
}

void printMatix(realType *A, int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j != N - 1) {
                printf("%d, ", A[i * N + j]);
            } else {
                printf("%d", A[i * N + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void checkIfSame(realType *a, realType *b, int N) {



    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert( a[i * N + j] == b[i * N + j] );
#ifdef UNMUTED
            printf("%d  = %d \n", a[i * N + j], b[i * N + j] );
#endif
        }
    }


    std::cout << "IT IS GREAT SUCCESS!" << std::endl;
}

__global__ void matrixMult(realType *a, realType *b, realType *c, int N) {


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    realType tempSum = 0;
    if ((row < N) && (col < N)) {

        for (int k = 0; k < N; k++) {
            tempSum += a[row * N + k] * b[k * N + col];
        }
    }

    c[row * N + col] = tempSum;
}


int main() {


    int N = 3 <<2; // shift operator x * (2^y)


    size_t bytes = sizeof(realType) * N * N;

    printf("Our size in bites %d \n", bytes);

    int BLOCK_SIZE = 16; //16*16=256
    int GRID_SIZE = (int) ceil((float)N / BLOCK_SIZE); // multiple of 32

    std::cout << "GRID_SIZE: " << GRID_SIZE << std::endl;

    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    //-----------------
    // managed memory
    //-----------------
    realType *h_a, *h_b, *h_c, *h_d;
    realType *d_a, *d_b, *d_c;

    //Alocate host memory
    h_a = (realType *) malloc(bytes);
    h_b = (realType *) malloc(bytes);
    h_c = (realType *) malloc(bytes);
    h_d = (realType *) malloc(bytes);

    //Allocate CUDA memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrixInit(h_a, N);
    matrixInit(h_b, N);


    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    matrixMult <<<grid, block>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();


    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    matrixMultHost(h_a, h_b, h_d, N);

    checkIfSame(h_d, h_c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_d);
    free(h_c);
    free(h_b);
    free(h_a);



    return 0;
}

