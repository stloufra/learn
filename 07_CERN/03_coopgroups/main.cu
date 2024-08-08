#include <iostream>
#include <Eigen/Dense>

#include <cooperative_groups.h>

#define SIZE 3
namespace cg = cooperative_groups;

template<unsigned int Num>
__device__ void tileTry(cg::thread_block_tile<Num>& tile){

    for(int i = 0; i < 16; i+=tile.meta_group_size()*gridDim.x){
        if(tile.thread_rank() == 0) {
            printf("Go to %d\n", i);
        }
    }

}

template <typename T, unsigned int TileSize>
__device__ inline void copyToFromSharedMemoryByTile(  T* dst,
                                                      const T* src,
                                                      int numElements,
                                                      cg::thread_block_tile<TileSize>& tile) {
    int tileId = tile.meta_group_rank();
    int threadId = tile.thread_rank();

    for (int i = threadId; i < numElements; i += tile.num_threads()) {
        dst[i + numElements*tileId] = src[i + numElements*tileId];
    }
}

template <typename T, unsigned int TileSize>
__device__ inline void whichThread(           T* dst,
                                                      int numElements,
                                                      cg::thread_block_tile<TileSize>& tile) {
    int tileId = tile.meta_group_rank();
    int threadId = tile.thread_rank();

    for (int i = threadId; i < numElements; i += tile.num_threads()) {
        dst[i + numElements*tileId] = threadId;
    }
}

template <typename T, unsigned int TileSize>
__device__ inline void whichThreadEigen(      Eigen::Matrix3d& dst,
                                              int numElements,
                                              cg::thread_block_tile<TileSize>& tile) {
    int tileId = tile.meta_group_rank();
    int threadId = tile.thread_rank();


    dst[tileId] = threadId;
}

template <typename T, unsigned int TileSize>
__device__ inline void whichGroup(           T* dst,
                                              int numElements,
                                              cg::thread_block_tile<TileSize>& tile) {
    int tileId = tile.meta_group_rank();
    int threadId = tile.thread_rank();

    for (int i = threadId; i < numElements; i += tile.num_threads()) {
        dst[i + numElements*tileId] = tileId;
    }
}

double ab(double& a,
        double& b){
    return a*b;
}

template <int N, typename T>
__global__ void myKernel(T* A){



    namespace cg = cooperative_groups;

    cg::thread_block block = cg::this_thread_block();

    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);


    __shared__ Eigen::Matrix3d sMat[2];
    __shared__ int lo[2];

    for(int i=0; i < SIZE*SIZE; i++){
        A[i] = 4;
        A[SIZE*SIZE + i] = 5;
    }

    Eigen::Map<Eigen::Matrix3d> gMat1(A, 3, 3);
    Eigen::Map<Eigen::Matrix3d> gMat2(A+SIZE*SIZE, 3, 3);






    whichThread(sMat[0].data(), SIZE*SIZE, tile);

    sMat[1] = gMat1;

    for(int i=0; i < SIZE*SIZE; i++){
    A[i] = sMat[0].data()[i];
    A[SIZE*SIZE + i] = sMat[1].data()[i];
    }

    gMat2 = sMat[0];






    //copyToFromSharedMemoryByTile(sA, A, SIZE*SIZE, tile);
    //whichGroup(sA, SIZE*SIZE, tile);
    //copyToFromSharedMemoryByTile(A, sA, SIZE*SIZE, tile);



    auto tdx = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();

   /* if(tile.thread_rank() == 0) {
        printf("Local start %d\n", tdx);
    }

    tileTry(tile);*/

}






template <typename T>
void printMatrix(T *A) {
    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i < SIZE; i++) {
            if (i != SIZE - 1) {
                printf("%f ", (float)(A[j * SIZE + i]));
            } else {
                printf("%f", (float)(A[j * SIZE + i]));
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    double a = 3;
    double b = 4;

    a = ab(a, b);

    printf("a = %f, b = %f",a,b);

    // Define dimensions
    using realType = double;

    int I = SIZE; // Example value for I
    int J = SIZE; // Example value for J

    // Allocate and initialize data on host and device
    realType h_data[2*I * J];

    for (int i = 0; i < 2*I*J; i++){
        h_data[i] = i;
    }

    printMatrix(h_data);

    realType *d_data;

    cudaMalloc(&d_data, 2*I * J * sizeof(realType));
    cudaMemcpy(d_data, h_data, 2*I * J * sizeof(realType), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    size_t blockDim = 2;
    size_t gridDim = 2;

    // Launch the kernel
    myKernel<4><<<gridDim, blockDim*4>>>(d_data);

    // Copy result back to host
    cudaMemcpy(h_data, d_data, 2*I * J * sizeof(realType), cudaMemcpyDeviceToHost);

    printMatrix(h_data);
    printMatrix((h_data + SIZE*SIZE));


    // Cleanup
    cudaFree(d_data);

    return 0;




}