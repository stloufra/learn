#include <iostream>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <cooperative_groups.h>
#include <cublasdx.hpp>
#include <cuda_runtime.h>

#define NUM_OF_MAT 150
#define MATRIX_PRINT 149

using GEMMABeqC =
decltype (cublasdx::Size<3, 3, 3>()+
cublasdx::Precision<double>()+
cublasdx::Type<cublasdx::type::real>()+
cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed, cublasdx::transpose_mode::non_transposed>()+
cublasdx::Function<cublasdx::function::MM>()+
cublasdx::SM<860>()+
cublasdx::LeadingDimension<3, 3, 3>()+
cublasdx::Block()+
cublasdx::BlockDim<32>());


using GEMMABteqC =
decltype (cublasdx::Size<3, 3, 3>()+
cublasdx::Precision<double>()+
cublasdx::Type<cublasdx::type::real>()+
cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed, cublasdx::transpose_mode::transposed>()+
cublasdx::Function<cublasdx::function::MM>()+
cublasdx::SM<860>()+
cublasdx::LeadingDimension<3, 3, 3>()+
cublasdx::Block()+
cublasdx::BlockDim<32>());

// Function to check CUDA errors
#define CHECK_CUDA(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
                exit(EXIT_FAILURE); \
            } \
        } while (0)

template<typename T>
__host__ __device__ void fillMatrixVal(T &A, double a) {

    auto N = T::ColsAtCompileTime;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            A(i, j) = ((double) j + 0.523f) * (13.f / 18.f) * ((double) i + 0.87f) + a / 25.13f;
        }
    }
}


template<typename T>
__host__ __device__ void fillMatrixVal2(T &A, double a) {

    auto N = T::ColsAtCompileTime;

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            A(i, j) = ((double) j + 0.923f) * (17.f / 11.f) * ((double) i + 0.31f) + a / 12.13f;
        }
    }
}


template<typename MATRIX, typename TILE>
__device__ inline void ABAteqC(MATRIX const &A, MATRIX const &B, MATRIX &C, MATRIX &holder, TILE &tile) {

    auto N = MATRIX::ColsAtCompileTime;

    auto idx = tile.thread_rank();
    double tmp;
    unsigned int row;
    unsigned int col;

    for (auto i = idx; i < N * N; i += tile.num_threads()) {
        row = i / N;
        col = i % N;
        tmp = 0;

        for (unsigned int m = 0; m < N; m++) {
            for (unsigned int k = 0; k < N; k++) {
                tmp += A(row, k) * B(k, m) * A(col, m);
            }
        }
        holder(row, col) = tmp;
    }
    tile.sync();
    C = holder.eval();
    tile.sync();
}

template<typename MATRIX, typename TILE>
__device__ inline void jacobiMult(MATRIX const &A, MATRIX const &B, MATRIX &C, MATRIX &holder, TILE &tile) {
    ABAteqC(A, B, C, holder, tile);
}

template<typename MATRIX, typename TILE, typename NAME>
__device__ void printFromDevice(MATRIX A, TILE tile, NAME name) {
    auto const N = MATRIX::ColsAtCompileTime;
    auto idx = tile.thread_rank();

    if (idx == 0 && tile.meta_group_rank() == 0) {
        if (N == 3) {
            printf("MATRIX %s IS:{[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]\n}", name, A(0, 0), A(0, 1), A(0, 2), A(1, 0),
                   A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
        }
        if (N == 2) {
            printf("MATRIX %s IS:{[%f,%f],[%f,%f]\n}", name, A(0, 0), A(0, 1), A(1, 0), A(1, 1));
        }
    }

}

template<typename TILE, typename NAME>
__device__ void printFromDevicePointer(double *A, TILE tile, NAME name) {
    auto idx = tile.thread_rank();
    if (idx == 0 && tile.meta_group_rank() == 0) {
        printf("MATRIX %s IS:{[%f,%f,%f],[%f,%f,%f],[%f,%f,%f]\n}", name, A[0], A[1], A[2], A[3],A[4], A[5], A[6], A[7],A[8]);
            printf("\n");
        }
    }


template<typename MATRIX>
__global__ void myKERNEL(MATRIX A,
                         MATRIX B) {


    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);

    auto tileId = tile.meta_group_rank() + blockIdx.x * tile.meta_group_size();

#ifdef DEBUG
    printFromDevice(A[MATRIX_PRINT], tile, "A");
    printFromDevice(B[MATRIX_PRINT], tile, "B");

#endif
    __shared__ Eigen::Matrix3d A_shr[NUM_OF_MAT];
    __shared__ Eigen::Matrix3d B_shr[NUM_OF_MAT];
    __shared__ Eigen::Matrix3d holder[NUM_OF_MAT];

    for (auto idx = tileId; idx < NUM_OF_MAT; idx += tile.meta_group_size() * gridDim.x) {
        A_shr[idx] = A[idx];
        B_shr[idx] = B[idx];

        jacobiMult(A_shr[idx], B_shr[idx], B_shr[idx], holder[idx], tile);
        B[idx] = B_shr[idx];
    }

    block.sync();
#ifdef DEBUG
    printFromDevice(B[MATRIX_PRINT], tile, "C after");
    if(block.thread_rank()==0) printf("\n");
#endif
}

template<typename MATRIX>
__global__ void cublasdxKERNEL(MATRIX A,
                               MATRIX B) {

    auto const N = 3;

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<4> tile = cg::tiled_partition<4>(block);

    auto tileId = tile.meta_group_rank() + blockIdx.x * tile.meta_group_size();

#ifdef DEBUG
    printFromDevice(A[MATRIX_PRINT], tile, "A");
    printFromDevice(B[MATRIX_PRINT], tile, "B");

#endif

    __shared__ Eigen::Matrix3d A_shr[NUM_OF_MAT];
    __shared__ Eigen::Matrix3d B_shr[NUM_OF_MAT];
    __shared__ Eigen::Matrix3d holder[NUM_OF_MAT];



    for (auto idx = tileId; idx < NUM_OF_MAT; idx += tile.meta_group_size() * gridDim.x) {

        A_shr[idx] = A[idx];
        B_shr[idx] = B[idx];
    }


    for(int i = 0; i < tile.meta_group_size(); i++) {
        for (auto j = i; j < NUM_OF_MAT; j += gridDim.x * tile.meta_group_size()){
            auto idx = blockIdx.x * tile.meta_group_size() + j;
            GEMMABeqC().execute(1.0, A_shr[idx].data(), B_shr[idx].data(), 0.0, B_shr[idx].data());
            GEMMABteqC().execute(1.0, B_shr[idx].data(), A_shr[idx].data(), 0.0, B_shr[idx].data());
        }
    }


    for (auto idx = tileId; idx < NUM_OF_MAT; idx += tile.meta_group_size() * gridDim.x) {
        B[idx] = B_shr[idx];
    }


    block.sync();
#ifdef DEBUG
    printFromDevice(B[MATRIX_PRINT], tile, "C after");
    if(block.thread_rank()==0) printf("\n");
#endif
}


int main() {

    using Matrix = Eigen::Matrix3d;


    Matrix A[NUM_OF_MAT];
    Matrix B[NUM_OF_MAT];
    Matrix C[NUM_OF_MAT];
    Matrix Ccheck[NUM_OF_MAT];
    Matrix Cmy[NUM_OF_MAT];

    Matrix Ad[NUM_OF_MAT];
    Matrix Bd[NUM_OF_MAT];

    Matrix AdCU[NUM_OF_MAT];
    Matrix BdCU[NUM_OF_MAT];


    size_t num_bytes = sizeof(Matrix) * NUM_OF_MAT;
    size_t num_bytes_copy = sizeof(double) * Matrix::ColsAtCompileTime * Matrix::ColsAtCompileTime;

    CHECK_CUDA(cudaMalloc((void **) &Ad, num_bytes));
    CHECK_CUDA(cudaMalloc((void **) &Bd, num_bytes));


    for (int i = 0; i < NUM_OF_MAT; i++) {
        fillMatrixVal(A[i], (double) i);
        fillMatrixVal2(B[i], (double) i);
        CHECK_CUDA(cudaMemcpyAsync(Ad[i].data(), A[i].data(), num_bytes_copy, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(Bd[i].data(), B[i].data(), num_bytes_copy, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(AdCU[i].data(), A[i].data(), num_bytes_copy, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpyAsync(BdCU[i].data(), B[i].data(), num_bytes_copy, cudaMemcpyHostToDevice));
        Ccheck[i] = A[i] * B[i]* A[i].transpose();
    }

    std::cout << "TEST\n\n";
    std::cout << A[MATRIX_PRINT] << "\nX\n" << B[MATRIX_PRINT] << "\nX\n" << A[MATRIX_PRINT].transpose() << "\n=\n" << Ccheck[MATRIX_PRINT] << "\n\n";

    auto startMY = std::chrono::high_resolution_clock::now();
    myKERNEL<<<10, 32>>>(Ad, Bd);
    auto endMY = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaGetLastError());

    auto startCU = std::chrono::high_resolution_clock::now();
    cublasdxKERNEL<<<10, 32>>>(AdCU, BdCU);
    auto endCU = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_OF_MAT; i++) {
        CHECK_CUDA(cudaMemcpy(Cmy[i].data(), Bd[i].data(), num_bytes_copy, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(C[i].data(), BdCU[i].data(), num_bytes_copy, cudaMemcpyDeviceToHost));
    }

    //std::cout << "MY\n\n";
    //std::cout << A[MATRIX_PRINT] << "\nX\n" << B[MATRIX_PRINT] << "\nX\n" << A[MATRIX_PRINT].transpose() << "\n=\n"
    //<< Cmy[MATRIX_PRINT] << "\n\n";

    std::cout << "CUBLAS\n\n";
    std::cout << A[MATRIX_PRINT] << "\nX\n" << B[MATRIX_PRINT] << "\nX\n" << A[MATRIX_PRINT].transpose() << "\n=\n"
              << C[MATRIX_PRINT] << "\n\n";

    std::chrono::duration<double> durationMY = endMY - startMY;
    std::chrono::duration<double> durationCU = endCU - startCU;

    std::cout<< "TIME TAKEN BY MY: "<< durationMY.count() << "\n";
    std::cout<< "TIME TAKEN BY CUBLAS: " << durationCU.count() << "\n";

    for (int i = 0; i < NUM_OF_MAT; i++) {
        auto diff = (C[i] - Cmy[i]);
        auto err = diff.norm();
        //if (err > 0.00001) printf("HERE I AM WRONG! %d\n", i);
    }

    cudaFree((void **) &Ad);
    cudaFree((void **) &Bd);
    cudaFree((void **) &AdCU);
    cudaFree((void **) &BdCU);
    return 0;


}