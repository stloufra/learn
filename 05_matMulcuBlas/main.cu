
#include <cublas_v2.h>
#include <curand.h>
#include <time.h>
#include <iostream>
#include <cassert>
#include <cublasdx.hpp>

using realType = float;

// Function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to check CUBLAS errors
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Function to check CURAND errors
#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "CURAND error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


void matrixInitRandomDeviceOne(realType *A, int N) {

    curandGenerator_t randGen;

    CHECK_CURAND(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT));

    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen, (unsigned long long) clock()));

    CHECK_CURAND(curandGenerateUniform(randGen, A, N * N));

    curandDestroyGenerator(randGen);


}

void matrixInitRandomDevice(realType *A, int N, int NumberOfMat) {
    for (int i = 0; i < NumberOfMat; i++) {

        realType *currentMatrix = A + i * N * N;
        matrixInitRandomDeviceOne(currentMatrix, N);

    }
}

void makeMatrixPossitiveDefiniteOne(realType *A, int N) {

    size_t bytes = sizeof(realType) * N * N;

    realType *A_clone, *A_final;

    CHECK_CUDA(cudaMalloc(&A_clone, bytes));
    CHECK_CUDA(cudaMalloc(&A_final, bytes));

    CHECK_CUDA(cudaMemcpy(A_clone, A, bytes, cudaMemcpyDeviceToDevice));

    cublasHandle_t handle;

    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A_clone, N, A, N, &beta, A_final, N));

    cublasDestroy(handle);

    CHECK_CUDA(cudaMemcpy(A, A_final, bytes, cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(A_clone));
    CHECK_CUDA(cudaFree(A_final));


}

void makeMatrixPossitiveDefinite(realType *A, int N, int NumberOfMat) {
    for (int i = 0; i < NumberOfMat; i++) {

        realType *currentMatrix = A + i * N * N;
        makeMatrixPossitiveDefiniteOne(currentMatrix, N);

    }
}

void matrixMultCheckOne(realType *A, realType *B, realType *C, int N) {

    cublasHandle_t handle;

    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, B, N, A, N, &beta, C, N));

    cublasDestroy(handle);

}

void matrixMultCheck(realType *A, realType *B, realType *C, int N, int NumberOfMat) {
    for (int i = 0; i < NumberOfMat; i++) {

        realType *currentMatrixA = A + i * N * N;
        realType *currentMatrixB = B + i * N * N;
        realType *currentMatrixC = C + i * N * N;
        matrixMultCheckOne(currentMatrixB, currentMatrixA, currentMatrixC, N);

    }
}

void printMatrix(realType *A, int N) {

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (i != N - 1) {
                printf("%f, ", (float) A[j * N + i]);
            } else {
                printf("%f", (float) A[j * N + i]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixFromDevice(realType *A, int N) {

    realType *A_host;

    A_host = (realType *) malloc(N * N * sizeof(realType));

    CHECK_CUDA(cudaMemcpy(A_host, A, N * N * sizeof(realType), cudaMemcpyDeviceToHost));

    printMatrix(A_host, N);

    free(A_host);
}

inline __device__ void shared_copy(realType *dst, const realType *src, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template<class GEMM>
__global__ void
matrixMultShared(realType *A, realType *B, realType *C, int N, int numberOfEL, int numThreadsCUBLAS) {

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const size_t sizeOfMatrix = N * N;


    if (index % 9 == 0 && index < (numberOfEL)) {

        auto indXofMatrix = (unsigned int) (ceilf(index / 9));

        realType *A_cur = A + indXofMatrix * sizeOfMatrix;
        realType *B_cur = B + indXofMatrix * sizeOfMatrix;
        realType *C_cur = C + indXofMatrix * sizeOfMatrix;

        __shared__ realType A_shr[GEMM::a_size*sizeof(realType)];
        __shared__ realType B_shr[GEMM::b_size*sizeof(realType)];
        __shared__ realType C_shr[GEMM::c_size*sizeof(realType)];

        shared_copy(A_shr, A_cur, sizeOfMatrix);
        shared_copy(B_shr, B_cur, sizeOfMatrix);

        __syncthreads();

        GEMM().execute(1.0, B_shr, A_shr, 1.0, C_shr);

        __syncthreads();


        shared_copy(C_cur, C_shr, sizeOfMatrix);

    }

}

__global__ void checkEqualWithinEpsilon(const float *arr1, const float *arr2, int *result, int size, float epsilon) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < size) {
        if (fabsf(arr1[idx] - arr2[idx]) > epsilon) {
            atomicExch(result, 0);
        }
        else{
            atomicExch(result, 1);
        }
    }
}


int main() {

    unsigned int const N = 3;
    unsigned int const numberOfMatrixes = 10;
    const auto totalNumberOfElements = N * N * numberOfMatrixes;
    size_t bytes = sizeof(realType) * totalNumberOfElements;

    constexpr auto t_mode = cublasdx::transpose_mode::non_transposed;

    using GEMM = decltype(cublasdx::Size<N, N, N>()
                          + cublasdx::Precision<realType>()
                          + cublasdx::Type<cublasdx::type::real>()
                          + cublasdx::TransposeMode<t_mode, t_mode>()
                          + cublasdx::Function<cublasdx::function::MM>()
                          + cublasdx::SM<860>()
                          + cublasdx::Block());

    if(cublasdx::is_complete_blas<GEMM>::value)
        std::cout << "GEMM (M x N x K): "
                  << cublasdx::size_of<GEMM>::m << " x "
                  << cublasdx::size_of<GEMM>::n << " x "
                  << cublasdx::size_of<GEMM>::k << std::endl;

    int *d_result;
    realType *d_a, *d_b, *d_c, *d_check;
    realType *h_c, *h_check;

    //Alocate host memory
    h_c = (realType *) malloc(bytes);
    h_check = (realType *) malloc(bytes);

    //Allocate CUDA memory
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_check, bytes));

    CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

    //Init Matrixes
    matrixInitRandomDevice(d_a, N, numberOfMatrixes);
    matrixInitRandomDevice(d_b, N, numberOfMatrixes);

    //Make it possitive definite by mutliplying with transpose-self
    makeMatrixPossitiveDefinite(d_a, N, numberOfMatrixes);
    makeMatrixPossitiveDefinite(d_b, N, numberOfMatrixes);


    //----------------- GEMM KERNEL MAGIC------------
    unsigned int numberThreadsCublas = 3;
    unsigned int NUM_THREADS = 32;
    unsigned int a = (int) ceil(numberThreadsCublas * totalNumberOfElements / NUM_THREADS);
    unsigned int NUM_BLOCKS = a + 1;

    std::cout << "Running on " << NUM_BLOCKS << " blocks, each consisting of " << NUM_THREADS << " threads. Total of "
              << NUM_THREADS * NUM_BLOCKS << " threads.\n";

    matrixMultShared<GEMM><<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N, numberOfMatrixes, numberThreadsCublas);
    //-----------------------------------------------

    matrixMultCheck(d_a, d_b, d_c, N, numberOfMatrixes);


    const float epsilon = 0.0001f;
    checkEqualWithinEpsilon<<<NUM_BLOCKS, NUM_THREADS>>>(d_c, d_check, d_result, totalNumberOfElements, epsilon);


    int h_result = 1;

    CHECK_CUDA( cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_result == 0) {
        std::cout << "Not all elements are within epsilon." << std::endl;
    } else {
        std::cout << "All elements are within epsilon." << std::endl;
    }




    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_check);

    free(h_c);
    free(h_check);


    return 0;
}