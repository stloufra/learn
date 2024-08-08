#ifndef DEFS
#define DEFS

#include <cuda_runtime.h>
#include <cublasdx.hpp>
#include <cublas_v2.h>
#include <curand.h>

#define N 3

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

using realType = float; //leave for now cuBLAS defined for float

using GEMM = decltype(cublasdx::Size<N, N, N>()
                      + cublasdx::Precision<realType>()
                      + cublasdx::Type<cublasdx::type::real>()
                      + cublasdx::TransposeMode<cublasdx::transpose_mode::non_transposed, cublasdx::transpose_mode::non_transposed>()
                      + cublasdx::Function<cublasdx::function::MM>()
                      + cublasdx::SM<860>()
                      //+ cublasdx::LeadingDimension<N,N,N>()
                      + cublasdx::Block()
                      //+ cublasdx::BlockDim<32>()
                              );

#endif