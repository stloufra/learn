#include <cublasdx.hpp>
using namespace cublasdx;


// Naive copy; one thread does all the work
template<class T>
inline __device__ void naive_copy(T* dst, const T* src, unsigned int size) {
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        for (unsigned int i = 0; i < size; ++i) {
            dst[i] = src[i];
        }
    }
}

template<class GEMM>
__global__ void gemm_kernel(GEMM::value_type alpha, GEMM::value_type *a, GEMM::value_type *b, GEMM::value_type beta, GEMM::value_type *c) {
    using value_type = typename GEMM::value_type;
    extern __shared__ value_type smem[];

    value_type* sa = smem;
    value_type* sb = smem + GEMM::a_size;
    value_type* sc = smem + GEMM::a_size + GEMM::b_size;

    // Load data from global to shared memory
    naive_copy(sa, a, GEMM::a_size);
    naive_copy(sb, b, GEMM::b_size);
    naive_copy(sc, c, GEMM::c_size);
    __syncthreads();

    // Execute GEMM
    GEMM().execute(alpha, sa, sb, beta, sc);
    __syncthreads();

    // Store data to global memory
    naive_copy(c, sc, GEMM::c_size);
}

int main() {
    constexpr auto t_mode = cublasdx::transpose_mode::non_transposed;
    using GEMM = decltype(Size<32, 32, 32>()
                          + Precision<double>()
                          + Type<type::real>()
                          + TransposeMode<t_mode, t_mode>()
                          + Function<function::MM>()
                          + SM<860>()
                          + Block()
                          + BlockDim<256>());

    // Allocate managed memory for A, B, C matrices in one go
    value_type* abc;
    auto        size       = GEMM::a_size + GEMM::b_size + GEMM::c_size;
    auto        size_bytes = size * sizeof(value_type);
    cudaMallocManaged(&abc, size_bytes);
    // Generate data
    for (size_t i = 0; i < size; i++) {
        abc[i] = double(i) / size;
    }

    value_type* a = abc;
    value_type* b = abc + GEMM::a_size;
    value_type* c = abc + GEMM::a_size + GEMM::b_size;

    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel<GEMM><<<1, GEMM::block_dim, GEMM::shared_memory_size>>>(1.0, a, b, 1.0, c);
    cudaDeviceSynchronize();

    cudaFree(abc);

    return 0;
}