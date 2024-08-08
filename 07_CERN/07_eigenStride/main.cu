//
// Created by stloufra on 8/5/24.
//

#include <Eigen/Dense>
#include <iostream>

constexpr int NUM_MATRICES = 8;
constexpr int MATRIX_SIZE = 8;
constexpr int TOTAL_ELEMENTS = NUM_MATRICES * MATRIX_SIZE * MATRIX_SIZE;

int main() {

#define NUM_OF_MAT 8

    using Matrix = Eigen::Matrix<double, 3, 2>;
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using Map = Eigen::Map<Matrix, 0, Stride>;


    size_t num_bytes = sizeof(Matrix) * NUM_OF_MAT;


    double* MEM;
    MEM = (double *) malloc(num_bytes);

    for(int i = 0; i < 6; i++){
        for(int j = 0; j < NUM_OF_MAT; j++) {
            MEM[(i*NUM_OF_MAT) + j] = i + 10*j;
        }
    }

    for(int i = 0; i < 3*2*NUM_OF_MAT; i++){
        printf("MEM[%d] = %f\n", i, MEM[i]);
    }




    // Create a strided map
    Stride stride(NUM_OF_MAT, 2*NUM_OF_MAT);
    Map strided_map1(MEM, stride);
    Map strided_map2(MEM + 2, stride);

    std::cout << "Strided map 1:\n" << strided_map1 << "\n\n";
    std::cout << "Strided map 2:\n" << strided_map2 << "\n\n";

    strided_map1(0,0) = 250;

    std::cout << "Strided map:\n" << strided_map1 << "\n\n";

    using Matrix3x4 = Eigen::Matrix<double, 3, 4>;

    Matrix3x4 K;

    std::cout<<K;


    free(MEM);

    //__shared__ realType A[SHMEM_SIZE];

    return 0;
}
