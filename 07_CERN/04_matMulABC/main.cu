#include <iostream>
#include <Eigen/Dense>


#define N 3

template<typename T>
void printMatrix(T A) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (i != N - 1) {
                printf("%f ", (float) (A(i, j)));
            } else {
                printf("%f", (float) (A(i, j)));
            }
        }
        printf("\n");
    }
    printf("\n");
}

template<typename T>
void fillMatrix(T &A) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {

            if (i == 2) {
                A(i, j) = i * j + j / (i + 1);
            } else {
                A(i, j) = i * j - j / (i + 1);
            }
        }
    }
}

template<typename T, typename fT>
void fillMatrixVal(T &A, fT a) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {


            A(i, j) = a;

        }
    }
}

template<typename T>
void myMulABAteqC(T &A, T &B, T &C) {

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {

            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                for (int m = 0; m < N; m++) {
                    sum += A(i,k)*B(k,m)*A(j,m);

                }
            }
            //std::cout<<"sum"<< sum << "\n";
        C(i,j) = sum;


        }
    }
}

template <typename D1>
inline constexpr void printEig(Eigen::DenseBase<D1> const& A) {
using M1 = Eigen::DenseBase<D1>;


}


int main() {

    Eigen::Matrix3d A;
    Eigen::Matrix3d B;
    Eigen::Matrix3d C;
    Eigen::Matrix3d Cmy;

    fillMatrix(A);
    fillMatrixVal(B, 2);


    B = A + B;

    A = A + B / 2;

    C = A * B * A.transpose();

    std::cout << A << "\nX\n" << B << "\nX\n" << A.transpose() << "\n=\n" << C << "\n\n";

    myMulABAteqC(A,B,A);


    std::cout << A << "\nX\n" << B << "\nX\n" << A.transpose() << "\n=\n" << Cmy << "\n\n";




    return 0;


}