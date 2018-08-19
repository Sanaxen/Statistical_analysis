#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"


int main(int argc, char** argv)
{
	Matrix<dnn_double> A;
	Matrix<dnn_double> B;
	dnn_double mat[]={
		-1.01,  3.98,  3.30,  4.43,  7.31,
		0.86,  0.53,  8.26,  4.96, -6.43,
		-4.60, -7.04, -3.89, -7.66, -6.16,
		3.31,  5.29,  8.20, -7.33,  2.47,
		-4.81,  3.55, -1.51,  6.18,  5.58
	};
	A = Matrix<dnn_double>(mat, 5, 5);

	printf("svd_decomposition START\n");
	svd_decomposition<dnn_double> svd(A);

	svd.Sigma.print("ƒ°");
	Matrix<dnn_double> tmp = A - svd.U*svd.Sigma*svd.V.transpose();

	tmp.print();
	printf("svd_decomposition END\n\n");
	return 0;
}