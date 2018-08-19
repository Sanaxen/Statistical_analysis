#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"


int main(int argc, char** argv)
{
	Matrix<dnn_double> A;
	Matrix<dnn_double> B;

	printf("eigenvalues START\n");
	/*
	Eigenvalues
	(  2.86, 10.76) (  2.86,-10.76) ( -0.69,  4.70) ( -0.69, -4.70) -10.46

	Left eigenvectors
	(  0.04,  0.29) (  0.04, -0.29) ( -0.13, -0.33) ( -0.13,  0.33)   0.04
	(  0.62,  0.00) (  0.62,  0.00) (  0.69,  0.00) (  0.69,  0.00)   0.56
	( -0.04, -0.58) ( -0.04,  0.58) ( -0.39, -0.07) ( -0.39,  0.07)  -0.13
	(  0.28,  0.01) (  0.28, -0.01) ( -0.02, -0.19) ( -0.02,  0.19)  -0.80
	( -0.04,  0.34) ( -0.04, -0.34) ( -0.40,  0.22) ( -0.40, -0.22)   0.18

	Right eigenvectors
	(  0.11,  0.17) (  0.11, -0.17) (  0.73,  0.00) (  0.73,  0.00)   0.46
	(  0.41, -0.26) (  0.41,  0.26) ( -0.03, -0.02) ( -0.03,  0.02)   0.34
	(  0.10, -0.51) (  0.10,  0.51) (  0.19, -0.29) (  0.19,  0.29)   0.31
	(  0.40, -0.09) (  0.40,  0.09) ( -0.08, -0.08) ( -0.08,  0.08)  -0.74
	(  0.54,  0.00) (  0.54,  0.00) ( -0.29, -0.49) ( -0.29,  0.49)   0.16
	*/
#if 0
	dnn_double mat[] = {
		-1.01,  3.98,  3.30,  4.43,  7.31,
		0.86,  0.53,  8.26,  4.96, -6.43,
		-4.60, -7.04, -3.89, -7.66, -6.16,
		3.31,  5.29,  8.20, -7.33,  2.47,
		-4.81,  3.55, -1.51,  6.18,  5.58
	};
	A = Matrix<dnn_double>(mat, 5, 5);
#else
	dnn_double mat[] = {
		16,-1,1,2,
		2,12,1,-1,
		1,3,-24,2,
		4,-2,1,20
	};
	A = Matrix<dnn_double>(mat, 4, 4);

#endif

	A.print();

	eigenvalues eig(A);
	printf("info=%d\n", eig.calc(true));

	Matrix<dnn_double> iv = eig.getImageValue();
	Matrix<dnn_double> rv = eig.getRealValue();

	rv.print();
	iv.print();

	for (int i = 0; i < A.m; i++)
	{
		std::vector<Matrix<dnn_double>> vec = eig.getRightVector(i);
		printf("value:%f + %fI\n", rv.v[i], iv.v[i]);
		printf("vector:\n");
		for (int j = 0; j < A.m; j++)
		{
			printf("(%6.2f, %6.2f) ", vec[0](0, j), vec[1](0, j));
		}
		printf("\ncheck=");
		(A*vec[0] - rv.v[i] * vec[0]).print();
	}

	printf("eigenvalues END\n\n");

	return 0;
}