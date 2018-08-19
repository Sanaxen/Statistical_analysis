#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"


int main(int argc, char** argv)
{
	Matrix<dnn_double> A;
	Matrix<dnn_double> B;
	printf("linear_east_square START\n");

	{
#if 10
		dnn_double a[] = {
			1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
			-7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
			-4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
			4.53,  3.83, -6.64,  2.06, -2.47,  4.70
		};
		dnn_double b[] = {
			8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
			9.35, -4.43, -0.70, -0.26, -7.36, -2.52
		};
		Matrix<dnn_double> A(a, 4, 6);
		Matrix<dnn_double> B(b, 2, 6);

		A = A.transpose(A);
		B = B.transpose(B);
#else
		dnn_double a[] = {
			-0.57, - 1.28, - 0.39,   0.25,
			- 1.93,   1.08, - 0.31, - 2.14,
			2.30,   0.24,   0.40, - 0.35,
			- 1.93,   0.64, - 0.66,   0.08,
			0.15,   0.30,   0.15, - 2.13,
			- 0.02,   1.03, - 1.43,   0.50 };
		dnn_double b[] = {
			-2.67,
			- 0.55,
			3.34,
			- 0.77,
			0.48,
			4.10
		};
		Matrix<dnn_double> A(a, 6, 4);
		Matrix<dnn_double> B(b, 6, 1);

		//dnn_double a[] = {
		//	1,6,2,
		//	1,-2,-8,
		//	1,-2,4,
		//	1,6,14
		//};
		//dnn_double b[] = {
		//	96,
		//	192,
		//	192,
		//	-96
		//};
		//Matrix<dnn_double> A(a, 4, 3);
		//Matrix<dnn_double> B(b, 4, 1);
#endif
		A.print("A");
		B.print("B");
		linear_east_square les;

		printf("error=%d\n",les.fit(A, B));
		les.x.print("x");

		printf("error=%d\n", les.fit2(A, B));
		les.x.print("x");
	}
	printf("linear_east_square END\n\n");
	return 0;
}