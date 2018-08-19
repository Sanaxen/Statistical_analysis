#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"


int main(int argc, char** argv)
{
	printf("linear_equation START\n");
	Matrix<dnn_double> A(3,3);
	Matrix<dnn_double> B(3,1);

	A(0, 0) = 1;
	A(1, 0) = 3;
	A(2, 0) = 1;

	A(0, 1) = 1;
	A(1, 1) = 1;
	A(2, 1) = 2;

	A(0, 2) = 1;
	A(1, 2) = 3;
	A(2, 2) = -5;

	B(0, 0) = 1;
	B(1, 0) = 5;
	B(2, 0) = 10;

	//A.print("A");
	//B.print("B");

	linear_equation lneq;
	lneq.solv(A, B);
	Matrix<dnn_double>x = lneq.x;
	(A*x - B).print("--");

	B = lneq.inv(A);
	B.print();
	printf("linear_equation END\n\n");

	return 0;
}