#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"


int main(int argc, char** argv)
{
	printf("linear_east_square START\n");

	std::mt19937 mt(1234);
	std::uniform_real_distribution<> rand01(-1.0, 1.0);

	Matrix<dnn_double> X(3, 1);
	for (int j = 0; j < 3; j++)
	{
		X.v[j] = j;
	}

	Matrix<dnn_double> A(5, 10);
	Matrix<dnn_double> B(10, 1);
	B = B.zeros(10, 1);
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			A(i, j) = rand01(mt);
		}
		dnn_double y = 0.0;
			
		for (int j = 0; j < 10; j++) y += X(j, 0)*A(i, j);

		B(i, 0) = y + rand01(mt);
	}
	for (int i = 5; i < 10; i++)
	{
		B(i, 0) = B(4, 0) + rand01(mt);
	}

	A.print_e("A");
	B.print_e("B");
	linear_east_square les;

	les.fit(A, B);
	les.coef.print("coef");

	printf("eps:%f\n", SumAll(Pow(A*les.coef - B, dnn_double(2))));

	les.fit2(A, B);
	les.coef.print("coef");
	printf("eps:%f\n", SumAll(Pow(A*les.coef - B, dnn_double(2))));
	printf("linear_east_square END\n\n");

	return 0;
}