#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/statistical/LinearRegression.h"
//#include "../../include/util/mathutil.h"
#include "../../include/util/csvreader.h"


int main(int argc, char** argv)
{
	printf("multiple_regression START\n");
	std::string csvfile("sample.csv");

	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
		}
	}


	FILE* fp = fopen(csvfile.c_str(), "r");
	if (fp == NULL)
	{
		dnn_double a[] = {
			4,2,3,5,4,
			4,3,3,3,4,
			4,1,2,4,4,
			4,1,3,5,3,
			5,2,2,5,5,
			4,4,1,5,4,
			4,2,4,4,4,
			3,4,3,4,3,
			3,2,1,2,3,
			3,5,1,2,4,
			4,2,2,5,5,
			5,4,3,5,4,
			4,2,4,5,4,
			4,4,3,5,5,
			3,2,2,5,3,
			5,2,1,4,5,
			4,2,2,4,4
		};
		Matrix<dnn_double> T(a, 17, 5);
		T.print_csv("sample.csv");
	}

	CSVReader csv1(csvfile, ',', header);
	Matrix<dnn_double> T = csv1.toMat();
	T = csv1.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			T = T.removeCol(0);
		}
	}

	Matrix<dnn_double> A(T.m, T.n-1);
	Matrix<dnn_double> B(T.m, 1);

	for (int i = 0; i < T.m; i++)
	{
		for (int j = 0; j < T.n - 1; j++)
		{
			A(i, j) = T(i, j + 1);

		}
		B(i, 0) = T(i, 0);
	}

	A.print("A");
	B.print("B");

	std::vector<std::string> header_names;
	header_names.resize(A.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < A.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < A.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}


	multiple_regression mreg;

	mreg.set(A.n);
	mreg.fit(A, B);
	mreg.report(header_names, 0.05);

	printf("multiple_regression END\n\n");
	return 0;
}