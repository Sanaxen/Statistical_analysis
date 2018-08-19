//#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def

#include "../../../include/Matrix.hpp"
#include "../../../include/statistical/fastICA.h"

static Matrix<dnn_double> mat_read(FILE *fp, int *rows, int *cols)
{
	int i, j; Matrix<dnn_double> M;

	fscanf(fp, "%d %d", rows, cols);
	M = Matrix<dnn_double>(*rows, *cols);

	for (i = 0; i<*rows; i++) {
		for (j = 0; j<*cols; j++)
			fscanf(fp, "%lf ", &(M(i, j)));
	}

	return M;
}
static Matrix<dnn_double> mat_read2(FILE *fp, int *rows)
{
	int i, j; Matrix<dnn_double> M;

	std::vector<double> d;

	int s;
	do
	{
		double x;
		s = fscanf(fp, "%lf,", &x);
		if (s == 1) d.push_back(x);
		else
		{
			s = fscanf(fp, "%lf\n", &x);
			if (s == 1) d.push_back(x);
		}
	} while (s == 1);

	*rows = d.size();
	M = Matrix<dnn_double>(&d[0], d.size(), 1);

	fclose(fp);

	return M;
}

// !! [example\sample_data\fastICA\run.bat]
int main(int argc, char *argv[])
{
	int rows, cols, compc;
	FILE *fp;

	compc = 2;

	// Matrix creation
	if ((fp = fopen("mix1.csv", "r")) == NULL) {
		perror("Error opening input file");
		exit(-1);
	}
	Matrix<dnn_double> X1 = mat_read2(fp, &rows);

	if ((fp = fopen("mix2.csv", "r")) == NULL) {
		perror("Error opening input file");
		exit(-1);
	}
	Matrix<dnn_double> X2 = mat_read2(fp, &rows);

	X1.print_csv("input1.csv");
	X2.print_csv("input2.csv");

	Matrix<dnn_double> X(rows, 2);
	for (int i = 0; i < rows; i++)
	{
		X(i, 0) = X1(i, 0);
		X(i, 1) = X2(i, 0);
	}

	ICA ica;

	ica.set(compc);

	// ICA computation
	ica.fit(X);

	for (int i = 0; i < rows; i++)
	{
		X1(i, 0) = ica.S(i, 0);
		X2(i, 0) = ica.S(i, 1);
	}
	X1.print_csv("output1.csv");
	X2.print_csv("output2.csv");

	return 0;
}
