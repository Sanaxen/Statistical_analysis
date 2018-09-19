//#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def

#include "../../../include/Matrix.hpp"
#include "../../../include/statistical/fastICA.h"
#include "../../../include/util/csvreader.h"
#ifdef USE_GNUPLOT
#include "../../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
#endif

static Matrix<dnn_double> mat_read(FILE *fp, int *rows, int *cols)
{
	int i, j; Matrix<dnn_double> M;

	fscanf(fp, "%d %d", rows, cols);
	M = Matrix<dnn_double>(*rows, *cols);
	
	for (i=0; i<*rows; i++) {
		for (j=0; j<*cols; j++)
			fscanf(fp, "%lf ", &(M(i,j)));	
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

	compc = 3;

#if 0
	// Matrix creation
	if ((fp = fopen("mix_1.csv", "r")) == NULL) {
		perror("Error opening input file");
		exit(-1);
	}
	Matrix<dnn_double> X1 = mat_read2(fp, &rows);

	if ((fp = fopen("mix_2.csv", "r")) == NULL) {
		perror("Error opening input file");
		exit(-1);
	}
	Matrix<dnn_double> X2 = mat_read2(fp, &rows);

	if ((fp = fopen("mix_3.csv", "r")) == NULL) {
		perror("Error opening input file");
		exit(-1);
	}
	Matrix<dnn_double> X3 = mat_read2(fp, &rows);
#else
	CSVReader csv1("mix_1.csv", ',', false);
	CSVReader csv2("mix_2.csv", ',', false);
	CSVReader csv3("mix_3.csv", ',', false);

	Matrix<dnn_double> X1 = csv1.toMat().transpose();
	Matrix<dnn_double> X2 = csv2.toMat().transpose();
	Matrix<dnn_double> X3 = csv3.toMat().transpose();
	rows = X1.m;
#endif

	Matrix<dnn_double> X(rows, 3);
	for (int i = 0; i < rows; i++)
	{
		X(i, 0) = X1(i, 0);
		X(i, 1) = X2(i, 0);
		X(i, 2) = X3(i, 0);
	}

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH));
		plot1.linewidth = 1;
		plot1.title = "mixing signale";
		plot1.plot_lines(X, std::vector<std::string>(), 2000);
		plot1.draw();
	}
#endif

	ICA ica;

	ica.set(compc);

	// ICA computation
	ica.fit(X);

	// Output
	ica.K.print();
	ica.W.print();
	ica.A.print();

	for (int i = 0; i < rows; i++)
	{
		X1(i, 0) = ica.S(i, 0);
		X2(i, 0) = ica.S(i, 1);
		X3(i, 0) = ica.S(i, 2);
	}
	X1.print_csv("output1.csv");
	X2.print_csv("output2.csv");
	X3.print_csv("output3.csv");

	X1.print("X1");
	X2.print("X2");
	X3.print("X3");
#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH));

		plot1.linewidth = 1;
		plot1.title = "source signale";
		plot1.plot_lines(X1, std::vector<std::string>(), 2000);
		plot1.plot_lines(X2, std::vector<std::string>(), 2000);
		plot1.plot_lines(X3, std::vector<std::string>(), 2000);
		plot1.draw();
	}
#endif

	return 0;
}
