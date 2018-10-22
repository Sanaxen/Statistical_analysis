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
	
#ifndef USE_FLOAT
	for (i=0; i<*rows; i++) {
		for (j=0; j<*cols; j++)
			fscanf(fp, "%lf ", &(M(i,j)));	
	}
#else
	for (i = 0; i<*rows; i++) {
		for (j = 0; j<*cols; j++)
			fscanf(fp, "%f ", &(M(i, j)));
	}
#endif

	return M;	
}
static Matrix<dnn_double> mat_read2(FILE *fp, int *rows)
{
	int i, j; Matrix<dnn_double> M;

	std::vector<dnn_double> d;

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
	std::string csvfile("sample.csv");

	int max_ica_iteration = MAX_ITERATIONS;
	double ica_tolerance = TOLERANCE;
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
		else if (argname == "--iter") {
			max_ica_iteration = atoi(argv[count + 1]);
		}
		else if (argname == "--tol") {
			ica_tolerance = atof(argv[count + 1]);
		}
	}

	int rows, cols, compc;
	FILE* fp = fopen(csvfile.c_str(), "r");
	Matrix<dnn_double> X;

	if (fp == NULL)
	{
	compc = 3;

	CSVReader csv1("mix_1.csv", ',', false);
	CSVReader csv2("mix_2.csv", ',', false);
	CSVReader csv3("mix_3.csv", ',', false);

	Matrix<dnn_double> X1 = csv1.toMat().transpose();
	Matrix<dnn_double> X2 = csv2.toMat().transpose();
	Matrix<dnn_double> X3 = csv3.toMat().transpose();
	rows = X1.m;

		X = Matrix<dnn_double>(rows, 3);

	for (int i = 0; i < rows; i++)
	{
		X(i, 0) = X1(i, 0);
		X(i, 1) = X2(i, 0);
		X(i, 2) = X3(i, 0);
	}
		X.print_csv("sample.csv");
	}
	else
	{
		fclose(fp);
	}

	CSVReader csv(csvfile, ',', header);
	X = csv.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			X = X.removeCol(0);
		}
	}
	compc = X.n;
	rows = X.m;

	std::vector<std::string> header_names;
	header_names.resize(X.n);
	if (header && csv.getHeader().size() > 0)
	{
		for (int i = 0; i < X.n; i++)
		{
			header_names[i] = csv.getHeader(i+ start_col);
		}
	}
	else
	{
		for (int i = 0; i < X.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH));
		plot1.linewidth = 1;
		plot1.set_title("mixing signale");
		plot1.plot_lines(X, header_names, 2000);
		plot1.draw();
	}
#endif

	ICA ica;

	ica.set(compc);

	// ICA computation
	ica.fit(X, max_ica_iteration, ica_tolerance);

	// Output
	ica.K.print();
	ica.W.print();
	ica.A.print();

	std::vector<Matrix<dnn_double>> Xo;
	Xo.resize(compc);

	for (int c = 0; c < compc; c++)
	{
		Xo[c] = Matrix<dnn_double>(rows, 1);
	for (int i = 0; i < rows; i++)
	{
			Xo[c](i, 0) = ica.S(i, c);
		}
	}
	for (int c = 0; c < compc; c++)
	{
		char buf[256];
		sprintf(buf, "output%d.csv", c);

		Xo[c].print_csv(buf);
		Xo[c].print(buf);
	}

#ifdef USE_GNUPLOT
	{
		for (int i = 0; i < X.n; i++)
		{
			char buf[32];
			sprintf(buf, "source%d", i);
			header_names[i] = buf;
		}
		gnuPlot plot1(std::string(GNUPLOT_PATH));

		plot1.linewidth = 1;
		plot1.set_title("source signale");
		for (int c = 0; c < compc; c++)
		{
			plot1.plot_lines(Xo[c], header_names, 2000);
		}
		plot1.draw();
	}
#endif

	return 0;
}
