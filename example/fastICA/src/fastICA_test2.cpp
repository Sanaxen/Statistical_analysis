//#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def

#include "../../../include/Matrix.hpp"
#include "../../../include/statistical/fastICA.h"
#include "../../../include/util/csvreader.h"
#ifdef USE_GNUPLOT
#include "../../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
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
	std::vector<std::string> x_var;

	int max_ica_iteration = MAX_ITERATIONS;
	double ica_tolerance = TOLERANCE;
	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
			continue;
		}
		else if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}
		else if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
			continue;
		}
		else if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--iter") {
			max_ica_iteration = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--tol") {
			ica_tolerance = atof(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
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

	std::vector<std::string> header_names;
	header_names.resize(T.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < T.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < T.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}
	for (int i = 0; i < T.n; i++)
	{
		printf("[%s]\n", header_names[i].c_str());
	}
	std::vector<int> x_var_idx;

	if (x_var.size())
	{
		for (int i = 0; i < x_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (x_var[i] == header_names[j])
				{
					x_var_idx.push_back(j);
				}
				else if ("\"" + x_var[i] + "\"" == header_names[j])
				{
					x_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
				}
			}
		}
		if (x_var_idx.size() == 0)
		{
			for (int i = 0; i < x_var.size(); i++)
			{
				x_var_idx.push_back(atoi(x_var[i].c_str()));
			}
		}
		if (x_var_idx.size() != x_var.size())
		{
			printf("--x_var ERROR\n");
			return -1;
		}
	}
	if (x_var_idx.size() == 0 && x_var.size() == 0)
	{
		for (int i = 0; i < T.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			x_var.push_back(std::string(buf));
			x_var_idx.push_back(i);
		}
	}

	for (int i = 0; i < x_var_idx.size(); i++)
	{
		printf("[%s]%d\n", header_names[i].c_str(), x_var_idx[i]);
	}
	Matrix<dnn_double> A;
	if (x_var.size())
	{
		std::vector<std::string> header_names_wrk = header_names;
		A = T.Col(x_var_idx[0]);
		header_names[0] = header_names_wrk[x_var_idx[0]];
		for (int i = 1; i < x_var.size(); i++)
		{
			A = A.appendCol(T.Col(x_var_idx[i]));
			header_names[i] = header_names_wrk[x_var_idx[i]];
		}
	}


	compc = A.n;
	rows = A.m;


#ifdef USE_GNUPLOT
	int win_size[2] = { 640 * 2,480 * 2 };
	{
		int win_size[2] = { 640 * 2,480 * 2 };
		gnuPlot plot1(std::string(GNUPLOT_PATH));
		plot1.set_capture(win_size, std::string("mixing_signale.png"));
		plot1.linewidth = 1;
		plot1.set_title("mixing signale");
		plot1.plot_lines(A, header_names, 2000);
		plot1.draw();
	}
#endif

	ICA ica;

	ica.set(compc);

	// ICA computation
	ica.fit(A, max_ica_iteration, ica_tolerance);

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
		for (int i = 0; i < A.n; i++)
		{
			char buf[32];
			sprintf(buf, "source%d", i);
			header_names[i] = buf;
		}
		for (int i = 0; i < A.n; i++)
		{
			printf("[%s]\n", header_names[i].c_str());
		}
		gnuPlot plot1(std::string(GNUPLOT_PATH));
		plot1.set_capture(win_size, std::string("source_signale.png"));

		plot1.linewidth = 1;
		plot1.set_title("source signale");
		for (int c = 0; c < compc; c++)
		{
			std::vector<std::string> header;
			header.push_back(header_names[c]);
			plot1.plot_lines(Xo[c], header, 2000);
		}
		plot1.draw();
	}
#endif

	return 0;
}
