#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/statistical/LinearRegression.h"
//#include "../../include/util/mathutil.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif
int main(int argc, char** argv)
{
	printf("multiple_regression START\n");
	std::string csvfile("sample.csv");
	
	std::vector<std::string> x_var;
	std::string y_var = "";

	bool normalize = false;
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
		if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
		}
		if (argname == "--y_var") {
			y_var = argv[count + 1];
		}
		if (argname == "--normalize") {
			normalize = (atoi(argv[count + 1]) != 0) ? true : false;
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
	if (normalize)
	{
		auto& mean = T.Mean();
		T = T.whitening(mean, T.Std(mean));
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
	std::vector<int> x_var_idx;
	int y_var_idx = -1;

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
	//printf("y_var=%s\n", y_var.c_str());
	//printf("header_names.size():%d\n", header_names.size());
	if (y_var != "")
	{
		for (int j = 0; j < header_names.size(); j++)
		{
			if (y_var == header_names[j])
			{
				y_var_idx = j;
			}
			else
			{
				char buf[32];
				sprintf(buf, "%d", j);
				if (y_var == std::string(buf))
				{
					y_var_idx = j;
				}
				sprintf(buf, "\"%d\"", j);
				if (y_var == std::string(buf))
				{
					y_var_idx = j;
				}
			}
		}
		if (y_var_idx == -1)
		{
			y_var_idx = atoi(y_var.c_str());

			printf("--y_var ERROR\n");
			return -1;
		}
	}

	Matrix<dnn_double> A;
	Matrix<dnn_double> B;
	
	if (x_var.size() && y_var_idx >= 0)
	{
		std::vector<std::string> header_names_wrk = header_names;
		A = T.Col(x_var_idx[0]);
		header_names[1] = header_names_wrk[x_var_idx[0]];
		for (int i = 1; i < x_var.size(); i++)
		{
			A = A.appendCol(T.Col(x_var_idx[i]));
			header_names[i+1] = header_names_wrk[x_var_idx[i]];
		}
		B = T.Col(y_var_idx);
		header_names[0] = header_names_wrk[y_var_idx];
	}
	else
	if (x_var.size() == 0 && y_var_idx >= 0)
	{
		printf("y_var=%s\n", y_var.c_str());
		std::vector<std::string> header_names_wrk = header_names;
		
		A = T.removeCol(y_var_idx);
		B = T.Col(y_var_idx);
		header_names[0] = header_names_wrk[y_var_idx];
		for (int i = 0; i < T.n; i++)
		{
			if (i == y_var_idx)
			{
				continue;
			}
			if (i < y_var_idx)
			{
				header_names[i+1] = header_names_wrk[i];
			}
			else
			{
				header_names[i] = header_names_wrk[i];
			}
		}
	}
	else
	{

		A = Matrix<dnn_double>(T.m, T.n - 1);
		B = Matrix<dnn_double>(T.m, 1);
		for (int i = 0; i < T.m; i++)
		{
			for (int j = 0; j < T.n - 1; j++)
			{
				A(i, j) = T(i, j + 1);

			}
			B(i, 0) = T(i, 0);
		}
	}

	A.print("A");
	B.print("B");

	A.print_csv("A.csv");
	fflush(stdout);

	multiple_regression mreg;

	mreg.set(A.n);
	mreg.fit(A, B);
	mreg.report(header_names, 0.05);

	Matrix<dnn_double> cor = A.Cor();
	cor.print_csv("cor.csv");
#ifdef USE_GNUPLOT
	if (A.n > 1)
	{
		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 1, false);
		plot1.Heatmap(cor, header_names, header_names);
		plot1.draw();
	}
	else
	{
		double max_x = A.Max();
		double min_x = A.Min();
		double step = (max_x - min_x) / 3.0;
		Matrix<dnn_double> x(4, 2);
		Matrix<dnn_double> v(1, 1);
		for (int i = 0; i < 4; i++)
		{
			v(0, 0) = min_x + i*step;
			x(i, 0) = v(0, 0);
			x(i, 1) = mreg.predict(v);
		}
		std::vector<std::string> line_header_names(1);
		line_header_names[0] = "linear regression";

		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6, false);
		plot1.plot_lines2(x, line_header_names);

		plot1.scatter(T, x_var_idx[0], y_var_idx, 1, 30, header_names, 5);

		if (true)
		{
			plot1.probability_ellipse(T, x_var_idx[0], y_var_idx);
		}
		plot1.draw();
	}
#endif


	printf("multiple_regression END\n\n");
	return 0;
}