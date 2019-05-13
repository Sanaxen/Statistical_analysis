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
#include "../../../include/util/cmdline_args.h"


int main(int argc, char *argv[])
{
	int resp = commandline_args(&argc, &argv);
	if (resp == -1)
	{
		printf("command line error.\n");
		return -1;
	}

	std::string csvfile("sample.csv");
	std::string x_var = "";
	std::vector<std::string> y_var;
	int max_lines = 2000;

	bool capture = false;
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
			x_var = std::string(argv[count + 1]);
			continue;
		}
		else if (argname == "--y_var") {
			y_var.push_back(argv[count + 1]);
			continue;
		}
		else if (argname == "--max_lines") {
			max_lines = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
			continue;
		}else
		if (argname == "--capture") {
			capture = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
	}

	Matrix<dnn_double> X;

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
	std::vector<int> y_var_idx;

	if (y_var.size())
	{
		for (int i = 0; i < y_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (y_var[i] == header_names[j])
				{
					y_var_idx.push_back(j);
				}
				else if ("\"" + y_var[i] + "\"" == header_names[j])
				{
					y_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (y_var[i] == std::string(buf))
					{
						y_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (y_var[i] == std::string(buf))
					{
						y_var_idx.push_back(j);
					}
				}
			}
		}
		if (y_var_idx.size() == 0)
		{
			for (int i = 0; i < x_var.size(); i++)
			{
				y_var_idx.push_back(atoi(y_var[i].c_str()));
			}
		}
		if (y_var_idx.size() != y_var.size())
		{
			printf("--y_var ERROR\n");
			return -1;
		}
	}
	else
	{
		for (int i = 0; i < header_names.size(); i++)
		{
			y_var_idx.push_back(i);
			y_var.push_back(header_names[i]);
		}
	}

	for (int i = 0; i < y_var.size(); i++)
	{
		printf("[%s]%d\n", header_names[i].c_str(), y_var_idx[i]);
	}

	int x_var_idx = -1;
	if (x_var != "")
	{
		for (int j = 0; j < header_names.size(); j++)
		{
			if (x_var == header_names[j])
			{
				x_var_idx = j;
			}
			else if ("\"" + x_var + "\"" == header_names[j])
			{
				x_var_idx = j;
			}
			else
			{
				char buf[32];
				sprintf(buf, "%d", j);
				if (x_var == std::string(buf))
				{
					x_var_idx = j;
				}
				sprintf(buf, "\"%d\"", j);
				if (x_var == std::string(buf))
				{
					x_var_idx = j;
				}
			}
		}
		if (x_var_idx == -1)
		{
			x_var_idx = atoi(x_var.c_str());

			printf("--x_var ERROR\n");
			return -1;
		}
	}

	Matrix<dnn_double> A;
	if (y_var.size())
	{
		std::vector<std::string> header_names_wrk = header_names;
		A = T.Col(y_var_idx[0]);
		header_names[0] = header_names_wrk[y_var_idx[0]];
		for (int i = 1; i < y_var.size(); i++)
		{
			A = A.appendCol(T.Col(y_var_idx[i]));
			header_names[i] = header_names_wrk[y_var_idx[i]];
		}
	}

	if (x_var != "" && x_var_idx >= 0)
	{
		Matrix<dnn_double> tmp = A;
		A = T.Col(x_var_idx);
		A = A.appendCol(tmp);
	}
	else
	{
		Matrix<dnn_double> tmp = A;
		Matrix<dnn_double> tmp2(A.m,1);
		for (int i = 0; i < A.m; i++) tmp2(i, 0) = i;
		A = tmp2;
		A = A.appendCol(tmp);
	}
	A.print();


#ifdef USE_GNUPLOT
	int win_size[2] = { 640 * 2,480 * 2 };
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH));
		if (capture)
		{
			plot1.set_capture(win_size, std::string("lines.png"));
		}
		plot1.linewidth = 1;
		plot1.set_title("line plot");
		plot1.set_label_x((char*)x_var.c_str());
		plot1.linewidth = 2.0;
		plot1.plot_lines2(A, header_names, max_lines);
		plot1.draw();
	}
#endif

	if (resp == 0)
	{
		for (int i = 0; i < argc; i++)
		{
			delete[] argv[i];
		}
		delete argv;
	}
	return 0;
}
