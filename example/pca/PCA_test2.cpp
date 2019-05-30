#include <iostream>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif
#include "../../include/util/cmdline_args.h"

int main(int argc, char** argv)
{
	int resp = commandline_args(&argc, &argv);

	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;
	std::vector<std::string> x_var;

	std::string csvfile("2-3c.csv");

	bool capture = false;
	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
			continue;
		}else
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}else
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
			continue;
		}else
		if (argname == "--capture") {
			capture = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}
		else if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
	}

	CSVReader csv(csvfile, ',', header);
	x = csv.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			x = x.removeCol(0);
		}
	}
	std::vector<std::string> header_names;
	header_names.resize(x.n);
	if (header && csv.getHeader().size() > 0)
	{
		for (int i = 0; i < x.n; i++)
		{
			header_names[i] = csv.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < x.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}
	csv.clear();

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
				else if ("\"" + header_names[j] + "\"" == x_var[i])
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
		for (int i = 0; i < x.n; i++)
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
		A = x.Col(x_var_idx[0]);
		header_names[0] = header_names_wrk[x_var_idx[0]];
		for (int i = 1; i < x_var.size(); i++)
		{
			A = A.appendCol(x.Col(x_var_idx[i]));
			header_names[i] = header_names_wrk[x_var_idx[i]];
		}
		x = A;
	}

	variablesNum = x.n;
	n = x.m;
	x.print();

	PCA pca2;
	pca2.set(variablesNum);

	int stat = pca2.fit(x, true);

	pca2.Report();

	pca2.principal_component().print("Žå¬•ª");
	pca2.principal_component().print_csv("output1.csv");

	stat = pca2.fit(x, true, false);

	pca2.Report();

	pca2.principal_component().print_csv("output2.csv");

#ifdef USE_GNUPLOT

	int win_size[2] = { 640 * 2, 480 * 2 };
	gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);
	if (capture)
	{
		plot1.set_capture(win_size, std::string("principal_component.png"));
	}
	header_names[0] = "First principal component";
	header_names[1] = "Second principal component";
	plot1.scatter(pca2.principal_component(), 0, 1, 1, 30, header_names, 6);
	plot1.probability_ellipse(pca2.principal_component(), 0, 1);
	plot1.draw();
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

