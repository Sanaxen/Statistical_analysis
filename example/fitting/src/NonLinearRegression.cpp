#define _cublas_Init_def
#define NOMINMAX
#include "../../../include/Matrix.hpp"
#include "../../../include/statistical/fastICA.h"
#include "../../../include/util/csvreader.h"
#ifdef USE_GNUPLOT
#include "../../../include/util/plot.h"

#endif

#include <iostream>
#include <string.h>
#ifdef USE_MKL
#define CNN_USE_INTEL_MKL
#endif
#include "../../../include/util/dnn_util.hpp"
#include "../../../include/nonlinear/NonLinearRegression.h"
#include "../../../include/nonlinear/MatrixToTensor.h"

#include "gen_test_data.h"


int main(int argc, char** argv)
{
	std::vector<std::string> x_var;
	std::vector<std::string> y_var;
	std::string normalization_type = "zscore";

	double dec_random = 0.0;
	float fluctuation = 0.0;
	int read_max = -1;
	bool header = false;
	int start_col = 0;
	int x_dim = 0, y_dim = 0;
	std::string csvfile("sample.csv");
	std::string report_file("NonLinearRegression.txt");

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--read_max") {
			read_max = atoi(argv[count + 1]);
		}
		else if (argname == "--x") {
			x_dim = atoi(argv[count + 1]);
		}
		else if (argname == "--y") {
			y_dim = atoi(argv[count + 1]);
		}
		else if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		else if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
		}
		else if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		else if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
		}
		else if (argname == "--y_var") {
			y_var.push_back(argv[count + 1]);
		}
		else if (argname == "--normal")
		{
			normalization_type = argv[count + 1];
			printf("--normal %s\n", argv[count + 1]);
		}
		else if (argname == "--dec_random") {
			dec_random = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--fluctuation") {
			fluctuation = atof(argv[count + 1]);
			continue;
		}
	}

	FILE* fp = fopen(csvfile.c_str(), "r");
	if (fp == NULL)
	{
		make_data_set(csvfile, 1000);
	}
	else
	{
		fclose(fp);
	}

	CSVReader csv1(csvfile, ',', header);
	Matrix<dnn_double> z = csv1.toMat();
	z = csv1.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			z = z.removeCol(0);
		}
	}

	std::vector<std::string> header_names;
	header_names.resize(z.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < z.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < z.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}

	std::vector<int> x_var_idx;
	std::vector<int> y_var_idx;

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
			for (int i = 0; i < y_var.size(); i++)
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

	if (x_var.size() == 0 && x_dim > 0)
	{
		for (int i = 0; i < x_dim; i++)
		{
			char buf[32];
			sprintf(buf, "\"%d\"", i);
			x_var.push_back(buf);
			x_var_idx.push_back(i);
		}
	}
	if (x_var.size() > 0 && x_dim > 0)
	{
		if (x_var.size() != x_dim)
		{
			printf("arguments number error:--x_var != --x");
			return -1;
		}
	}

	if (y_var.size() > 0 && y_dim > 0)
	{
		if (y_var.size() != y_dim)
		{
			printf("arguments number error:--y_var != --y");
			return -1;
		}
	}
	if (y_var.size() == 0 && y_dim > 0)
	{
		for (int i = 0; i < z.n; i++)
		{
			bool dup = false;
			for (int j = 0; j < x_var.size(); j++)
			{
				if (x_var_idx[j] == i)
				{
					dup = true;
					break;
				}
			}
			if (!dup)
			{
				char buf[128];
				sprintf(buf, "\"%d\"", i);
				y_var.push_back(buf);
				y_var_idx.push_back(i);
			}
		}
	}
	if (x_var.size() > 0 && x_dim == 0)
	{
		x_dim = x_var.size();
	}
	if (y_var.size() > 0 && y_dim == 0)
	{
		y_dim = y_var.size();
	}

	for (int i = 0; i < x_var.size(); i++)
	{
		printf("x_var:%s %d\n", x_var[i].c_str(), x_var_idx[i]);
	}
	for (int i = 0; i < y_var.size(); i++)
	{
		printf("y_var:%s %d\n", y_var[i].c_str(), y_var_idx[i]);
	}
	Matrix<dnn_double> x = z.Col(x_var_idx[0]);
	for (int i = 1; i < x_dim; i++)
	{
		x = x.appendCol(z.Col(x_var_idx[i]));
	}
	Matrix<dnn_double> y = z.Col(y_var_idx[0]);
	for (int i = 1; i < y_dim; i++)
	{
		y = y.appendCol(z.Col(y_var_idx[i]));
	}
	printf("x_dim:%d y_dim:%d\n", x_dim, y_dim);
	x.print();
	y.print();

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X, read_max);
	MatrixToTensor(y, Y, read_max);

	NonLinearRegression regression(X, Y, normalization_type, dec_random, fluctuation);

	regression.tolerance = 1.0e-3;
	regression.learning_rate = 1;
	regression.visualize_loss(10);
	regression.plot = 10;

	int n_layers = -1;
	int input_unit = -1;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--read_max") {
			continue;
		}
		else if (argname == "--x") {
			continue;
		}
		else if (argname == "--y") {
			continue;
		}
		else if (argname == "--csv") {
			continue;
		}else
		if (argname == "--col") {
			continue;
		}
		else
		if (argname == "--header") {
			continue;
		}
		else if (argname == "--x_var") {
			continue;
		}
		else if (argname == "--y_var") {
			continue;
		}
		else if (argname == "--normal") {
			continue;
		}
		else if (argname == "--dec_random") {
			continue;
		}
		else if (argname == "--fluctuation") {
			continue;
		}
		else if (argname == "--capture") {
			regression.capture = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--progress") {
			regression.progress = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--tol") {
			regression.tolerance = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--learning_rate") {
			regression.learning_rate = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--test_mode") {
			regression.test_mode = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--test") {
			regression.data_set(atof(argv[count + 1]));
			continue;
		}
		else if (argname == "--epochs") {
			regression.n_train_epochs = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--minibatch_size") {
			regression.n_minibatch = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--plot") {
			regression.plot = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--n_layers") {
			n_layers = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--input_unit") {
			input_unit = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--opt_type") {
			regression.opt_type = argv[count + 1];
			continue;
		}
		else if (argname == "--early_stopping") {
			regression.early_stopping = atoi(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}
	std::cout << "Running with the following parameters:" << std::endl
		<< "Learning rate   : " << regression.learning_rate << std::endl
		<< "Minibatch size  : " << regression.n_minibatch << std::endl
		<< "Number of epochs: " << regression.n_train_epochs << std::endl
		<< "plotting cycle  : " << regression.plot << std::endl
		<< "tolerance       : " << regression.tolerance << std::endl
		<< "optimizer       : " << regression.opt_type << std::endl
		<< "input_unit      : " << input_unit << std::endl
		<< "n_layers        : " << n_layers << std::endl
		<< "test_mode       : " << regression.test_mode << std::endl
		<< "Decimation of random points       : " << regression.dec_random << std::endl
		<< "random fluctuation       : " << regression.fluctuation << std::endl
		<< std::endl;

	regression.fit(n_layers, input_unit);
	regression.report(0.05, report_file);

#ifdef USE_GNUPLOT
	{
		int win_size[2] = { 640 * 3, 480 * 3 };
		std::vector<std::string> header_names(2);
		header_names[0] = "observed";
		header_names[1] = "predict";

		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 10);
		plot1.set_capture(win_size, std::string("observed_predict.png"));

		Matrix<dnn_double> T(regression.Diff.size()*regression.Diff[0].size()/2, 2);
		for (int i = 0; i < regression.Diff.size(); i++)
		{
			for (int j = 0; j < regression.Diff[0].size()/2; j++)
			{
				T(i*regression.Diff[0].size()/2 + j, 0) = regression.Diff[i][2 * j];
				T(i*regression.Diff[0].size()/2 + j, 1) = regression.Diff[i][2 * j + 1];
			}
		}

		plot1.scatter(T, 0, 1, 1, 30, header_names, 5);
		if (10)
		{
			double max_x = T.Col(0).Max();
			double min_x = T.Col(0).Min();
			double step = (max_x - min_x) / 5.0;
			Matrix<dnn_double> x(6, 2);
			Matrix<dnn_double> v(1, 1);
			for (int i = 0; i < 6; i++)
			{
				v(0, 0) = min_x + i * step;
				x(i, 0) = v(0, 0);
				x(i, 1) = v(0, 0);
			}
			plot1.set_label(0.5, 0.85, 1, "observed=predict");
			plot1.plot_lines2(x, header_names);
			plot1.draw();
		}
	}
#endif

	return 0;
}
