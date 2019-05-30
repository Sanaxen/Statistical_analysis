#define USE_MKL
#define CNN_USE_AVX

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

#include "../../../include/nonlinear/image_util.h"
#include "gen_test_data.h"
#include "../../include/util/cmdline_args.h"


int main(int argc, char** argv)
{
	int resp = commandline_args(&argc, &argv);
	if (resp == -1)
	{
		printf("command line error.\n");
		return -1;
	}

	std::vector<std::string> x_var;
	std::vector<std::string> y_var;
	std::string normalization_type = "zscore";

	int classification = -1;
	std::string regression_type = "";
	double dec_random = 0.0;
	float fluctuation = 0.0;
	int read_max = -1;
	bool header = false;
	int start_col = 0;
	int x_dim = 0, y_dim = 0;
	int x_s = 0;
	int y_s = 0;
	bool test_mode = false;
	std::string csvfile("sample.csv");
	std::string report_file("NonLinearRegression.txt");

	std::string data_path = "";
	//{
	//	std::ofstream tmp_(report_file);
	//	if (!tmp_.bad())
	//	{
	//		tmp_ << "" << std::endl;
	//		tmp_.flush();
	//	}
	//}

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--read_max") {
			read_max = atoi(argv[count + 1]);
		}
		else if (argname == "--dir") {
			data_path = argv[count + 1];
		}
		else if (argname == "--x") {
			if (sscanf(argv[count + 1], "%d:%d", &x_s, &x_dim) == 2)
			{
			}
			else
			{
				x_dim = atoi(argv[count + 1]);
			}
		}
		else if (argname == "--y") {
			if (sscanf(argv[count + 1], "%d:%d", &y_s, &y_dim) == 2)
			{
			}
			else
			{
				y_dim = atoi(argv[count + 1]);
			}
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
		else if (argname == "--test_mode") {
			test_mode = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--dec_random") {
			dec_random = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--fluctuation") {
			fluctuation = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--regression") {
			regression_type = argv[count + 1];
			continue;
		}
		else if (argname == "--classification") {
			classification = atoi(argv[count + 1]);
			continue;
		}
	}

	if (data_path != "")
	{
		FILE* fp = fopen(csvfile.c_str(), "r");
		if (fp)
		{
			fclose(fp);
		}
		else
		{
			int stat = filelist_to_csv(csvfile, data_path, test_mode == false, classification, header, normalization_type);
			if (stat != 0)
			{
				return -1;
			}
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
	for (int i = 0; i < header_names.size(); i++)
	{
		std::replace(header_names[i].begin(), header_names[i].end(), ' ', '_');
	}
	csv1.clear();

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
				else if ("\"" + header_names[j] + "\"" == y_var[i])
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
			sprintf(buf, "\"%d\"", i + x_s);
			x_var.push_back(buf);
			x_var_idx.push_back(i + x_s);
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
				if (x_var_idx[j] == i + y_s)
				{
					dup = true;
					break;
				}
			}
			if (!dup)
			{
				char buf[128];
				sprintf(buf, "\"%d\"", i + y_s);
				y_var.push_back(buf);
				y_var_idx.push_back(i + y_s);
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
		if (x_var.size() > 80 && i == 4) break;
	}
	if (x_var.size() > 80)
	{
		printf("...\n");
		for (int i = x_var.size() - 4; i < x_var.size(); i++)
		{
			printf("x_var:%s %d\n", x_var[i].c_str(), x_var_idx[i]);
		}
	}
	printf("\n");
	for (int i = 0; i < y_var.size(); i++)
	{
		printf("y_var:%s %d\n", y_var[i].c_str(), y_var_idx[i]);
		if (y_var.size() > 80 && i == 4) break;
	}
	if (y_var.size() > 80)
	{
		printf("...\n");
		for (int i = y_var.size() - 4; i < y_var.size(); i++)
		{
			printf("y_var:%s %d\n", y_var[i].c_str(), y_var_idx[i]);
		}
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

	if (data_path != "" && read_max > 0)
	{
		std::mt19937 mt(read_max);
		std::uniform_int_distribution<int> rand_ts(0, x.m - 1);
		std:vector<int> index(x.m - 1, -1);
		
		Matrix<dnn_double> xx = x.Row(0);
		Matrix<dnn_double> yy = y.Row(0);
		index[0] = 1;
		for (int i = 0; i < read_max-1; i++)
		{
			int idx = rand_ts(mt);
			while (index[idx] != -1)
			{
				idx = rand_ts(mt);
			}
#pragma omp parallel sections
			{
				#pragma omp section
				{
					xx = xx.appendRow(x.Row(idx));
				}
				#pragma omp section
				{
					yy = yy.appendRow(y.Row(idx));
				}
			}
			index[idx] = 1;
		}
		x = xx;
		y = yy;
	}

	x.print();
	y.print();

	{
		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d,%s\n", y_var_idx[0], header_names[y_var_idx[0]].c_str());
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[x_var_idx[i]].c_str());
		}
		fclose(fp);
	}

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X, read_max);
	MatrixToTensor(y, Y, read_max);
	x = Matrix<dnn_double>(1, 1);
	y = Matrix<dnn_double>(1, 1);

	NonLinearRegression regression(X, Y, normalization_type, dec_random, fluctuation, regression_type, classification);
	if (regression.getStatus() == -1)
	{
		if (classification < 2)
		{
			printf("class %.3f %.3f\n", regression.class_minmax[0], regression.class_minmax[1]);
		}
		return -1;
	}
	regression.tolerance = 1.0e-3;
	regression.learning_rate = 1;
	regression.visualize_loss(10);
	regression.plot = 10;
	regression.test_mode = test_mode;

	double test_num = 0;
	int n_layers = -1;
	int input_unit = -1;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--dir") {
			continue;
		}
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
		else if (argname == "--regression") {
			continue;
		}
		else if (argname == "--classification") {
			continue;
		}
		else if (argname == "--test_mode") {
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
		else if (argname == "--test") {
			test_num = atof(argv[count + 1]);
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
		else if (argname == "--dropout") {
			regression.dropout = atof(argv[count + 1]);
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
		else if (argname == "--observed_predict_plot") {
			regression.visualize_observed_predict_plot = atoi(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}
	regression.data_set(test_num);

	if (regression.iY.size() < regression.n_minibatch)
	{
		printf("data %d < minibatch %d\n", regression.iY.size(), regression.n_minibatch);
		return -1;
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
		<< "regression       : " << regression.regression << std::endl
		<< "classification       : " << regression.classification << std::endl
		<< "dropout       : " << regression.dropout << std::endl
		<< std::endl;

	{
		FILE* fp = fopen("debug_commandline.txt", "w");
		fprintf(fp, ":%s\n", regression.regression.c_str());
		for (int i = 0; i < argc; i++)
		{
			fprintf(fp, "%s ", argv[i]);
		}
		fclose(fp);
	}
	regression.fit(n_layers, input_unit);
	regression.report(0.05, report_file);
	if (classification < 2)
	{
		regression.visualize_observed_predict_plot = true;
		regression.visualize_observed_predict();
	}
	
	{
		std::ofstream stream("Time_to_finish.txt");
		if (!stream.bad())
		{
			stream << "Time to finish:" << 0 << "[sec] = " << 0 << "[min]" << std::endl;
			stream.flush();
		}
	}

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
