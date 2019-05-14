//#define USE_MKL

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
#include "../../../include/nonlinear/TimeSeriesRegression.h"
#include "../../../include/nonlinear/MatrixToTensor.h"

#include "gen_test_data.h"
#include "../../include/util/cmdline_args.h"


//#define X_DIM	2
//#define Y_DIM	3

#define IN_SEQ_LEN	15

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
	std::string t_var = "";
	int sequence_length = -1;
	std::string normalization_type = "";

	int classification = -1;
	int read_max = -1;
	bool header = false;
	int start_col = 0;
	int x_dim = 0, y_dim = 0;
	int use_cnn = 1;
	std::string csvfile("sample.csv");
	std::string report_file("TimeSeriesRegression.txt");
	{
		std::ofstream tmp_(report_file);
		if (!tmp_.bad())
		{
			tmp_ << "" << std::endl;
			tmp_.flush();
		}
	}

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--x") {
			x_dim = atoi(argv[count + 1]);
		}else
		if (argname == "--read_max") {
			read_max = atoi(argv[count + 1]);
		}
		else if (argname == "--y") {
			y_dim = atoi(argv[count + 1]);
		}
		else if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
		}
		if (argname == "--y_var") {
			y_var.push_back(argv[count + 1]);
		}
		if (argname == "--t_var") {
			t_var = argv[count + 1];
		}
		if (argname == "--seq_len") {
			sequence_length = atoi(argv[count + 1]);
		}
		if (argname == "--normal")
		{
			normalization_type = argv[count + 1];
			printf("--normal %s\n", argv[count + 1]);
		}
		else if (argname == "--classification") {
			classification = atoi(argv[count + 1]);
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
	int t_var_idx = -1;

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

	if (t_var != "")
	{
		for (int j = 0; j < header_names.size(); j++)
		{
			if (t_var == header_names[j])
			{
				t_var_idx = j;
			}
			else if ("\"" + t_var + "\"" == header_names[j])
			{
				t_var_idx = j;
			}
			else
			{
				char buf[32];
				sprintf(buf, "%d", j);
				if (t_var == std::string(buf))
				{
					t_var_idx = j;
				}
				sprintf(buf, "\"%d\"", j);
				if (t_var == std::string(buf))
				{
					t_var_idx = j;
				}
			}
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
		if (x_var.size()/*+yx_var.size()*/ != x_dim)
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
		x_dim = x_var.size()/*+yx_var.size()*/;
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
	//for (int i = 0; i < yx_var.size(); i++)
	//{
	//	printf("yx_var:%s %d\n", yx_var[i].c_str(), yx_var_idx[i]);
	//}
	for (int i = 0; i < header_names.size(); i++)
	{
		std::replace(header_names[i].begin(), header_names[i].end(), ' ', '_');
	}
	{
		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d,%s\n", y_var_idx[0], header_names[y_var_idx[0]].c_str());
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[x_var_idx[i]].c_str());
		}
		fclose(fp);
	}

	Matrix<dnn_double> x;
	if (x_dim > 0)
	{
		x = z.Col(x_var_idx[0]);
		for (int i = 1; i < x_dim/*- yx_var.size()*/; i++)
		{
			x = x.appendCol(z.Col(x_var_idx[i]));
		}

		//Shift the explanatory variable by one time ago
		Matrix<dnn_double> xx = x.Row(1);
		for (int i = 2; i < x.m - 1; i++)
		{
			xx = xx.appendRow(x.Row(i));
		}
		x.print("x");
		xx = xx.appendRow(x.Row(x.m - 1));
		x = xx;
		x.print("x");
	}

	Matrix<dnn_double> y = z.Col(y_var_idx[0]);
	for (int i = 1; i < y_dim; i++)
	{
		y = y.appendCol(z.Col(y_var_idx[i]));
	}
	y = y.removeRow(y.m - 1);

	y.print("y");

	if (x_var.size())
	{
		for (int i = 0; i < x_var.size(); i++)
		{
			//y = y.appendCol(z.Col(x_var_idx[i]));
			y = y.appendCol(x.Col(i));
		}
	}
	//y.print("y+yx");

	Matrix<dnn_double> tvar;
	if (t_var_idx >= 0)
	{
		tvar = z.Col(t_var_idx);
	}
	else
	{
		tvar = Matrix<dnn_double>(y.m, 1);
		for (int i = 0; i < y.m; i++) tvar(i, 0) = i;
	}


	if(0)
	{
		std::random_device rnd;     // 非決定的な乱数生成器を生成
		std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタ
		std::uniform_real_distribution<> rand(0.0, 1.0);
		FILE* fp = fopen("sample.csv", "w");

		float t = 0;
		float dt = 1;
		int k = 1;
		for (int i = 0; i < 10000; i++)
		{
			float r = rand(mt);
			if (r > 0.2)
			{
				fprintf(fp, "%f,%f,%f\n", t + dt*k, 0.0, 1.0);
			}
			else
			{
				fprintf(fp, "%f,%f,%f\n", t + dt*k, 1.0, 0.0);
			}
			k++;
		}
		fclose(fp);
	}
	printf("sequence_length:%d\n", sequence_length);
	printf("x_dim:%d y_dim:%d\n", x_dim, y_dim);
	x.print();
	y.print();

	if(0)
	{
		int n = 3;
		std::vector<dnn_double> new_y;
		std::vector<dnn_double> new_x;
		std::mt19937 mt(1);
		std::normal_distribution<> norm(0.0, 1.0);
		for (int i = 0; i < y.m - 1; i++)
		{
			for (int k = 0; k < y.n; k++)
			{
				double ys = y(i + 1, k) - y(i, k);
				double yd = ys / (double)n;
				for (int j = 0; j < n; j++)
				{
					if (j > 0 && j < n - 1)
					{
						new_y.push_back(y(i, k) + j*yd + 0.3*fabs(ys)*norm(mt));
					}
					else
					{
						new_y.push_back(y(i, k) + j*yd);
					}
				}
				if (i == y.m - 1)
				{
					new_y.push_back(y(y.m - 1, k));
				}
			}
			for (int k = 0; k < x.n; k++)
			{
				double xs = x(i + 1, k) - x(i, k);
				double xd = xs / (double)n;
				for (int j = 0; j < n; j++)
				{
					new_x.push_back(x(i, k) + j*xd);
				}
				if (i == y.m - 1)
				{
					new_y.push_back(x(y.m - 1, k));
				}
			}
		}
		Matrix<dnn_double> yy(new_y.size() / y.n, y.n);
		Matrix<dnn_double> xx(new_x.size() / x.n, x.n);
		memcpy(yy.v, &(new_y[0]), sizeof(dnn_double)*new_y.size());
		memcpy(xx.v, &(new_x[0]), sizeof(dnn_double)*new_x.size());

		yy.print_csv("yy.csv");
		xx.print_csv("xx.csv");
		
		y = yy;
		x = xx;
		int sequence_length2 = (sequence_length - 1)*(n - 1);
		sequence_length = sequence_length2 - sequence_length2%sequence_length;
		//exit(0);
	}

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X, read_max);
	MatrixToTensor(y, Y, read_max);


	TimeSeriesRegression timeSeries(X, Y, y_dim, normalization_type, classification);
	if (timeSeries.getStatus() == -1)
	{
		if (classification < 2)
		{
			printf("class %.3f %.3f\n", timeSeries.class_minmax[0], timeSeries.class_minmax[1]);
		}
		return -1;
	}

	timeSeries.timevar = tvar;
	timeSeries.x_dim = x_dim;
	timeSeries.y_dim = y_dim;
	timeSeries.tolerance = 0.009;
	timeSeries.learning_rate = 1.0;
	timeSeries.visualize_loss(10);
	timeSeries.plot = 10;

	bool test_mode = false;
	int n_layers = -1;
	int n_rnn_layers = -1;
	int hidden_size = -1;
	bool capture = false;
	float test = 0.4;

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--read_max") {
			continue;
		}else
		if (argname == "--x") {
			continue;
		}
		else if (argname == "--y") {
			continue;
		}
		else if (argname == "--csv") {
			continue;
		} 
		else if (argname == "--col") {
			continue;
		} 
		else if (argname == "--header") {
			continue;
		}
		else if (argname == "--x_var") {
			continue;
		}
		else if (argname == "--y_var") {
			continue;
		}
		else if (argname == "--t_var") {
			continue;
		}
		else if (argname == "--normal") {
			continue;
		}
		else if (argname == "--classification") {
			continue;
		}
		else if (argname == "--bptt_max") {
			timeSeries.n_bptt_max = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--capture") {
			timeSeries.capture = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--progress") {
			timeSeries.progress = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		} else if (argname == "--tol") {
			timeSeries.tolerance = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--learning_rate") {
			timeSeries.learning_rate = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--test_mode") {
			timeSeries.test_mode = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--test") {
			test = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--epochs") {
			timeSeries.n_train_epochs = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--support") {
			timeSeries.support = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--minibatch_size") {
			timeSeries.n_minibatch = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--plot") {
			timeSeries.plot = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--rnn_type") {
			timeSeries.rnn_type = argv[count + 1];
			continue;
		}
		else if (argname == "--opt_type") {
			timeSeries.opt_type = argv[count + 1];
			continue;
		}
		else if (argname == "--seq_len") {
			sequence_length = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--n_layers") {
			n_layers = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--n_rnn_layers") {
			n_rnn_layers = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--hidden_size") {
			hidden_size = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--early_stopping") {
			timeSeries.early_stopping = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--prophecy") {
			timeSeries.prophecy = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_cnn") {
			timeSeries.use_cnn = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--dropout") {
			timeSeries.dropout = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--observed_predict_plot") {
			timeSeries.visualize_observed_predict_plot = atoi(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}

	timeSeries.data_set(sequence_length, test);

	if (sequence_length > timeSeries.n_minibatch)
	{
		printf("!!Warning!! sequence_length:%d > Minibatch:%d\n", sequence_length, timeSeries.n_minibatch);
	}
	std::cout << "Running with the following parameters:" << std::endl
		<< "Learning rate   :   " << timeSeries.learning_rate << std::endl
		<< "Minibatch size  :   " << timeSeries.n_minibatch << std::endl
		<< "Number of epochs:	" << timeSeries.n_train_epochs << std::endl
		<< "plotting cycle  :	" << timeSeries.plot << std::endl
		<< "tolerance       :	" << timeSeries.tolerance << std::endl
		<< "hidden_size     :   " << hidden_size << std::endl
		<< "sequence_length :   " << sequence_length << std::endl
		<< "optimizer       :   " << timeSeries.opt_type << std::endl
		<< "support         :   " << timeSeries.support << std::endl
		<< "n_rnn_layers    :   " << n_rnn_layers << std::endl
		<< "n_layers        :   " << n_layers << std::endl
		<< "test_mode       :   " << timeSeries.test_mode << std::endl
		<< "n_bptt_max       :  " << timeSeries.n_bptt_max << std::endl
		<< "classification   : " << timeSeries.classification << std::endl
		<< "dropout        : " << timeSeries.dropout << std::endl
		<< std::endl;

	{
		FILE* fp = fopen("debug_commandline.txt", "w");
		for (int i = 0; i < argc; i++)
		{
			fprintf(fp, "%s ", argv[i]);
		}
		fclose(fp);
	}
	timeSeries.fit(sequence_length, n_rnn_layers, n_layers, hidden_size);
	timeSeries.report(0.05, report_file);
	if (classification < 2)
	{
		timeSeries.visualize_observed_predict_plot = true;
		timeSeries.visualize_observed_predict();
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
