#define USE_MKL
#define CNN_USE_AVX
//#define USE_LIBTORCH

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

#include "../../../include/nonlinear/break_solver.h"
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
		printf("ERROR:command line error.\n");
		return -1;
	}
	clear_stopping_solver();
	clear_distribution_dat();

	std::vector<std::string> xx_var;	//non normalize var
	double xx_var_scale = 1.0;			// non normalize var scaling
	std::vector<std::string> x_var;
	std::vector<std::string> y_var;
	std::string normalization_type = "zscore";
	bool use_trained_scale = true;

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
	int use_cnn = 1;
	bool test_mode = false;
	bool dump_input = false;
	std::string weight_init_type = "xavier";
	bool layer_graph_only = false;
	std::string multi_files = "";

	int use_libtorch = 0;
	std::string device_name = "cpu";
	int multiplot_step = 3;

	std::string activation_fnc = "tanh";

	std::string csvfile("sample.csv");
	std::string report_file("NonLinearRegression.txt");

	std::string data_path = "";
	bool L1_loss = false;
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
		if (argname == "--multi_files") {
			multi_files = std::string(argv[count + 1]);
		}else
		if (argname == "--activation_fnc") {
			activation_fnc = std::string(argv[count + 1]);
		}else
		if (argname == "--multiplot_step") {
			multiplot_step = atoi(argv[count + 1]);
		}
		else
		if (argname == "--device_name") {
			device_name = std::string(argv[count + 1]);
		}
		else
		if (argname == "--use_libtorch") {
			use_libtorch = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		else
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
		else if (argname == "--xx_var_scale") {
			xx_var_scale = atof(argv[count + 1]);
		}
		else if (argname == "--xx_var") {
			xx_var.push_back(argv[count + 1]);
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
		else if (argname == "--dump_input") {
			dump_input = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--layer_graph_only") {
			layer_graph_only = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--use_trained_scale")
		{
			use_trained_scale = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--L1_loss")
		{
			L1_loss = (atoi(argv[count + 1])) ? true : false;
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

	std::string multi_files_dir = "";
	std::vector<std::string> multi_files_list;
	if (multi_files != "")
	{
		char tmp[640];
		char filename[640];
		strcpy(tmp, multi_files.c_str());
		char* p = tmp;
		if (p[0] == '\"' || p[0] == '\'')
		{
			strcpy(filename, p + 1);
		}
		p = filename;
		if (p[strlen(p) - 1] == '\"' || p[strlen(p) - 1] == '\'')
		{
			p[strlen(p) - 1] = '\0';
		}
		multi_files_dir = get_dirname(std::string(filename));
		getFileNames(multi_files_dir, multi_files_list);
	}

	if (multi_files_list.size() > 1)
	{
		csvfile = multi_files_list[0];
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
	if (multi_files_list.size() > 1)
	{
		z = concat_csv(multi_files_list, header, start_col, 0);

		z.print_csv((char*)(multi_files_dir + std::string("\\concat.csv")).c_str(), header_names);
		return 0;
	}

	std::vector<int> xx_var_idx;
	std::vector<int> x_var_idx;
	std::vector<int> y_var_idx;

	if (xx_var.size())
	{
		for (int i = 0; i < xx_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (xx_var[i] == header_names[j])
				{
					xx_var_idx.push_back(j);
				}
				else if ("\"" + xx_var[i] + "\"" == header_names[j])
				{
					xx_var_idx.push_back(j);
				}
				else if ("\"" + header_names[j] + "\"" == xx_var[i])
				{
					xx_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (xx_var[i] == std::string(buf))
					{
						xx_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (xx_var[i] == std::string(buf))
					{
						xx_var_idx.push_back(j);
					}
				}
			}
		}
		if (xx_var_idx.size() == 0)
		{
			for (int i = 0; i < xx_var.size(); i++)
			{
				xx_var_idx.push_back(atoi(x_var[i].c_str()));
			}
		}
		if (xx_var_idx.size() != xx_var.size())
		{
			printf("ERROR:--x_var ERROR\n");
			return -1;
		}
	}

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
			printf("ERROR:--x_var ERROR\n");
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
			printf("ERROR:--y_var ERROR\n");
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
			printf("ERROR:arguments number error:--x_var != --x");
			return -1;
		}
	}

	if (y_var.size() > 0 && y_dim > 0)
	{
		if (y_var.size() != y_dim)
		{
			printf("ERROR:arguments number error:--y_var != --y");
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
	for (int i = 0; i < xx_var_idx.size(); i++)
	{
		printf("xx_var_idx:%s %d\n", xx_var[i].c_str(), xx_var_idx[i]);
	}

	std::vector<int> normalizeskipp;
	std::vector<int> flag;
	int flg_idx = -1;
	for (int i = 0; i < xx_var_idx.size(); i++)
	{
		if (flg_idx < xx_var_idx[i]) flg_idx = xx_var_idx[i];
	}
	for (int i = 0; i < x_var_idx.size(); i++)
	{
		if (flg_idx < x_var_idx[i]) flg_idx = x_var_idx[i];
	}
	for (int i = 0; i < y_var_idx.size(); i++)
	{
		if (flg_idx < y_var_idx[i]) flg_idx = y_var_idx[i];
	}
	flag.resize(flg_idx+1, 0);

	if (xx_var_idx.size())
	{
		for (int k = 0; k < xx_var_idx.size(); k++)
		{
			flag[xx_var_idx[k]] = 1;
		}
	}

	//for (int k = 0; k < flg_idx; k++)
	//{
	//	printf("flag[%d] %d\n", k, flag[k]);
	//}

	//printf("0:%d -- %d\n", x_var_idx[0], flag[x_var_idx[0]]);
	Matrix<dnn_double> x = z.Col(x_var_idx[0]);
	if (flag[x_var_idx[0]]) normalizeskipp.push_back(1);
	else  normalizeskipp.push_back(0);
	for (int i = 1; i < x_dim; i++)
	{
		//printf("%d:%d -- %d\n", i, x_var_idx[i], flag[x_var_idx[i]]);
		x = x.appendCol(z.Col(x_var_idx[i]));
		if (flag[x_var_idx[i]]) normalizeskipp.push_back(1);
		else  normalizeskipp.push_back(0);
	}
	//for (int k = 0; k < normalizeskipp.size(); k++)
	//{
	//	printf("@@[%d] %d\n", k, normalizeskipp[k]);
	//}

	Matrix<dnn_double> y = z.Col(y_var_idx[0]);
	for (int i = 1; i < y_dim; i++)
	{
		y = y.appendCol(z.Col(y_var_idx[i]));
	}
	//for (int k = 0; k < normalizeskipp.size(); k++)
	//{
	//	printf("@[%d] %d\n", k, normalizeskipp[k]);
	//}
	printf("x_dim:%d y_dim:%d\n", x_dim, y_dim);

	if (dump_input)
	{
		Matrix<dnn_double> tmp_mat;
		//Matrix<dnn_double>& tvar = Matrix<dnn_double>(y.m, 1);
		//for (int i = 0; i < y.m; i++) tvar(i, 0) = i;
		tmp_mat = y;
		tmp_mat = tmp_mat.appendCol(x);
		tmp_mat.print_csv("fit_input.csv");
		exit(0);
	}

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
		if (fp)fclose(fp);
		
		fp = fopen("select_variables2.dat", "w");
		for (int i = 0; i < xx_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", xx_var_idx[i], header_names[xx_var_idx[i]].c_str());
		}
		if (fp)fclose(fp);
	}

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X, read_max);
	MatrixToTensor(y, Y, read_max);
	x = Matrix<dnn_double>(1, 1);
	y = Matrix<dnn_double>(1, 1);

#ifdef USE_LIBTORCH
	if (use_libtorch)
	{
		torch_train_init();
		torch_setDevice(device_name.c_str());
	}
#endif

	NonLinearRegression regression(X, Y, normalizeskipp, normalization_type, dec_random, fluctuation, regression_type, classification, test_mode, use_trained_scale);
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

	regression.header = header_names;
	regression.normalizeskipp = normalizeskipp;
	regression.xx_var_scale = xx_var_scale;
	regression.x_idx = x_var_idx;
	regression.y_idx = y_var_idx;
	regression.weight_init_type = weight_init_type;
	regression.layer_graph_only = layer_graph_only;
	regression.activation_fnc = activation_fnc;

#ifdef USE_LIBTORCH
	regression.use_libtorch = use_libtorch;
	regression.device_name = device_name;
#endif
	regression.L1_loss = L1_loss;

	double test_num = 0;
	int n_layers = -1;
	int input_unit = -1;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--multi_files") {
			continue;
		}
		if (argname == "--activation_fnc") {
			continue;
		}
		else
		if (argname == "--multiplot_step") {
			continue;
		}else
		if (argname == "--device_name") {
			continue;
		}else
		if (argname == "--use_libtorch") {
			continue;
		}else
		if (argname == "--dir") {
			continue;
		}
		else 
		if (argname == "--layer_graph_only") {
			continue;
		}else
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
		else if (argname == "--xx_var") {
			continue;
		}
		else if (argname == "--xx_var_scale") {
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
		else if (argname == "--use_trained_scale")
		{
			continue;
		}
		else if (argname == "--L1_loss")
		{
			continue;
		}
		else if (argname == "--inversion") {
			regression.inversion = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--residual")
		{
			regression.residual = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--padding_prm")
		{
			regression.padding_prm = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_cnn") {
			regression.use_cnn = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_cnn_add_bn") {
			regression.use_cnn_add_bn = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_add_bn") {
			regression.use_add_bn = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--n_sampling")
		{
			regression.n_sampling = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--weight_init_type") {
			regression.weight_init_type = std::string(argv[count + 1]);
			continue;
		}
		else if (argname == "--batch_shuffle")
		{
			regression.batch_shuffle = (0 < atoi(argv[count + 1])) ? true : false;
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
		else if (argname == "--eval_minibatch_size") {
			regression.n_eval_minibatch = atoi(argv[count + 1]);
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
		printf("ERROR:data %d < minibatch %d\n", regression.iY.size(), regression.n_minibatch);
		return -1;
	}
	if (regression.classification > 1)
	{
		printf("!!Warning!! classification:%d > 1 , n_sampling:%d -> 0\n", classification, regression.n_sampling);
		regression.n_sampling = 0;
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
		<< "weight_init_type       : " << regression.weight_init_type << std::endl
		<< "use_trained_scale: " << regression.use_trained_scale << std::endl
		<< "device_name: " << regression.device_name << std::endl
		<< "use_libtorch: " << regression.use_libtorch << std::endl
		<< "batch_shuffle: " << regression.batch_shuffle << std::endl
		<< "n_sampling: " << regression.n_sampling << std::endl
		<< "use_cnn      : " << regression.use_cnn << std::endl
		<< "padding_prm      : " << regression.padding_prm << std::endl
		<< "residual      : " << regression.residual << std::endl
		<< "dump_input      : " << dump_input << std::endl
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

	multiplot_gnuplot_script(regression.y_idx.size(), multiplot_step, header_names, y_var_idx, false, regression.n_sampling);
	
	regression.fit(n_layers, input_unit);
	
	if (layer_graph_only)
	{
		goto end;
	}
	regression.report(0.05, report_file);
	if (classification < 2)
	{
		regression.visualize_observed_predict_plot = true;
		regression.visualize_observed_predict();
	}
	regression.gen_visualize_fit_state();

#ifdef USE_LIBTORCH
	if (regression.n_sampling > 0)
	{
		std::uniform_real_distribution r(SAMPLING_RANGE_MIN, SAMPLING_RANGE_MAX);
		for (int i = 0; i < regression.n_sampling; i++)
		{
			set_sampling(r(regression.mt_distribution));
			regression.gen_visualize_fit_state(true);
		}
		reset_sampling();
	}
#endif
	multiplot_gnuplot_script(regression.y_idx.size(), multiplot_step, header_names, y_var_idx, true, regression.n_sampling);

	{
		std::ofstream stream("Time_to_finish.txt");
		if (!stream.bad())
		{
			stream << "Time to finish:" << 0 << "[sec] = " << 0 << "[min]" << std::endl;
			stream.flush();
		}
	}
#ifdef USE_LIBTORCH
	if (use_libtorch)
	{
		torch_delete_model();
	}
#endif

end:;
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

