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
#define CNN_USE_TBB

#include "../../../include/util/dnn_util.hpp"
#include "../../../include/nonlinear/TimeSeriesRegression.h"
#include "../../../include/nonlinear/MatrixToTensor.h"

#include "../../../include/nonlinear/image_util.h"
#include "gen_test_data.h"
#include "../../include/util/cmdline_args.h"


//#define X_DIM	2
//#define Y_DIM	3

//#define IN_SEQ_LEN	15

int main(int argc, char** argv)
{
	int resp = commandline_args(&argc, &argv);
	if (resp == -1)
	{
		printf("ERROR:command line error.\n");
		return -1;
	}


	std::vector<std::string> x_var;
	std::vector<std::string> y_var;
	std::vector<std::string> xx_var;
	double xx_var_scale = 1.0;
	std::string t_var = "";
	int sequence_length = -1;
	int out_sequence_length = 1;
	std::string normalization_type = "";
	bool use_latest_observations = true;
	bool use_trained_scale = true;
	bool use_defined_scale = false;
	bool use_logdiffernce = false;
	int use_differnce = 0;
	bool use_differnce_auto_inv = false;
	bool use_differnce_output_only = false;
	bool use_libtorch = false;
	std::string device_name = "cpu";
	int multiplot_step = 3;
	std::string activation_fnc = "tanh";
	bool use_attention = true;

	int classification = -1;
	int read_max = -1;
	bool header = false;
	int start_col = 0;
	int x_dim = 0, y_dim = 0;
	int use_cnn = 1;
	int x_s = 0;
	int y_s = 0;
	bool test_mode = false;
	int ts_decomp_frequency = 0;
	bool dump_input = false;
	int fc_hidden_size = -1;
	std::string weight_init_type = "xavier";
	bool layer_graph_only = false;
	std::string timeformat = "";

	std::string data_path = "";
	int xvar_time_sift = 1;
	int target_position = 1;
	int mean_row = 1;

	std::string csvfile("sample.csv");
	std::string report_file("TimeSeriesRegression.txt");
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
		if (argname == "--use_attention") {
			use_attention = (atoi(argv[count + 1]) != 0) ? true : false;
		}else
		if (argname == "--activation_fnc") {
			activation_fnc = std::string(argv[count + 1]);
		}
		else
		if (argname == "--mean_row") {
			mean_row = atoi(argv[count + 1]);
			printf("mean_row:%d\n", mean_row);
		}else
		if (argname == "--target_position") {
			target_position = atoi(argv[count + 1]);
			printf("target_position:%d\n", target_position);
		}
		else if (argname == "--time_sift") {
			xvar_time_sift = atoi(argv[count + 1]);
			printf("xvar_time_sift:%d\n", xvar_time_sift);
		}
		else
		if (argname == "--multiplot_step") {
			multiplot_step = atoi(argv[count + 1]);
		}else
		if (argname == "--device_name") {
			device_name = std::string(argv[count + 1]);
		}
		else
			if (argname == "--use_libtorch") {
			use_libtorch = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		else
		if (argname == "--ts_decomp_frequency") {
			ts_decomp_frequency = atoi(argv[count + 1]);
		}
		else
		if (argname == "--dir") {
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
		}else
		if (argname == "--read_max") {
			read_max = atoi(argv[count + 1]);
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
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
		}
		if (argname == "--xx_var_scale") {
			xx_var_scale = atof(argv[count + 1]);
		}
		if (argname == "--xx_var") {
			xx_var.push_back(argv[count + 1]);
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
		if (argname == "--out_seq_len") {
			out_sequence_length = atoi(argv[count + 1]);
		}
		if (argname == "--normal")
		{
			normalization_type = argv[count + 1];
			printf("--normal %s\n", argv[count + 1]);
		}
		else if (argname == "--test_mode") {
			test_mode = (0 < atoi(argv[count + 1])) ? true : false;
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
		else if (argname == "--fc_hidden_size") {
			fc_hidden_size = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--layer_graph_only") {
			layer_graph_only = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--timeformat") {
			timeformat = std::string(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_trained_scale")
		{
			use_trained_scale = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--use_defined_scale")
		{
			use_defined_scale = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--use_differnce")
		{
			use_differnce = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--use_logdiffernce")
		{
			use_logdiffernce = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--use_differnce_output_only")
		{
			use_differnce_output_only = (0 < atoi(argv[count + 1])) ? true : false;
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
			int stat = filelist_to_csv(csvfile, data_path, test_mode==false, classification, header, normalization_type);
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
	csv1.clear();

	std::vector<int> xx_var_idx;
	std::vector<int> x_var_idx;
	std::vector<int> y_var_idx;
	int t_var_idx = -1;

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
				xx_var_idx.push_back(atoi(xx_var[i].c_str()));
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
			sprintf(buf, "\"%d\"", i+x_s);
			x_var.push_back(buf);
			x_var_idx.push_back(i+x_s);
		}
	}
	if (x_var.size() > 0 && x_dim > 0)
	{
		if (x_var.size()/*+yx_var.size()*/ != x_dim)
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
				if (x_var_idx[j] == i+y_s)
				{
					dup = true;
					break;
				}
			}
			if (!dup)
			{
				char buf[128];
				sprintf(buf, "\"%d\"", i+y_s);
				y_var.push_back(buf);
				y_var_idx.push_back(i+y_s);
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
		if (fp)fclose(fp);

		fp = fopen("select_variables2.dat", "w");
		for (int i = 0; i < xx_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", xx_var_idx[i], header_names[xx_var_idx[i]].c_str());
		}
		if (fp)fclose(fp);
	}

	std::vector<int> normalize_skilp;
	Matrix<dnn_double> x;
	if (x_dim > 0)
	{
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
		flag.resize(flg_idx + 1, 0);

		if (xx_var_idx.size())
		{
			for (int k = 0; k < xx_var_idx.size(); k++)
			{
				flag[xx_var_idx[k]] = 1;
			}
		}

		if (xx_var_idx.size())
		{
			for (int k = 0; k < xx_var_idx.size(); k++)
			{
				flag[xx_var_idx[k]] = 1;
			}
		}

		if (flag[x_var_idx[0]]) normalize_skilp.push_back(1);
		else normalize_skilp.push_back(0);
		x = z.Col(x_var_idx[0]);
		for (int i = 1; i < x_dim/*- yx_var.size()*/; i++)
		{
			x = x.appendCol(z.Col(x_var_idx[i]));
			if (flag[x_var_idx[i]]) normalize_skilp.push_back(1);
			else normalize_skilp.push_back(0);
		}
		
		x.print("x(pre_sift)");
		if (xvar_time_sift > 0)
		{
			//Shift the explanatory variable by one time ago
			Matrix<dnn_double> xx = x.Row(xvar_time_sift);
			for (int i = xvar_time_sift + 1; i < x.m - 1; i++)
			{
				xx = xx.appendRow(x.Row(i));
			}
			xx = xx.appendRow(x.Row(x.m - 1));
			x = xx;
			x.print("x");
		}
	}		
	if (xvar_time_sift < 0)
	{
		Matrix<dnn_double> xx = x;
		for (int i =  0; i < -xvar_time_sift; i++)
		{
			xx = xx.removeRow(xx.m - 1);
		}
		x = xx;
		x.print("x");
	}

	//Shift the Objective  variable by one time ago
	z.print("z");
	Matrix<dnn_double> y;
	y = z.Col(y_var_idx[0]);
	for (int i = 1; i < y_dim; i++)
	{
		y = y.appendCol(z.Col(y_var_idx[i]));
	}

	if (x_dim)
	{
		//Because the explanatory variable is shifted by (xvar_time_sift) unit, 
		//it is shortened by one unit.
		y.print("y(pre_sift)");
		if (xvar_time_sift > 0)
		{
			for (int i = 1; i <= xvar_time_sift; i++)
			{
				y = y.removeRow(y.m - 1);
			}
		}
		if (xvar_time_sift < 0)
		{
			for (int i = 1; i <= -xvar_time_sift; i++)
			{
				y = y.removeRow(0);
			}
		}
		printf("Because the explanatory variable is shifted by %d unit,\n", xvar_time_sift);
		printf("it is shortened by %d unit.\n", xvar_time_sift);
	}
	y.print("y");

	if (x_var.size())
	{
		for (int i = 0; i < x_var.size(); i++)
		{
			//y = y.appendCol(z.Col(x_var_idx[i]));
			y = y.appendCol(x.Col(i));
		}
	}
	y.print("y+yx");

	std::vector<std::string> timestamp;
	Matrix<dnn_double> tvar;
	if (t_var_idx >= 0)
	{
		tvar = z.Col(t_var_idx);

		if (csv1.timeform.size() && timeformat != "")
		{
			for (int i = 0; i < z.m; i++)
			{
				if (csv1.timeform[i*z.n + t_var_idx] == "")
				{
					char tmp[32];
					sprintf(tmp, "%d/01/01", (int)(tvar(i, 0)+0.5f));
					timestamp.push_back(tmp);
					printf("%s\n", tmp);
					continue;
				}
				//tvar(i, 0) = i;
				timestamp.push_back(csv1.timeform[i*z.n + t_var_idx]);
				printf("%s\n", csv1.timeform[i*z.n + t_var_idx].c_str());
			}
		}
		if (csv1.timeform.size() && timeformat == "")
		{
			for (int i = 0; i < z.m; i++)
			{
				char tmp[32];
				sprintf(tmp, "%d", i);
				timestamp.push_back(tmp);
				tvar(i, 0) = i;
			}
		}
	}
	else
	{
		tvar = Matrix<dnn_double>(y.m, 1);
		for (int i = 0; i < y.m; i++) tvar(i, 0) = i;
	}

	if(dump_input)
	{
		std::vector<std::string> hed;
		for (int i = 0; i < y_var_idx.size(); i++)
		{
			hed.push_back(header_names[y_var_idx[i]]);
		}
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			hed.push_back(header_names[x_var_idx[i]]);
		}

		Matrix<dnn_double> tmp_mat;
		tmp_mat = y;
		tmp_mat.print_csv("ts_input.csv", hed);
		exit(0);
	}

	//if(0)
	//{
	//	std::random_device rnd;     // 非決定的な乱数生成器を生成
	//	std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタ
	//	std::uniform_real_distribution<> rand(0.0, 1.0);
	//	FILE* fp = fopen("sample.csv", "w");

	//	float t = 0;
	//	float dt = 1;
	//	int k = 1;
	//	for (int i = 0; i < 10000; i++)
	//	{
	//		float r = rand(mt);
	//		if (r > 0.2)
	//		{
	//			fprintf(fp, "%f,%f,%f\n", t + dt*k, 0.0, 1.0);
	//		}
	//		else
	//		{
	//			fprintf(fp, "%f,%f,%f\n", t + dt*k, 1.0, 0.0);
	//		}
	//		k++;
	//	}
	//	fclose(fp);
	//}
	printf("sequence_length:%d\n", sequence_length);
	printf("out_sequence_length:%d\n", out_sequence_length);
	printf("x_dim:%d y_dim:%d\n", x_dim, y_dim);

	if (data_path != "" && read_max > 0)
	{
		std::mt19937 mt(read_max);
		std::uniform_int_distribution<int> rand_ts(0, x.m - 1);
		std:vector<int> index(x.m - 1, -1);

		Matrix<dnn_double> xx = x.Row(0);
		Matrix<dnn_double> yy = y.Row(0);
		Matrix<dnn_double> tt = tvar.Row(0);
		index[0] = 1;
		for (int i = 1; i < read_max-1; i++)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					xx = xx.appendRow(x.Row(i));
				}
#pragma omp section
				{
					yy = yy.appendRow(y.Row(i));
				}
#pragma omp section
				{
					tt = tt.appendRow(tvar.Row(i));
				}
			}
		}
		x = xx;
		y = yy;
		tvar = tt;
	}

	x.print();
	y.print();

	//if(0)
	//{
	//	int n = 3;
	//	std::vector<dnn_double> new_y;
	//	std::vector<dnn_double> new_x;
	//	std::mt19937 mt(1);
	//	std::normal_distribution<> norm(0.0, 1.0);
	//	for (int i = 0; i < y.m - 1; i++)
	//	{
	//		for (int k = 0; k < y.n; k++)
	//		{
	//			double ys = y(i + 1, k) - y(i, k);
	//			double yd = ys / (double)n;
	//			for (int j = 0; j < n; j++)
	//			{
	//				if (j > 0 && j < n - 1)
	//				{
	//					new_y.push_back(y(i, k) + j*yd + 0.3*fabs(ys)*norm(mt));
	//				}
	//				else
	//				{
	//					new_y.push_back(y(i, k) + j*yd);
	//				}
	//			}
	//			if (i == y.m - 1)
	//			{
	//				new_y.push_back(y(y.m - 1, k));
	//			}
	//		}
	//		for (int k = 0; k < x.n; k++)
	//		{
	//			double xs = x(i + 1, k) - x(i, k);
	//			double xd = xs / (double)n;
	//			for (int j = 0; j < n; j++)
	//			{
	//				new_x.push_back(x(i, k) + j*xd);
	//			}
	//			if (i == y.m - 1)
	//			{
	//				new_y.push_back(x(y.m - 1, k));
	//			}
	//		}
	//	}
	//	Matrix<dnn_double> yy(new_y.size() / y.n, y.n);
	//	Matrix<dnn_double> xx(new_x.size() / x.n, x.n);
	//	memcpy(yy.v, &(new_y[0]), sizeof(dnn_double)*new_y.size());
	//	memcpy(xx.v, &(new_x[0]), sizeof(dnn_double)*new_x.size());

	//	yy.print_csv("yy.csv");
	//	xx.print_csv("xx.csv");
	//	
	//	y = yy;
	//	x = xx;
	//	int sequence_length2 = (sequence_length - 1)*(n - 1);
	//	sequence_length = sequence_length2 - sequence_length2%sequence_length;
	//	//exit(0);
	//}

	if (ts_decomp_frequency)
	{
		y.print_csv("ts_decomp.csv");
		FILE* fp = fopen("ts_decomp.R", "w");
		if (fp)
		{
			printf("時系列数:%d\n", y.n);
			fprintf(fp, "library(tseries)\n");

			fprintf(fp,
				"tmp_ <- read.csv( \"ts_decomp.csv\", header=F, stringsAsFactors = F, na.strings=\"NULL\")\n");
			if (!test_mode)
			{
				fprintf(fp, "tmp_out_ <- train[1]\n");
			}
			else
			{
				fprintf(fp, "tmp_out_ <- test[1]\n");
			}
			fprintf(fp, "tmp_out_ <- data.frame(tmp_out_[c(1:nrow(tmp_)),])\n");
			fprintf(fp, "names(tmp_)[1]<-\"入力データ\"\n");
			fprintf(fp, "tmp_out_ <- cbind(tmp_out_, tmp_)\n");

			for (int i = 0; i < y.n; i++)
			{
				fprintf(fp,
					"xt<-ts(as.numeric(tmp_[, %d]), frequency = %d)\n"
					"xt.stl<-stl(xt, s.window = \"periodic\")\n"
					"season <-xt.stl$time.series[, 1]\n"
					"trend <-xt.stl$time.series[, 2]\n"
					"remainder <-xt.stl$time.series[, 3]\n"
					"write.csv(season, \"_season.csv\", row.names = FALSE)\n"
					"write.csv(trend, \"_trend.csv\", row.names = FALSE)\n"
					"write.csv(remainder, \"_remainder.csv\", row.names = FALSE)\n"
					"png(\"ts_decomp%d.png\", height = 960, width = 960)\n"
					"plot(decompose(xt))\n"
					"dev.off()\n"
					"#plot(decompose(xt))\n"
					"season <-read.csv(\"_season.csv\", header = T)\n"
					"trend <-read.csv(\"_trend.csv\", header = T)\n"
					"remainder <-read.csv(\"_remainder.csv\", header = T)\n"
					"x_ <-trend + remainder\n"
					"names(season)[1]<-\"周期的季節パターン\"\n"
					"names(trend)[1]<-\"長期の変化傾向\"\n"
					"names(remainder)[1]<-\"残りの不規則成分\"\n"
					"names(x_)[1]<-\"季節変動除去済\"\n"
					"tmp_out_ <-cbind(tmp_out_, season)\n"
					"tmp_out_ <-cbind(tmp_out_, trend)\n"
					"tmp_out_ <-cbind(tmp_out_, remainder)\n"
					"tmp_out_ <-cbind(tmp_out_, x_)\n"
					"\n"
					"test <-adf.test(xt)\n"
					"sink('Augmented-Dickey-Fuller-Test.txt')\n"
					"print(test)\n"
					"sink()\n"
					"\n\n",
					i + 1, ts_decomp_frequency, i + 1);
			}
			fprintf(fp, "ts_decomp <- tmp_out_\n");
		}
		else
		{
			printf("[ts_decomp.R] write Error.\n");
		}
		if ( fp )fclose(fp);
		return 0;
	}

	if (mean_row > 1)
	{
		printf("mean_row:%d\n", mean_row);
		y = MeanRow(mean_row, y);
		y.print_csv("mean_y.csv");

		if (x_dim >= 1)
		{
			x = MeanRow(mean_row, x);
			x.print_csv("mean_x.csv");
		}

	}
	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X, read_max);
	MatrixToTensor(y, Y, read_max);

#if 0
	{
		Matrix<double> c(20, 2);
		for (int i = 0; i < 2; i++)
		{
			for (int k = 0; k < 20; k++) c(k, i) = k+1;
		}

		const int lag = 3;
		c.print();
		tiny_dnn::tensor_t C;
		MatrixToTensor(c, C, read_max);

		tiny_dnn::tensor_t d = diff_vec(C, lag);

		Matrix<double> a;
		TensorToMatrix(d, a);

		a.print();
		
		d = diffinv_vec(C, d, lag);
		Matrix<double> b;
		TensorToMatrix(d, b);

		b.print();
		exit(0);
	}
#endif

#ifdef USE_LIBTORCH
	if (use_libtorch)
	{
		torch_train_init();
		torch_setDevice(device_name.c_str());
	}
#endif

	TimeSeriesRegression timeSeries(X, Y, normalize_skilp, xx_var_scale, y_dim, x_dim, normalization_type, classification, test_mode, use_trained_scale, use_defined_scale, use_differnce, use_logdiffernce, use_differnce_output_only);
	if (timeSeries.getStatus() == -1)
	{
		if (classification < 2)
		{
			printf("class %.3f %.3f\n", timeSeries.class_minmax[0], timeSeries.class_minmax[1]);
		}
		return -1;
	}
	if (use_differnce_output_only)
	{
		return 0;
	}
	//printf("TimeSeriesRegression constract\n"); fflush(stdout);

#ifdef USE_LIBTORCH
	timeSeries.use_libtorch = use_libtorch;
	timeSeries.device_name = device_name;
#endif
	timeSeries.timeformat = timeformat;
	timeSeries.timestamp = timestamp;
	timeSeries.timevar = tvar;
	timeSeries.x_dim = x_dim;
	timeSeries.y_dim = y_dim;
	timeSeries.tolerance = 0.009;
	timeSeries.learning_rate = 1.0;
	timeSeries.visualize_loss(10);
	timeSeries.plot = 1;
	timeSeries.test_mode = test_mode;
	timeSeries.weight_init_type = weight_init_type;
	timeSeries.normalize_skilp = normalize_skilp;
	timeSeries.xx_var_scale = xx_var_scale;
	timeSeries.target_position = target_position;
	timeSeries.activation_fnc = activation_fnc;
	timeSeries.use_attention = use_attention;

	int n_layers = -1;
	int n_rnn_layers = -1;
	int hidden_size = -1;
	bool capture = false;
	float test = 0.4;

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--use_attention") {
			continue;
		}
		if (argname == "--activation_fnc") {
			continue;
		}
		if (argname == "--mean_row") {
			continue;
		}
		if (argname == "--target_position") {
			continue;
		}
		if (argname == "--time_sift") {
			continue;
		}
		if (argname == "--multiplot_step") {
			continue;
		}
		if (argname == "--device_name") {
			continue;
		}
		if (argname == "--use_libtorch") {
			continue;
		}
		if (argname == "--dir") {
			continue;
		}
		if (argname == "--layer_graph_only") {
			continue;
		}
		else
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
		if (argname == "--xx_var_scale") {
			continue;
		}
		else if (argname == "--xx_var") {
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
		else if (argname == "--test_mode") {
			continue;
		}
		else if (argname == "--fc_hidden_size") {
			continue;
		}
		else if (argname == "--timeformat") {
			continue;
		}
		else if (argname == "--use_differnce")
		{
			continue;
		}
		else if (argname == "--use_logdiffernce")
		{
			continue;
		}
		else if (argname == "--use_trained_scale")
		{
			continue;
		}
		else if (argname == "--use_defined_scale")
		{
			continue;
		}
		else if (argname == "--state_reset_mode")
		{
			timeSeries.state_reset_mode = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--batch_shuffle")
		{
			timeSeries.batch_shuffle = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--use_differnce_auto_inv")
		{
			timeSeries.use_differnce_auto_inv = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else if (argname == "--weight_init_type") {
			timeSeries.weight_init_type = std::string(argv[count + 1]);
			continue;
		}
		else if (argname == "--bptt_max") {
			timeSeries.n_bptt_max = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--clip_grad") {
			timeSeries.clip_gradients = atof(argv[count + 1]);
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
		else if (argname == "--test") {
			test = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--epochs") {
			timeSeries.n_train_epochs = atoi(argv[count + 1]);
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
		if (argname == "--out_seq_len") {
			out_sequence_length = atoi(argv[count + 1]);
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
		else if (argname == "--use_latest_observations")
		{
			timeSeries.use_latest_observations = (0 < atoi(argv[count + 1])) ? true : false;
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}

	timeSeries.sequence_length = sequence_length;
	timeSeries.out_sequence_length = out_sequence_length;
	timeSeries.data_set(test);
	timeSeries.layer_graph_only = layer_graph_only;

	timeSeries.header = header_names;
	timeSeries.x_idx = x_var_idx;
	timeSeries.y_idx = y_var_idx;

	timeSeries.fc_hidden_size = fc_hidden_size;

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
		<< "sequence_length :   " << timeSeries.sequence_length << std::endl
		<< "out_sequence_length :   " << timeSeries.out_sequence_length << std::endl
		<< "optimizer       :   " << timeSeries.opt_type << std::endl
		<< "n_rnn_layers    :   " << n_rnn_layers << std::endl
		<< "n_layers        :   " << n_layers << std::endl
		<< "test_mode       :   " << timeSeries.test_mode << std::endl
		<< "n_bptt_max      :  " << timeSeries.n_bptt_max << std::endl
		<< "classification  : " << timeSeries.classification << std::endl
		<< "dropout         : " << timeSeries.dropout << std::endl
		<< "clip_gradients  : " << timeSeries.clip_gradients << std::endl
		<< "timeformat      : " << timeSeries.timeformat << std::endl
		<< "fc_hidden_size  : " << fc_hidden_size << std::endl
		<< "weight_init_type       : " << timeSeries.weight_init_type << std::endl
		<< "use_latest_observations: " << timeSeries.use_latest_observations << std::endl
		<< "use_trained_scale: " << timeSeries.use_trained_scale << std::endl
		<< "use_defined_scale: " << timeSeries.use_defined_scale << std::endl
		<< "use_differnce: " << timeSeries.use_differnce << std::endl
		<< "use_logdiffernce: " << timeSeries.use_logdiffernce << std::endl
		<< "use_differnce_auto_inv: " << timeSeries.use_differnce_auto_inv << std::endl
		<< "device_name: " << timeSeries.device_name << std::endl
		<< "use_libtorch: " << timeSeries.use_libtorch << std::endl
		<< "state_reset_mode: " << timeSeries.state_reset_mode << std::endl
		<< "batch_shuffle: " << timeSeries.batch_shuffle << std::endl
		<< "sift_time:" << xvar_time_sift << std::endl
		<< "target_position: " << target_position << std::endl
		<< "use_attention: " << use_attention << std::endl
		<< "dump_input      : " << dump_input << std::endl
		
		<< std::endl;
//
	{
		FILE* fp = fopen("debug_commandline.txt", "w");
		for (int i = 0; i < argc; i++)
		{
			fprintf(fp, "%s ", argv[i]);
		}
		fclose(fp);
	}

	multiplot_gnuplot_script_ts(y_var_idx.size(), multiplot_step, header_names, y_var_idx, timeformat, false);

	timeSeries.fit(sequence_length, n_rnn_layers, n_layers, hidden_size);
	if (layer_graph_only)
	{
		goto end;
	}

	//printf("timeSeries.report start\n"); fflush(stdout);
	timeSeries.report(0.05, report_file);
	if (classification < 2)
	{
		timeSeries.visualize_observed_predict_plot = true;
		timeSeries.visualize_observed_predict();
	}
	//printf("timeSeries.report end\n"); fflush(stdout);
	//printf("timeSeries.gen_visualize_fit_state start\n"); fflush(stdout);
	timeSeries.gen_visualize_fit_state();
	//printf("timeSeries.gen_visualize_fit_state end\n"); fflush(stdout);
	multiplot_gnuplot_script_ts(y_var_idx.size(), multiplot_step, header_names, y_var_idx, timeformat, true);

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

