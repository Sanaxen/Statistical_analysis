#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/util/csvreader.h"



int main(int argc, char** argv)
{
	printf("formatting START\n");
	std::string csvfile("sample.csv");
	
	std::vector<std::string> x_var;
	std::string y_var = "";

	std::string nomalize = "";
	int fold_cv = 0;
	bool capture = false;
	bool heat_map = false;
	bool normalize = false;
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
		if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
			continue;
		}else
		if (argname == "--y_var") {
			y_var = argv[count + 1];
			continue;
		}else
		//n-fold cross validation
		if (argname == "--fold_cv") {
			fold_cv = atoi(argv[count + 1]);
			continue;
		}else
		if (argname == "--nomalize")
		{
			nomalize = argv[count + 1];
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
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

			for (int i = 0; i < x_var.size(); i++)
			{
				if (i <= x_var_idx.size())
				{
					printf("%s %d\n", x_var[i], x_var_idx[i]);
				}
			}
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
			else if ("\"" + y_var + "\"" == header_names[j])
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
	for (int i = 0; i < header_names.size(); i++)
	{
		std::replace(header_names[i].begin(), header_names[i].end(), ' ', '_');
	}

	Matrix<dnn_double> A;
	
	if (x_var.size() && y_var_idx >= 0)
	{
		std::vector<std::string> header_names_wrk = header_names;
		A = T.Col(y_var_idx);
		header_names[0] = header_names_wrk[y_var_idx];
		for (int i = 0; i < x_var.size(); i++)
		{
			A = A.appendCol(T.Col(x_var_idx[i]));
			header_names[i] = header_names_wrk[x_var_idx[i]];
		}
	}

	if(nomalize == "zscore")
	{
		std::vector<double> mean(A.n, 0.0);
		std::vector<double> sigma(A.n, 0.0);

		for (int i = 1; i < A.n; i++) 		{
			mean[i] = 0.0;
			for (int k = 0; k < A.m; k++)
			{
				mean[i] += A(k, i);
			}
		}
		for (int k = 1; k <A.n; k++)
		{
			mean[k] /= A.m;
		}
		for (int i = 1; i < A.n; i++)
		{
			sigma[i] = 0.0;
			for (int k = 0; k < A.m; k++)
			{
				sigma[i] += (A(k, i) - mean[i])*(A(k, i) - mean[i]);
			}
		}
		for (int k = 1; k < A.n; k++)
		{
			sigma[k] /= A.m;
		}
		for (int i = 1; i < A.n; i++)
		{
			for (int k = 0; k < A.m; k++)
			{
				A(k, i) = (A(k, i) - mean[i]) / (sigma[i] + 1.0e-10);
			}
		}
	}
	if (nomalize == "minmax")
	{
		std::vector<double> min_(A.n, 0.0);
		std::vector<double> maxmin_(A.n, 0.0);
		for (int k = 1; k < A.n; k++)
		{
			float max_value = -std::numeric_limits<float>::max();
			float min_value = std::numeric_limits<float>::max();
			for (int i = 0; i < A.m; i++)
			{
				if (max_value < A(i, k)) max_value = A(i, k);
				if (min_value > A(i, k)) min_value = A(i, k);
			}
			min_[k] = min_value;
			maxmin_[k] = (max_value - min_value);
			if (fabs(maxmin_[k]) < 1.0e-14)
			{
				min_[k] = 0.0;
				maxmin_[k] = max_value;
			}
		}
		for (int i = 0; i < A.m; i++)
		{
			for (int k = 1; k < A.n; k++)
			{
				A(i, k) = (A(i, k) - min_[k]) / maxmin_[k];
			}
		}
	}

	{
		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d,%s\n", y_var_idx, header_names[0].c_str());
		std::vector<int> var_indexs;
		int num = 0;
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[i + 1].c_str());
		}
		fclose(fp);
	}
	char fname[640];

	if (fold_cv <= 1)
	{
		sprintf(fname, "%s_libsvm.train", csvfile.c_str());
		FILE* fp = fopen(fname, "w");

		for (int i = 0; i < A.m; i++)
		{
			fprintf(fp, "%f", A(i, 0));
			for (int j = 1; j < A.n; j++)
			{
				fprintf(fp, " %d:%f", j, A(i, j));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(fname, "%s.no_header", csvfile.c_str());
		A.print_csv(fname);
		A.print_csv("Az_train.csv");
	}
	else
	{
		sprintf(fname, "%s_libsvm.train", csvfile.c_str());
		FILE* fp = fopen(fname, "w");

		int K = fold_cv;
		int test = (int)((float)A.m / (float)K + 0.5);
		int train = A.m - test;

		if (train <= 0)
		{
			train = A.m;
			test = 0;
		}
		for (int i = 0; i < train; i++)
		{
			fprintf(fp, "%f", A(i, 0));
			for (int j = 1; j < A.n; j++)
			{
				fprintf(fp, " %d:%f", j, A(i, j));
			}
			fprintf(fp, "\n");
		}
		fclose(fp);

		sprintf(fname, "%s.no_header", csvfile.c_str());
		A.print_csv(fname);
		A.print_csv("Aa_train.csv");

		if (test > 1)
		{
			sprintf(fname, "%s_libsvm.test", csvfile.c_str());
			fp = fopen(fname, "w");

			for (int i = train; i < A.m; i++)
			{
				fprintf(fp, "%f", A(i, 0));
				for (int j = 1; j < A.n; j++)
				{
					fprintf(fp, " %d:%f", j, A(i, j));
				}
				fprintf(fp, "\n");
			}
			fclose(fp);

			A.print_csv("A_test.csv");
			sprintf(fname, "%s.no_header", csvfile.c_str());
			A.print_csv(fname);
		}
	}
	fflush(stdout);


	printf("END\n\n");
	return 0;
}