#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/statistical/LinearRegression.h"
//#include "../../include/util/mathutil.h"
#include "../../include/util/csvreader.h"
#include "../../include/statistical/RegularizationRegression.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif
#include "../../include/util/cmdline_args.h"


int main(int argc, char** argv)
{
	printf("multiple_regression START\n");
	std::string csvfile("sample.csv");
	
	std::vector<std::string> x_var;
	std::string y_var = "";

	int test_mode = 0;
	bool capture = false;
	bool heat_map = false;
	bool normalize = false;
	int start_col = 0;
	bool header = false;

	int max_iteration = 1000;
	double tol = 0.001;
	double lambda1 = 0.001;
	double lambda2 = 0.001;
	std::string solver_name = "";

	int resp = commandline_args(&argc, &argv);
	if (resp == -1)
	{
		printf("command line error.\n");
		return -1;
	}

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
		if (argname == "--normalize") {
			normalize = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}else
		if (argname == "--heat_map") {
			heat_map = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}else
		if (argname == "--capture") {
			capture = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}else
		if (argname == "--test_mode") {
			test_mode = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--iter") {
			max_iteration = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--tol") {
			tol = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--L1") {
			lambda1 = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--L2") {
			lambda2 = atof(argv[count + 1]);
			continue;
		}
		else if (argname == "--solver") {
			solver_name = argv[count + 1];
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
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
	csv1.clear();

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
			else if ("\"" + header_names[j] + "\"" == y_var)
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

	for (int i = 0; i < header_names.size(); i++)
	{
		std::replace(header_names[i].begin(), header_names[i].end(), ' ', '_');
	}

	A.print("A");
	B.print("B");

	//A.print_csv("A.csv");
	//fflush(stdout);
	//Matrix<dnn_double> tmp = B;
	//tmp = tmp.appendCol(A);
	//tmp.print_csv("M.csv");
	{
		FILE* fp = fopen("debug_commandline.txt", "w");
		for (int i = 0; i < argc; i++)
		{
			fprintf(fp, "%s ", argv[i]);
		}
		fclose(fp);
	}

	multiple_regression mreg;

	mreg.set(A.n);
	mreg.fit(A, B);
	if (mreg.getStatus() != 0)
	{
		printf("error:%d\n", mreg.getStatus());
		return -1;
	}

	if (solver_name != "")
	{
		printf("³‘¥‰»\n");
		RegressionBase* solver = NULL;

		if (solver_name == "lasso")
		{
			solver = new LassoRegression(lambda1, max_iteration, tol);
		}
		if (solver_name == "elasticnet")
		{
			solver = new ElasticNetRegression(lambda1, lambda2, max_iteration, tol);
		}
		if (solver_name == "ridge")
		{
			solver = new RidgeRegression(lambda2, max_iteration, tol);
		}
		solver->y_var_idx = y_var_idx;
		solver->fit(A, B);
		solver->report(std::string("regularization.txt"), A, header_names, &B);

		mreg.bias = solver->coef(0, solver->coef.n-1);
		for (int i = 0; i < A.n; i++)
		{
			mreg.les.coef(i, 0) = solver->coef(0, i);
		}
		if (solver) delete solver;
	}
	
	if (test_mode)
	{
		FILE* fp = NULL;
		
		if (test_mode == 1)
		{
			fp = fopen("ln_regression_fit.model", "r");
		}
		if (test_mode == 2)
		{
			fp = fopen("dnn_regression_fit.model", "r");
		}
		if (fp)
		{
			double w;
			int n = 0;
			char buf[256];
			fgets(buf, 256, fp);
			sscanf(buf, "n %d\n", &n);
			if (n != A.n)
			{
				printf("model error or test data dimention error.\n");
				return -1;
			}
			fgets(buf, 256, fp);
			sscanf(buf, "bias %lf\n", &w); 
			mreg.bias = w;
			for (int i = 0; i < A.n; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "%lf\n", &w);
				mreg.les.coef(i, 0) = w;
			}
			fclose(fp);
		}
		if (fp == NULL)
		{
			printf("load model error\n");
			return -1;
		}
		mreg.fit(A, B, true, header_names);
	}

	if (A.n == 1)
	{
		mreg.report(std::string("regression1.txt"), header_names, 0.05);
	}
	else
	{
		mreg.report(std::string("regression.txt"), header_names, 0.05);
	}
	mreg.report(std::string(""), header_names, 0.05);
	
	std::vector<bool> zero_coef;
	if(!test_mode)
	{
		FILE* fp = fopen("ln_regression_fit.model", "w");
		if (fp)
		{
			fprintf(fp, "n %d\n", A.n);
			fprintf(fp, "bias %.16g\n", mreg.bias);
			for (int i = 0; i < A.n; i++)
			{
				fprintf(fp, "%.16g\n", mreg.les.coef(i, 0));
				if (fabs(mreg.les.coef(i, 0)) < 1.0e-6)
				{
					zero_coef.push_back(true);
				}
				else
				{
					zero_coef.push_back(false);
				}
			}
			fclose(fp);
		}
	}

	{
		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d,%s\n", y_var_idx, header_names[0].c_str());
		std::vector<int> var_indexs;
		int num = 0;
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			if (fp)
			{
				if (zero_coef.size() == 0)
				{
					fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[i + 1].c_str());
				}
				else
				{
					if (!zero_coef[i])
					{
						fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[i + 1].c_str());
					}
				}
			}
		}
		fclose(fp);
	}


	Matrix<dnn_double> cor = A.Cor();
	cor.print_csv("cor.csv");
#ifdef USE_GNUPLOT
	int win_size[2] = { 640 * 2, 480 * 2 };
	if (A.n > 1)
	{
		if (heat_map)
		{
			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 1);
			plot1.Heatmap(cor, header_names, header_names);
			plot1.draw();
		}
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

		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);
		if (capture)
		{
			plot1.set_capture(win_size, std::string("linear_regression.png"));
		}
		plot1.plot_lines2(x, line_header_names);

		plot1.scatter(T, x_var_idx[0], y_var_idx, 1, 30, header_names, 5);

		if (true)
		{
			plot1.probability_ellipse(T, x_var_idx[0], y_var_idx);
		}
		plot1.draw();
	}

	{
		Matrix<dnn_double> b(mreg.y_predict);
		Matrix<dnn_double> T = B;

		T = T.appendCol(b);
		std::vector<std::string> header_names(2);
		header_names[0] = "observed";
		header_names[1] = "predict";

		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 7);
		if (capture)
		{
			plot1.set_capture(win_size, std::string("observed_predict.png"));
		}
		plot1.scatter_xyrange_setting = false;
		plot1.scatter(T, 0, 1, 1, 30, header_names, 5);
		if (10)
		{
			double max_x = B.Max();
			double min_x = B.Min();
			double step = (max_x - min_x) / 5.0;
			Matrix<dnn_double> x(6, 2);
			Matrix<dnn_double> v(1, 1);
			for (int i = 0; i < 6; i++)
			{
				v(0, 0) = min_x + i*step;
				x(i, 0) = v(0, 0);
				x(i, 1) = v(0, 0);
			}
			plot1.set_label(0.5, 0.85, 1, "observed=predict");
			plot1.plot_lines2(x, header_names);
			plot1.draw();
		}
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
	printf("multiple_regression END\n\n");
	return 0;
}