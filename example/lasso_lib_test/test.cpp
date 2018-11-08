#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif

int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	Matrix<dnn_double> X;
	Matrix<dnn_double> y;
	std::vector<std::string> header_str;

	std::string y_var = "";
	int start_col = 0;
	bool header = false;
	int max_iteration = 1000;
	double tol = 0.001;
	double lambda1 = 0.001;
	double lambda2 = 0.001;
	std::string solver_name = "lasso";

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		else
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}else
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
		}
		else if (argname == "--iter") {
			max_iteration = atoi(argv[count + 1]);
		}
		else if (argname == "--tol") {
			tol = atof(argv[count + 1]);
		}
		else if (argname == "--L1") {
			lambda1 = atof(argv[count + 1]);
		}
		else if (argname == "--L2") {
			lambda2 = atof(argv[count + 1]);
		}
		else if (argname == "--solver") {
			solver_name = argv[count + 1];
		}
		else if (argname == "--y_var") {
			y_var = argv[count + 1];
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

	CSVReader csv1("Boston.csv");
	Matrix<dnn_double> df = csv1.toMat();

	df = df.removeCol(0);
	header_str = csv1.getHeader();
	header_str.erase(header_str.begin() + 0);

	y = df.Col(13);	//7
	y.print("", "%.3f ");

	X = df.removeCol(13, -1);
	X.print("", "%.3f ");
	printf("***%d\n", X.n);

	Matrix<dnn_double> T = y;
	T = T.appendCol(X);
	T.print_csv("sample.csv", header_str);

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH), 0);
		plot1.plot_lines(X, header_str);
		plot1.draw();

		gnuPlot plot2(std::string(GNUPLOT_PATH), 1);
		plot2.set_label_x("crim[î∆çﬂó¶]");
		plot2.set_label_y("mdev[èZëÓâøäiÇÃíÜâõíl]");
		plot2.scatter(T, 0, 13, 1, 30, header_str, 6, "rgbformulae 31, 13, 10");
		plot2.draw();


		gnuPlot plot3(std::string(GNUPLOT_PATH), 3);
		plot3.linecolor = "rgb \"light-cyan\"";
		plot3.set_label_x("mdev[èZëÓâøäiÇÃíÜâõíl]");
		plot3.plot_histogram(Histogram(y, 5), "mdev-histogram");
		plot3.draw();
	}
#endif

	LassoRegression lasso_my(1.0, 1000, 0.0001);

	lasso_my.fit(X, y);
	lasso_my.report(X, header_str);

	printf("scikit-learn\n");
	printf("22.5328063241\n"
		"[-0.          0. - 0.          0. - 0.          2.71517992\n"
		"- 0. - 0. - 0. - 0. - 1.34423287  0.18020715\n"
		"- 3.54700664]\n");
	

	LassoRegression lasso_my2(0.0, 1000, 0.0001);

	lasso_my2.fit(X, y);
	lasso_my2.report(X, header_str);
	printf("scikit-learn\n");
	printf("22.5328063241\n"
		"[-0.92906457  1.08263896  0.14103943  0.68241438 - 2.05875361  2.67687661\n"
		"0.01948534 - 3.10711605  2.6648522 - 2.07883689 - 2.06264585  0.85010886\n"
		"- 3.74733185]\n");

	/*
		scikit-learnÇÃLasso
		22.5328063241
		[-0.          0.         -0.          0.         -0.          2.71517992
		-0.         -0.         -0.         -0.         -1.34423287  0.18020715
		-3.54700664]
		*/

#ifdef USE_GNUPLOT
	{
		Matrix<dnn_double> yy = lasso_my.predict(X) - y;
		Matrix<dnn_double> yy2 = lasso_my2.predict(X) - y;

		gnuPlot plot1(std::string(GNUPLOT_PATH), 4);
		plot1.plot_lines(yy, std::vector<std::string>());
		plot1.plot_lines(yy2, std::vector<std::string>());
		plot1.draw();

		plot1 = gnuPlot(std::string(GNUPLOT_PATH), 4, true);
		plot1.plot_lines(yy, std::vector<std::string>());
		plot1.plot_lines(yy2, std::vector<std::string>());
		plot1.draw();
	}
#endif


	}
	else
	{
		fclose(fp);

		CSVReader csv1(csvfile, ',', header);

		Matrix<dnn_double> T = csv1.toMat_removeEmptyRow();
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

#ifdef USE_GNUPLOT
		{
			//gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 5, false);
			//
			//plot1.pointsize = 0.2;
			//plot1.multi_scatter(T, header_names, 2, "rgbformulae 4, 4, 4");
			//plot1.draw();
		}
#endif		
		int y_var_idx = 0;
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
		}

		y = T.Col(y_var_idx);
		X = T.removeCol(y_var_idx);

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
		{
			int k = 0;
			std::vector<std::string> header_names_tmp;

			header_names_tmp.push_back(header_names[y_var_idx]);
			for (int i = 0; i < T.n; i++, k++)
			{
				if (i == y_var_idx) continue;
				header_names_tmp.push_back(header_names[k]);
			}
			header_names = header_names_tmp;
		}
		printf("y_var:%s\n", header_names[0].c_str());

		X.print("X");
		y.print("y");
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
			solver = new RidgeRegression(lambda1, max_iteration, tol);
		}

		solver->fit(X, y);
		solver->report(X, header_names);

#if 0
#ifdef USE_GNUPLOT
		{
			std::vector<int> indexs;
			int num = 0;
			for (int i = 0; i < solver->coef.n - 1; i++)
			{
				if (fabs(solver->coef(0, i)) > 1.0e-6)
				{
					indexs.push_back(i + 1);
				}
			}

			gnuPlot plot2 = gnuPlot(std::string(GNUPLOT_PATH), 6, false);
			plot2.multi_scatter_for_list(indexs, T, header_names);
			plot2.draw();
		}
#endif
#endif
		if (solver) delete solver;
	}


	return 0;
}