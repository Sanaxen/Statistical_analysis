#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
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
	int auto_search = 0;
	std::string solver_name = "lasso";

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
			continue;
		}
		else
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}else
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
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
		else if (argname == "--auto") {
			auto_search = atoi(argv[count + 1]);
			continue;
		}
		else if (argname == "--solver") {
			solver_name = argv[count + 1];
			continue;
		}
		else if (argname == "--y_var") {
			y_var = argv[count + 1];
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
		lasso_my.report(std::string(""), X, header_str);

		printf("scikit-learn\n");
		printf("22.5328063241\n"
			"[-0.          0. - 0.          0. - 0.          2.71517992\n"
			"- 0. - 0. - 0. - 0. - 1.34423287  0.18020715\n"
			"- 3.54700664]\n");


		LassoRegression lasso_my2(0.0, 1000, 0.0001);

		lasso_my2.fit(X, y);
		lasso_my2.report(std::string(""), X, header_str);
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

			plot1 = gnuPlot(std::string(GNUPLOT_PATH), 4);
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
			solver = new RidgeRegression(lambda2, max_iteration, tol);
		}
		solver->y_var_idx = y_var_idx;
		solver->fit(X, y);
		solver->report(std::string("regularization.rep"), X, header_names, &y);

		double auto_search_aic = 0;
		double auto_search_l1 = 0;
		double auto_search_l2 = 0;

		std::vector<double> aic_list;
		std::vector<double> l1_list;
		std::vector<double> l2_list;
		if (auto_search < 2) auto_search = 3;
		if (auto_search)
		{
			int kmax = auto_search;
			double l1_max = 10.0;
			double l2_max = 10.0;
			double l1_min = 0.0;
			double l2_min = 0.0;
			for (int k = 0; k < kmax; k++)
			{
				int n = 5;
				double stp1 = l1_max / (double)n;
				double stp2 = l2_max / (double)n;

				Matrix<dnn_double> aic(n + 1, n + 1);
				Matrix<dnn_double> l1(n + 1, n + 1);
				Matrix<dnn_double> l2(n + 1, n + 1);
				Matrix<dnn_double> se(n + 1, n + 1);

				aic = aic.zeros(n + 1, n + 1);
				l1 = l1.zeros(n + 1, n + 1);
				l2 = l2.zeros(n + 1, n + 1);
				se = se.zeros(n + 1, n + 1);

				int m = 0;
				if (solver_name == "elasticnet") m = n;

#pragma omp parallel for
				for (int i = 0; i <= n; i++)
				{
					for (int j = 0; j <= m; j++)
					{
						RegressionBase* solver = NULL;
						if (solver_name == "lasso")
						{
							solver = new LassoRegression(l1_min + stp1*i, max_iteration, tol);
						}
						if (solver_name == "ridge")
						{
							solver = new RidgeRegression(l1_min + stp1*i, max_iteration, tol);
						}
						if (solver_name == "elasticnet")
						{
							solver = new ElasticNetRegression(l1_min + stp1*i, l2_min + stp2*j, max_iteration, tol);
						}
						solver->fit(X, y);
						aic(i, j) = solver->calc_AIC(X, y);
						se(i, j) = solver->se;
						l1(i, j) = l1_min + stp1*i;
						l2(i, j) = l2_min + stp2*j;

						if (solver) delete solver;
					}
				}

				aic_list.clear();
				l1_list.clear();
				l2_list.clear();
				for (int i = 0; i <= n; i++)
				{
					for (int j = 0; j <= m; j++)
					{
						aic_list.push_back(aic(i, j));
						l1_list.push_back(l1(i, j));
						l2_list.push_back(l2(i, j));
					}
				}

				int id1 = -1;
				int id2 = -1;
				double min_aic = 9999999.0;
				for (int i = 0; i <= n; i++)
				{
					for (int j = 0; j <= m; j++)
					{
						if (min_aic > aic(i,j))
						{
							min_aic = aic(i, j);
							id1 = i;
							id2 = j;
						}
						//printf("L1:%.3f SE:%.3f AIC:%.3f\n", l1(i, j), se(i, j), aic(i, j));
					}
				}
				l1_max = l1(id1, id2) + stp1;
				l2_max = l2(id1, id2) + stp2;
				l1_min = l1(id1, id2) - stp1;
				l2_min = l2(id1, id2) - stp2;
				if (l1_min < 0) l1_min = 0.0;
				if (l2_min < 0) l2_min = 0.0;
				
				auto_search_aic = aic(id1, id2);
				auto_search_l1 = l1(id1, id2);
				auto_search_l2 = l2(id1, id2);
			}

			{
				FILE* fp = fopen("AIC_list.dat", "w");
				if (fp)
				{
					for (int i = 0; i < aic_list.size(); i++)
					{
						fprintf(fp, "%.4f,%.4f,%.4f\n",
							aic_list[i], l1_list[i], l2_list[i]);
					}
					fclose(fp);
				}
#ifdef USE_GNUPLOT
				Matrix<dnn_double> aic_mat(aic_list);
				Matrix<dnn_double> l1_mat(l1_list);
				Matrix<dnn_double> l2_mat(l2_list);

				char text[128];
				if (solver_name == "lasso")
				{
					sprintf(text, "AIC:%.2f L1:%.3f", auto_search_aic,auto_search_l1);
				}
				if (solver_name == "ridge")
				{
					sprintf(text, "AIC:%.2f L2:%.3f", auto_search_aic, auto_search_l2);
				}
				if (solver_name == "elasticnet")
				{
					sprintf(text, "AIC:%.2f L1:%.3f L2:%.3f", auto_search_aic, auto_search_l1, auto_search_l2);
				}
				std::vector<std::string> header_names_tmp;
				Matrix<dnn_double> T;
				header_names_tmp.push_back("AIC");
				if (solver_name == "lasso" || solver_name == "elasticnet")
				{
					sprintf(text, "AIC:%.2f L1:%.3f", auto_search_aic, auto_search_l1);
					header_names_tmp.push_back("L1");
					T = aic_mat;
					T = T.appendCol(l1_mat);
				}
				if (solver_name == "ridge")
				{
					sprintf(text, "AIC:%.2f L2:%.3f", auto_search_aic, auto_search_l2);
					header_names_tmp.push_back("L2");
					T = aic_mat;
					T = T.appendCol(l2_mat);
				}

				int win_size[2] = { 640 * 3, 480 * 3 };
				char* palette = "rgbformulae 22, 13, -31";
				int grid = 30;
				float pointsize = 1.0;

				if (solver_name == "lasso" || solver_name == "ridge" || solver_name == "elasticnet")
				{

					//palette = "defined"; //"rgbformulae 7,5,15";
					gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 8);
					plot1.set_label(0.8, 0.95, 1, text);
					plot1.scatter_density_mode = false;
					plot1.set_palette(palette);
					plot1.set_capture(win_size, std::string("AIC_list.png"));
					plot1.scatter(T, 1, 0, pointsize, grid, header_names_tmp, 5, palette);
					plot1.draw();
				}
				if (solver_name == "elasticnet")
				{
					sprintf(text, "AIC:%.2f L2:%.3f", auto_search_aic, auto_search_l2);
					header_names_tmp.clear();
					header_names_tmp.push_back("AIC");
					header_names_tmp.push_back("L2");

					T = aic_mat;
					T = T.appendCol(l2_mat);
					gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 9);
					plot1.set_label(0.8, 0.95, 1, text);
					plot1.scatter_density_mode = false;
					plot1.set_palette(palette);
					plot1.set_capture(win_size, std::string("AIC_list2.png"));
					plot1.scatter(T, 1, 0, pointsize, grid, header_names_tmp, 5, palette);
					plot1.draw();
				}
#endif		
			}

			FILE* fp = fopen("auto_search.dat", "w");
			if (fp)
			{
				if (solver_name == "lasso")
				{
					fprintf(fp, "%.3f,%.3f,%.3f\n", auto_search_l1, 0, auto_search_aic);
				}else
				if (solver_name == "ridge")
				{
					fprintf(fp, "%.3f,%.3f,%.3f\n", 0, auto_search_l1, auto_search_aic);
				}
				else
				{
					fprintf(fp, "%.3f,%.3f,%.3f\n", auto_search_l1, auto_search_l2, auto_search_aic);
				}
				fclose(fp);
			}
			if (solver_name == "lasso")
			{
				printf("%.3f,%.3f,%.3f\n", auto_search_l1, 0, auto_search_aic);
			}else
			if (solver_name == "ridge")
			{
				printf("%.3f,%.3f,%.3f\n", 0, auto_search_l1, auto_search_aic);
			}else
			{
				printf("%.3f,%.3f,%.3f\n", auto_search_l1, auto_search_l2, auto_search_aic);
			}
		}
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