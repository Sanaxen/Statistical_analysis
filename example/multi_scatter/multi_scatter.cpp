#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/statistical/LinearRegression.h"
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

	bool ellipse = false;
	bool linear_regression = false;
	char* palette = NULL;
	int col1=-1, col2=-1;
	int start_col = 0;
	bool header = false;

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
		else
		if (argname == "--col1") {
			col1 = atoi(argv[count + 1]);
		}
		else
		if (argname == "--col2") {
			col2 = atoi(argv[count + 1]);
		}
		else
		if (argname == "--palette") {
			palette = argv[count + 1];
		}
		else
		if (argname == "--ellipse") {
			ellipse = argv[count + 1];
		}
		else
			if (argname == "--linear_regression") {
				linear_regression = atoi(argv[count + 1]) == 0 ? false : true;
			}
			else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
	}

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


	if ( col1 < 0 && col2 < 0)
	{
#ifdef USE_GNUPLOT
		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 5, false);
		plot1.multi_scatter(T, header_names, 2, palette);
		plot1.draw();
#endif		
		return 0;
	}


	if (col1 < 0) col1 = 1;
	if (col2 < 0) col2 = 0;
	X = T.Col(col1);
	y = T.Col(col2);

	X.print("X");
	y.print("y");

	printf("=========\n\n");
#ifdef USE_GNUPLOT
	{
		Matrix<dnn_double> a = T.Col(col1);
		a = a.appendCol(T.Col(col2));
		//a.print();

		Matrix<dnn_double>& cor = a.Cor();
		cor.print();
		char text[32];
		sprintf(text, "r=%.3f", cor(0, 1));


		int NN = 100;
		Matrix<dnn_double> XYeli(NN + 1, 2);
		if(ellipse )
		{
			Matrix<dnn_double>& x = T.Col(col1);
			x = x.appendCol(T.Col(col2));

			Matrix<dnn_double>& xmean = x.Mean();
			Matrix<dnn_double> xCovMtx = x.Cov(xmean);

			xmean.print("xmean");
			xCovMtx.print("xCovMtx");

			eigenvalues eig;
			eig.set(xCovMtx);
			eig.calc(true);

			Matrix<dnn_double> xy(NN+1, 2);
			double s = 2.0*M_PI / NN;
			Matrix<dnn_double>& lambda = eig.getRealValue();

			lambda.print("lambda");
			for (int i = 0; i <= NN; i++)
			{
				xy(i, 0) = sqrt(lambda.v[0])*cos(i*s);
				xy(i, 1) = sqrt(lambda.v[1])*sin(i*s);
			}
			std::vector<Matrix<dnn_double>>&vecs0 = eig.getRightVector(0);
			std::vector<Matrix<dnn_double>>&vecs1 = eig.getRightVector(1);

			vecs0[0].print("Re vecs0[0]");
			vecs0[1].print("Im vecs0[1]");
			vecs1[0].print("Re vecs1[0]");
			vecs1[1].print("Im vecs1[1]");
			Matrix<dnn_double> stdElc(NN+1, 2);

			for (int i = 0; i <= NN; i++)
			{
				stdElc(i, 0) = vecs0[0](0, 0)*xy(i, 0) + vecs1[0](0, 0)*xy(i, 1);
				stdElc(i, 1) = vecs0[0](1, 0)*xy(i, 0) + vecs1[0](1, 0)*xy(i, 1);
			}
			stdElc.print("stdElc");

			int N = x.m;
			int c1 = 2 * (N - 1) / N * (N + 1) / (N - 2);

			F_distribution f_distribution(2, N - 2);
			double F95 = sqrt(c1*f_distribution.p_value(0.05));

			printf("F95:%f\n", F95);
			for (int i = 0; i <= NN; i++)
			{
				XYeli(i, 0) = F95*stdElc(i, 0) + xmean.v[0];
				XYeli(i, 1) = F95*stdElc(i, 1) + xmean.v[1];
			}
		}
		if (linear_regression)
		{
			multiple_regression mreg;

			mreg.set(X.n);
			mreg.fit(X, y);


			double max_x = X.Max();
			double min_x = X.Min();
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
			
			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6, false);
			plot1.set_label(0.5, 0.85, 1, text);
			plot1.plot_lines2(x, line_header_names);

			plot1.scatter(T, col1, col2, header_names, 7, palette);

			if (ellipse)
			{
				plot1.plot_lines2d(XYeli, std::string("95% confidence ellipse / probability ellipse"));
			}
			plot1.draw();
		}
		else
		{
			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6, false);
			plot1.set_label(0.5, 0.5, 1, text);
			plot1.scatter(T, col1, col2, header_names, 6, palette);
			plot1.draw();
		}
	}
#endif		


	return 0;
}