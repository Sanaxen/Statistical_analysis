#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
#endif

int main(int argc, char** argv)
{
	CSVReader csv1("Boston.csv");
	Matrix<dnn_double> df = csv1.toMat();

	df = df.removeCol(0);
	std::vector<std::string> header = csv1.getHeader();
	header.erase(header.begin() + 0);

	Matrix<dnn_double>& y = df.Col(13);	//7
	y.print("", "%.3f ");

	Matrix<dnn_double>& X = df.removeCol(13,-1);
	X.print("", "%.3f ");
	printf("***%d\n", X.n);
#if 0
	Matrix<dnn_double> means = X.Mean();

	means.print("É ");
	Matrix<dnn_double>É–2 = X.Std(means);
	É–2.print("É–");

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			X(i, j) = (X(i, j)-means(0,j)) / É–2(0,j);
		}
	}
	X.print("", "%.3f ");
#endif

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH), 0);
		plot1.plot_lines(X, header);
		plot1.draw();

		gnuPlot plot2(std::string(GNUPLOT_PATH), 1);
		plot2.set_label_x("crim[î∆çﬂó¶]");
		plot2.set_label_y("mdev[èZëÓâøäiÇÃíÜâõíl]");
		plot2.scatter(X, y);
		plot2.scatter(X.Col(1), y);
		plot2.draw();

		gnuPlot plot3(std::string(GNUPLOT_PATH), 2);
		plot3.set_label_x("mdev[èZëÓâøäiÇÃíÜâõíl]");
		plot3.plot_histogram(Histogram(y,5));
		plot3.draw();
	}
#endif

	LassoRegression lasso_my(1.0, 1000, 0.0001);

	lasso_my.fit(X, y);
	lasso_my.report();

	printf("scikit-learn\n");
	printf("22.5328063241\n"
		"[-0.          0. - 0.          0. - 0.          2.71517992\n"
		"- 0. - 0. - 0. - 0. - 1.34423287  0.18020715\n"
		"- 3.54700664]\n");
	
	LassoRegression lasso_my2(0.0, 1000, 0.0001);

	lasso_my2.fit(X, y);
	lasso_my2.report();
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
	return 0;
}