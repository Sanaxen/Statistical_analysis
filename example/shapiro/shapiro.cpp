#include <iostream>
#include <algorithm>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/util/csvreader.h"
#include "../../include/util/swilk.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#endif

int main()
{
	shapiro_wilk shapiro;

	Matrix<dnn_double> xx, yy;
	CSVReader csv1("normal.csv", ',', false);
	CSVReader csv2("comparison.csv", ',', false);

	xx = csv1.toMat().transpose();
	yy = csv2.toMat().transpose();

	if (shapiro.test(xx) == 0)
	{
		printf("w:%f p_value:%f\n", shapiro.get_w(), shapiro.p_value());
	}
	if (shapiro.test(yy) == 0)
	{
		printf("w:%f p_value:%f\n", shapiro.get_w(), shapiro.p_value());
	}

#ifdef USE_GNUPLOT
	std::vector<std::string> header_names;
	std::string name = "aaa";
	header_names.push_back(name);


	gnuPlot plot1(std::string(GNUPLOT_PATH), 3, false);
	plot1.plot_lines(xx, header_names);
	plot1.draw();

	gnuPlot plot2(std::string(GNUPLOT_PATH), 4, false);
	plot2.plot_histogram(Histogram(xx, 40));
	plot2.draw();

#endif

	CSVReader csv3("x.csv", ',', false);
	xx = csv3.toMat().transpose();
	if (shapiro.test(xx) == 0)
	{
		printf("w:%f p_value:%f\n", shapiro.get_w(), shapiro.p_value());
	}

	Chi_distribution chi_distribution(600);
	double chi_pdf = chi_distribution.p_value(0.05);
	printf("p value          :%f\n", chi_pdf);

	return 0;
}

