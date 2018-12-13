#include <iostream>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif

int main(int argc, char** argv)
{
	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;

	std::string csvfile("2-3c.csv");

	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
			continue;
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
	}

	CSVReader csv(csvfile, ',', header);
	x = csv.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			x = x.removeCol(0);
		}
	}
	std::vector<std::string> header_names;
	header_names.resize(x.n);
	if (header && csv.getHeader().size() > 0)
	{
		for (int i = 0; i < x.n; i++)
		{
			header_names[i] = csv.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < x.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}

	variablesNum = x.n;
	n = x.m;
	x.print();

	PCA pca2;
	pca2.set(variablesNum);

	int stat = pca2.fit(x, true);

	pca2.Report();

	pca2.principal_component().print("Žå¬•ª");
	pca2.principal_component().print_csv("output1.csv");

	stat = pca2.fit(x, true, false);

	pca2.Report();

	pca2.principal_component().print_csv("output2.csv");

#ifdef USE_GNUPLOT

	gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6, false);
	header_names[0] = "First principal component";
	header_names[1] = "Second principal component";
	plot1.scatter(pca2.principal_component(), 0, 1, 1, 30, header_names, 6);
	plot1.probability_ellipse(pca2.principal_component(), 0, 1);
	plot1.draw();
#endif

}

