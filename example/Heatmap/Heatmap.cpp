#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/statistical/LinearRegression.h"
//#include "../../include/util/mathutil.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif

int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	char* palette = "rgbformulae 21,22,23";
	int col_index = -1;
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
		}
		else
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
			continue;
		}
		else
		if (argname == "--col_index") {
			col_index = atoi(argv[count + 1]);
			continue;
		}
		else
		if (argname == "--palette") {
			palette = argv[count + 1];
			continue;
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

	std::vector<std::string> rows_names;
	
	if (col_index >= 0)
	{
		rows_names = csv1.ItemCol(col_index);
	}
	else
	{
		rows_names.resize(T.m);
		for (int i = 0; i < T.m; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			rows_names[i] = buf;
		}
	}


#ifdef USE_GNUPLOT
	gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 1, false);
	plot1.Heatmap(T, header_names, rows_names, palette);
	plot1.draw();
#endif
	return 0;
}