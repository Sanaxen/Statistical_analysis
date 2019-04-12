#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/statistical/LinearRegression.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#endif

int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	Matrix<dnn_double> X;
	Matrix<dnn_double> y;
	std::vector<std::string> header_str;

	bool back_color_dark = false;
	bool capture = false;
	int win_size[2] = { -1,-1 };
	int grid = 30;
	float pointsize = 1.0;
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
			ellipse = atoi(argv[count + 1]) == 0 ? false : true;
		}
		else
		if (argname == "--pointsize") {
			pointsize = atof(argv[count + 1]);
		}
		else
		if (argname == "--grid") {
			grid = atoi(argv[count + 1]);
		}
		else
		if (argname == "--win_size") {
			sscanf(argv[count + 1], "%d,%d", win_size, win_size + 1);
		}
		else
		if (argname == "--linear_regression") {
			linear_regression = atoi(argv[count + 1]) == 0 ? false : true;
		}
		else
		if (argname == "--capture") {
			capture = atoi(argv[count + 1]) == 0 ? false : true;
		}
		else
		if (argname == "--back_color_dark") {
			back_color_dark = atoi(argv[count + 1]) == 0 ? false : true;
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
		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 5);
		if (capture)
		{
			plot1.multi_scatter(T, header_names, true, win_size[0], win_size[1], 2, palette);
			plot1.draw(0);
		}
		else
		{
			plot1.multi_scatter(T, header_names, false, win_size[0], win_size[1], 2, palette);
			plot1.draw();
		}
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
		cor.print_csv("cor.csv");
		char text[32];
		sprintf(text, "r=%.3f", cor(0, 1));


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
			
			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);
			
			plot1.set_capture(win_size, std::string("scatter.png"));

			plot1.set_label(0.5, 0.85, 1, text);
			plot1.plot_lines2(x, line_header_names);
			plot1.scatter_back_color_dark_mode = back_color_dark;
			plot1.scatter(T, col1, col2, pointsize, grid, header_names, 5, palette);

			if (ellipse)
			{
				plot1.probability_ellipse(T, col1, col2);
			}
			plot1.draw();
		}
		else
		{
			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);
			plot1.set_capture(win_size, std::string("scatter.png"));

			plot1.set_label(0.5, 0.5, 1, text);
			plot1.scatter_back_color_dark_mode = back_color_dark;
			plot1.scatter(T, col1, col2, pointsize, grid, header_names, 5, palette);
			if (ellipse)
			{
				plot1.probability_ellipse(T, col1, col2);
			}

			plot1.draw();
		}
	}
#endif		


	return 0;
}