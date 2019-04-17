#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/statistical/LinearRegression.h"
#include "../../include/util/csvreader.h"
#include "../../include/util/swilk.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#endif

int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	Matrix<dnn_double> X;
	Matrix<dnn_double> y;
	std::vector<std::string> header_str;
	std::vector<std::string> x_var;

	int N = 20;
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
		else
		if (argname == "--x_var") {
			x_var.push_back( std::string(argv[count + 1]));
			continue;
		}else
		if (argname == "--N") {
			N = atoi(argv[count + 1]);
			continue;
		}else
		if (argname == "--palette") {
			palette = argv[count + 1];
			continue;
		}else {
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


	std::vector<int> x_var_idx;
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
			return -1;
		}
	}
	else
	{
		x_var.push_back("0");
		x_var_idx.push_back(0);
	}

	if (x_var.size())
	{
		std::vector<std::string> header_names_wrk = header_names;
		X = T.Col(x_var_idx[0]);
		header_names[0] = header_names_wrk[x_var_idx[0]];
		for (int i = 1; i < x_var.size(); i++)
		{
			X = X.appendCol(T.Col(x_var_idx[i]));
			header_names[i] = header_names_wrk[x_var_idx[i]];
		}
	}


	std::vector<int> shapiro_wilk_test;
	for (int k = 0; k < x_var.size(); k++)
	{
		printf("=========\n\n");
		printf("shapiro_wilk test(0.05) start\n");
		shapiro_wilk shapiro;
		{
			Matrix<dnn_double> tmp = X.Col(k);
			//tmp = tmp.whitening(tmp.Mean(), tmp.Std(tmp.Mean()));
			//tmp = tmp.Centers(tmp.Mean());

			int stat = shapiro.test(tmp);
			if (stat == 0)
			{
				printf("[%s]w:%-8.3f p_value:%-10.16f\n", header_names[k].c_str(),shapiro.get_w(), shapiro.p_value());
				if (shapiro.p_value() > 0.05)
				{
					shapiro_wilk_test.push_back(1);
				}
				else
				{
					shapiro_wilk_test.push_back(0);
				}
			}
			else
			{
				shapiro_wilk_test.push_back(0);
				printf("[%s] error shapiro.test=%d\n", header_names[k].c_str(), stat);
			}
		}
		printf("shapiro_wilk test end\n\n");
	}

#ifdef USE_GNUPLOT
	int win_size[2] = { 640 * 3,480 * 3 };
	if ( x_var.size()==1)
	{
		gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);
		if (palette != NULL)
		{
			plot1.set_palette(palette);
			plot1.histogram_gradation = true;
		}
		else
		{
			plot1.histogram_gradation = false;
		}
		plot1.set_capture(win_size, std::string("histogram.png"));
		plot1.plot_histogram(Histogram(X,N), (char*)header_names[0].c_str(), shapiro_wilk_test[0]);
		plot1.draw();
	}else
	{
		std::vector<int> flag(X.n, 0);
		gnuPlot plot1(std::string(GNUPLOT_PATH), 7);
		if (palette != NULL)
		{
			plot1.set_palette(palette);
			plot1.histogram_gradation = true;
		}
		else
		{
			plot1.histogram_gradation = false;
		}
		plot1.set_capture(win_size, std::string("histogram.png"));
		plot1.multi_histgram(std::string("histogram.png"), X, header_names, shapiro_wilk_test);
		plot1.draw();
	}
#endif		


	return 0;
}