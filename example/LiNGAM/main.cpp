#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
#endif

void read_csv(int n, char* filename, Matrix<dnn_double>& x)
{
	FILE* fp = fopen(filename, "r");
	if (fp == NULL)
	{
		return;
	}

	x = Matrix<dnn_double>(n, 1);
	for (int i = 0; i < n; i++)
	{
		dnn_double s = 0;
		if (i < n - 1)
		{
			int k = fscanf(fp, "%lf,", &s);
			if (k != 1)
			{
				printf("%d error\n", i);
			}
		}
		else
		{
			int k = fscanf(fp, "%lf", &s);
			if (k != 1)
			{
				printf("%d error\n", i);
			}
		}
		x(i, 0) = s;
	}
	fclose(fp);
}

//https://qiita.com/m__k/items/bd87c063a7496897ba7c
//LiNGAMƒ‚ƒfƒ‹‚Ì„’è•û–@‚É‚Â‚¢‚Ä
int main()
{
	Matrix<dnn_double> x, y, z, w;

	int len = 1000;
#if 10
	read_csv(len, "x.csv", x);
	read_csv(len, "y.csv", y);
	read_csv(len, "z.csv", z);
	read_csv(len, "w.csv", w);
#else
	CSVReader csv1("x.csv", ',', false);
	CSVReader csv2("y.csv", ',', false);
	CSVReader csv3("z.csv", ',', false);
	CSVReader csv4("w.csv", ',', false);
	x = csv1.toMat().transpose();
	y = csv2.toMat().transpose();
	z = csv3.toMat().transpose();
	w = csv4.toMat().transpose();

	len = x.m;
#endif

	Matrix<dnn_double>xs(len, 4);

	std::vector<std::string> names;
	names.resize(4);
	for (int i = 0; i < len; i++)
	{
		xs(i, 0) = y(i, 0);
		xs(i, 1) = z(i, 0);
		xs(i, 2) = w(i, 0);
		xs(i, 3) = x(i, 0);
		names[0] = "Y";
		names[1] = "Z";
		names[2] = "W";
		names[3] = "X";

		//xs(i, 0) = x(i, 0);
		//xs(i, 1) = y(i, 0);
		//xs(i, 2) = z(i, 0);
		//xs(i, 3) = w(i, 0);
		//names[0] = "X";
		//names[1] = "Y";
		//names[2] = "Z";
		//names[3] = "W";
	}
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(4);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");
	LiNGAM.remove_redundancy();
	LiNGAM.before_sorting();

	LiNGAM.B.print_e("B");

	LiNGAM.digraph(names, "digraph.txt");
	LiNGAM.report(names);

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH), 0, true);
		plot1.set_label_x("X");
		plot1.set_label_y("W");
		plot1.scatter(xs.Col(3), xs.Col(2), "X->W", 4, NULL);
		plot1.draw();

		system("cmd.exe /c gr.bat");
		plot1 = gnuPlot(std::string(GNUPLOT_PATH), 1);
		plot1.command(std::string("plot \"Digraph.png\" binary filetype=png with rgbimage\n"));
		plot1.draw();
	}
#endif
}