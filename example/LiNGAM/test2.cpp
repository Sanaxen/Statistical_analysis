#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

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
		double s = 0;
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
	Matrix<dnn_double> x, y, z, u, v, w;

	int len = 1000;
#if 0
	read_csv(len, "x.csv", x);
	read_csv(len, "y.csv", y);
	read_csv(len, "z.csv", z);
	read_csv(len, "u.csv", u);
	read_csv(len, "v.csv", v);
	read_csv(len, "w.csv", w);
#else
	CSVReader csv1("x.csv", ',', false);
	CSVReader csv2("y.csv", ',', false);
	CSVReader csv3("z.csv", ',', false);
	CSVReader csv4("u.csv", ',', false);
	CSVReader csv5("v.csv", ',', false);
	CSVReader csv6("w.csv", ',', false);
	x = csv1.toMat().transpose();
	y = csv2.toMat().transpose();
	z = csv3.toMat().transpose();
	u = csv4.toMat().transpose();
	v = csv5.toMat().transpose();
	w = csv6.toMat().transpose();

	len = x.m;
#endif

	Matrix<dnn_double>xs(len, 6);

	std::vector<std::string> names;
	names.resize(6);
	for (int i = 0; i < len; i++)
	{
		xs(i, 0) = z(i, 0);
		xs(i, 1) = y(i, 0);
		xs(i, 2) = v(i, 0);
		xs(i, 3) = w(i, 0);
		xs(i, 4) = x(i, 0);
		xs(i, 5) = u(i, 0);
		names[0] = "Z";
		names[1] = "Y";
		names[2] = "V";
		names[3] = "W";
		names[4] = "X";
		names[5] = "U";

	}
	//xs.print_csv("sample2.csv");
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(6);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");
	fflush(stdout);
	LiNGAM.remove_redundancy();
	LiNGAM.before_sorting();
	LiNGAM.B.print_e("B");

	std::vector<std::string> x_var;

	std::vector<int> residual_flag(xs.n, 0);
	LiNGAM.digraph(names, x_var, residual_flag, "digraph.txt");
	LiNGAM.report(names);

#ifdef USE_GNUPLOT
	{
		gnuPlot plot1(std::string(GNUPLOT_PATH), 0, true);
		plot1.set_label_x("X");
		plot1.set_label_y("W");
		plot1.scatter(xs, 4, 3, 1, 30, names);
		plot1.draw();

		system("cmd.exe /c gr.bat");
		plot1 = gnuPlot(std::string(GNUPLOT_PATH), 1);
		plot1.command(std::string("plot \"Digraph.png\" binary filetype=png with rgbimage\n"));
		plot1.draw();
	}
#endif
}