#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
#endif


//https://qiita.com/m__k/items/bd87c063a7496897ba7c
//LiNGAMƒ‚ƒfƒ‹‚Ì„’è•û–@‚É‚Â‚¢‚Ä
int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
	}


	FILE* fp = fopen(csvfile.c_str(), "r");
	if (fp == NULL)
	{
		Matrix<dnn_double> x, y, z, w;
		CSVReader csv1("x.csv", ',', false);
		CSVReader csv2("y.csv", ',', false);
		CSVReader csv3("z.csv", ',', false);
		CSVReader csv6("w.csv", ',', false);
		x = csv1.toMat().transpose();
		y = csv2.toMat().transpose();
		z = csv3.toMat().transpose();
		w = csv6.toMat().transpose();
		Matrix<dnn_double>xs(x.m, 4);

		for (int i = 0; i < x.m; i++)
		{
			xs(i, 0) = x(i, 0);
			xs(i, 1) = y(i, 0);
			xs(i, 2) = z(i, 0);
			xs(i, 3) = w(i, 0);
		}
		xs.print_csv("sample.csv");
		header = false;
	}

	CSVReader csv1(csvfile, ',', header);
	Matrix<dnn_double> xs = csv1.toMat();


	Lingam LiNGAM;

	size_t max_ica_iteration = MAX_ITERATIONS;
	double ica_tolerance = TOLERANCE;
	bool lasso = true;

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			continue;
		}else
		if (argname == "--header") {
			continue;
		}
		else if (argname == "--iter") {
			max_ica_iteration = atoi(argv[count + 1]);
		}
		else if (argname == "--tol") {
			ica_tolerance = atof(argv[count + 1]);
		}
		else if (argname == "--lasso") {
			lasso = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}
	std::vector<std::string> header_names;
	header_names.resize(xs.n);
	if (header && csv1.getHeader().size() > 0)
	{
		header_names = csv1.getHeader();
	}
	else
	{
		for (int i = 0; i < xs.n; i++)
		{ 
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}

	LiNGAM.set(xs.n);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");
	if (lasso)
	{
		LiNGAM.remove_redundancy();
	}
	LiNGAM.before_sorting();

	LiNGAM.B.print_e("B");

	LiNGAM.digraph(header_names, "digraph.txt");
	LiNGAM.report(header_names);
}