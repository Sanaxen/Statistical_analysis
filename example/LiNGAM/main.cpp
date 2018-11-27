#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"
#include "../../include/util/csvreader.h"
#include "../../include/util/swilk.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#endif


//https://qiita.com/m__k/items/bd87c063a7496897ba7c
//LiNGAMÉÇÉfÉãÇÃêÑíËï˚ñ@Ç…Ç¬Ç¢Çƒ
int main(int argc, char** argv)
{
	std::string csvfile("sample.csv");

	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
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
	xs = csv1.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			xs = xs.removeCol(0);
		}
	}


	Lingam LiNGAM;

	size_t max_ica_iteration = MAX_ITERATIONS;
	double ica_tolerance = TOLERANCE;
	double lasso = 0.0;

	bool error_distr = true;
	int error_distr_size[2] = { 1,1 };
	bool capture = false;
	bool sideways = false;
	int diaglam_size = 20;
	char* output_diaglam_type = "png";
	std::vector<std::string> x_var;

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			continue;
		}else
		if (argname == "--header") {
			continue;
		}else
		if (argname == "--col") {
			continue;
		}
		else if (argname == "--iter") {
			max_ica_iteration = atoi(argv[count + 1]);
		}
		else if (argname == "--tol") {
			ica_tolerance = atof(argv[count + 1]);
		}
		else if (argname == "--lasso") {
			lasso = atof(argv[count + 1]);
		}
		else if (argname == "--sideways") {
			sideways = atoi(argv[count + 1]) != 0 ? true : false;
		}
		else if (argname == "--output_diaglam_type") {
			output_diaglam_type = argv[count + 1];
		}
		else if (argname == "--diaglam_size") {
			diaglam_size = atoi(argv[count + 1]);
		}
		else if (argname == "--x_var") {
			x_var.push_back(argv[count + 1]);
		}
		else if (argname == "--error_distr") {
			error_distr = atoi(argv[count + 1]) != 0 ? true : false;
		}
		else if (argname == "--capture") {
			capture = atoi(argv[count + 1]) == 0 ? false : true;
		}
		else if (argname == "--error_distr_size") {
			sscanf(argv[count + 1], "%d,%d", error_distr_size, error_distr_size + 1);
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
		for (int i = 0; i < xs.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
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

	std::vector<int> x_var_idx;
	int y_var_idx = -1;

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
		}
	}

	LiNGAM.set(xs.n);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");
	if (lasso)
	{
		LiNGAM.remove_redundancy(lasso);
	}
	LiNGAM.before_sorting();

	LiNGAM.B.print_e("B");
	//LiNGAM.B = LiNGAM.B.chop(0.1);

	std::vector<int> residual_flag(xs.n, 0);
	if (error_distr)
	{
		Matrix<dnn_double> r(xs.m, xs.n);
		for (int j = 0; j < xs.m; j++)
		{
			Matrix<dnn_double> x(xs.n, 1);
			for (int i = 0; i < xs.n; i++)
			{
				x(i, 0) = xs(j, i);
			}
			Matrix<dnn_double>& rr = x - LiNGAM.B*x;
			for (int i = 0; i < xs.n; i++)
			{
				r(j, i) = rr(0, i);
			}
		}
		r.print_csv("error_distr.csv", header_names);

		std::vector<std::string> shapiro_wilk_values;
		{
			printf("shapiro_wilk test(0.05) start\n");
			shapiro_wilk shapiro;
			for (int i = 0; i < xs.n; i++)
			{

				Matrix<dnn_double> tmp = r.Col(i);
				//tmp = tmp.whitening(tmp.Mean(), tmp.Std(tmp.Mean()));
				//tmp = tmp.Centers(tmp.Mean());

				int stat = shapiro.test(tmp);
				if (stat == 0)
				{
					char buf[256];
					sprintf(buf,"w:%-4.4f p_value:%-.16g", shapiro.get_w(), shapiro.p_value());
					shapiro_wilk_values.push_back(buf);
					//printf("%s\n", buf);

					printf("[%-20.20s]w:%-8.3f p_value:%-10.16f\n", header_names[i].c_str(), shapiro.get_w(), shapiro.p_value());
					if (shapiro.p_value() > 0.05)
					{
						residual_flag[i] = 1;
					}
				}
				else
				{
					printf("error shapiro.test=%d\n", stat);
				}
			}
			printf("shapiro_wilk test end\n\n");
		}
#ifdef USE_GNUPLOT
		gnuPlot plot1(std::string(GNUPLOT_PATH), 3, false);
		int win_x = error_distr_size[0];
		int win_y = error_distr_size[1];
		plot1.multi_histgram(r, header_names, residual_flag, capture, win_x, win_y);
		if (capture)
			plot1.draw(0);
		else 
			plot1.draw();

#endif
	}
	LiNGAM.digraph(header_names, x_var, residual_flag, "digraph.txt", sideways, diaglam_size, output_diaglam_type);
	LiNGAM.report(header_names);


}