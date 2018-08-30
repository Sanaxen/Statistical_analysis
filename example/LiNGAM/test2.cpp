#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"

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
	Matrix<dnn_double> x, y, z, u, v, w;

	int len = 1000;
	read_csv(len, "x.csv", x);
	read_csv(len, "y.csv", y);
	read_csv(len, "z.csv", z);
	read_csv(len, "u.csv", u);
	read_csv(len, "v.csv", v);
	read_csv(len, "w.csv", w);

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
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(6);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");
	fflush(stdout);
	LiNGAM.remove_redundancy();
	LiNGAM.before_sorting();
	LiNGAM.B.print_e("B");

	LiNGAM.digraph(names, "digraph.txt");
	LiNGAM.report(names);
}