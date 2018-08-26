#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"
#include "../../include/util/lasso_lib.h"
#include "../../include/statistical/LinearRegression.h"

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
	read_csv(len, "x.csv", x);
	read_csv(len, "y.csv", y);
	read_csv(len, "z.csv", z);
	read_csv(len, "w.csv", w);

	Matrix<dnn_double>xs(len, 4);

	for (int i = 0; i < len; i++)
	{
		xs(i, 0) = x(i, 0);
		xs(i, 1) = y(i, 0);
		xs(i, 2) = z(i, 0);
		xs(i, 3) = w(i, 0);

	}
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(4);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e();

	for (int i = 0; i < LiNGAM.B.m; i++)
		for (int j = 0; j < i; j++)
			if (LiNGAM.B(i, j) == 0.0)
			{
				LiNGAM.B(i, j) = 0.01*(double)rand() / RAND_MAX;
			}
	LiNGAM.B.print_e();

	multiple_regression reg;
	reg.set(1);

#if 0
	reg.fit(x, y);
	reg.les.x.print();
	reg.report();

	Matrix<dnn_double>& xy = x.appendCol(y);
	reg.set(2);

	reg.fit(xy, z);
	reg.les.x.print();
	reg.report();

	Matrix<dnn_double>& xyz = xy.appendCol(z);
	reg.set(3);

	reg.fit(xyz, w);
	reg.les.x.print();
	reg.report();

#else
	Lasso_Regressor lasso(0.01, 1000, 1.0e-4);

	lasso.fit(x, y);
	//lasso.x.print();
	Matrix<dnn_double> c1(lasso.model->coef, 1+1, 1);
	c1.v[0] = (c1.v[0]/*- lasso.model->mean[0]*/) /lasso.model->var[0];
	c1.print();

	Matrix<dnn_double>& xy = x.appendCol(y);

	Lasso_Regressor lasso2(0.01, 1000, 1.0e-4);
	//Ridge_Regressor  lasso2(0.1);
	//ElasticNet_Regressor lasso2(0.01, 0.0001,  1000, 1.0e-4);
	lasso2.fit(xy, z);
	//lasso.x.print();
	Matrix<dnn_double> c2(lasso2.model->coef, 2+1, 1);
	c2.v[0] = (c2.v[0]/* - lasso2.model->mean[0]*/) / lasso2.model->var[0];
	c2.v[1] = (c2.v[1]/* - lasso2.model->mean[1]*/) / lasso2.model->var[1];
	c2.print();

	Matrix<dnn_double>& xyz = xy.appendCol(z);

	Lasso_Regressor lasso3(0.01, 100000, 1.0e-4);
	//Ridge_Regressor  lasso3(0.1);
	//ElasticNet_Regressor lasso3(0.01, 0.00001, 100000, 1.0e-4);
	lasso3.fit(xyz, w);
	//lasso.x.print();
	Matrix<dnn_double> c3(lasso3.model->coef, 3+1, 1);
	c3.v[0] = (c3.v[0]/* - lasso3.model->mean[0]*/) / lasso3.model->var[0];
	c3.v[1] = (c3.v[1] /*- lasso3.model->mean[1]*/) / lasso3.model->var[1];
	c3.v[2] = (c3.v[2]/* - lasso3.model->mean[2]*/) / lasso3.model->var[2];
	c3.print();
#endif
}