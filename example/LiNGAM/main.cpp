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
		xs(i, 0) = y(i, 0);
		xs(i, 1) = z(i, 0);
		xs(i, 2) = w(i, 0);
		xs(i, 3) = x(i, 0);
		//xs(i, 0) = x(i, 0);
		//xs(i, 1) = y(i, 0);
		//xs(i, 2) = z(i, 0);
		//xs(i, 3) = w(i, 0);

	}
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(4);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e("B");

	for (int i = 0; i < LiNGAM.B.m; i++)
		for (int j = 0; j < i; j++)
			if (LiNGAM.B(i, j) == 0.0)
			{
				LiNGAM.B(i, j) = 0.01*(double)rand() / RAND_MAX;
			}
	LiNGAM.B.print_e("add random eps");


#if 0
	multiple_regression reg;
	reg.set(1);

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

	Matrix<dnn_double> B = LiNGAM.B;
	xs.print();
	Matrix<dnn_double> X = xs.Col(LiNGAM.replacement[0]);
	Matrix<dnn_double> Y = xs.Col(LiNGAM.replacement[1]);
	for (int i = 1; i < LiNGAM.B.m; i++)
	{
		//X.print();
		//Y.print();
		size_t n_iter = 10000000;
		Lasso_Regressor lasso(0.01, n_iter, 1.0e-4);
		lasso.fit(X, Y);
		while (lasso.getStatus() != 0)
		{
			n_iter *= 2;
			lasso.fit(X, Y, n_iter, 1.0e-4);
			printf("n_iter=%d\n", lasso.param.n_iter);
		}

		Matrix<dnn_double> c(lasso.model->coef, i+1, 1);
		for (int k = 0; k < i; k++)
		{
			c.v[k] = c.v[k] / lasso.model->var[k];
			B(i, k) = c.v[k];
		}
		if (i == LiNGAM.B.m) break;
		c.print();
		X = X.appendCol(Y);
		Y = xs.Col(LiNGAM.replacement[i+1]);
	}
	B.print_e();

	{
		B = B.chop(0.001);
		B.print_e();

		std::vector<std::string> item;
		item.resize(B.m);
		item[0] = "X";
		item[1] = "Y";
		item[2] = "Z";
		item[3] = "W";

		FILE* fp = fopen("digraph.txt", "w");
		fprintf(fp, "digraph {\n");
		fprintf(fp, "node [fontname=\"MS UI Gothic\" layout=circo shape=circle]\n");
		
		for (int i = 0; i < B.n; i++)
		{
			fprintf(fp, "\"%s\"[color=blue shape=circle]\n", item[i].c_str());
			for (int j = 0; j < B.n; j++)
			{
				if (B(i, j) != 0.0)
				{
					fprintf(fp, "\"%s\"-> \"%s\" [label=\"%8.3f\" color=black]\n", item[j].c_str(), item[i].c_str(), B(i,j));
				}
			}
		}
		fprintf(fp, "}\n");
		fclose(fp);
	}
#if 0
	Lasso_Regressor lasso(0.01, 1000, 1.0e-4);

	lasso.fit(x, y);
	//lasso.x.print();
	Matrix<dnn_double> c1(lasso.model->coef, 1+1, 1);
	c1.v[0] = c1.v[0] /lasso.model->var[0];
	c1.print();
	lasso.report();

	Matrix<dnn_double>& xy = x.appendCol(y);

	Lasso_Regressor lasso2(0.01, 1000, 1.0e-4);
	//Ridge_Regressor  lasso2(0.1);
	//ElasticNet_Regressor lasso2(0.01, 0.0001,  1000, 1.0e-4);
	lasso2.fit(xy, z);
	//lasso.x.print();
	Matrix<dnn_double> c2(lasso2.model->coef, 2+1, 1);
	c2.v[0] = c2.v[0] / lasso2.model->var[0];
	c2.v[1] = c2.v[1] / lasso2.model->var[1];
	c2.print();
	lasso2.report();

	Matrix<dnn_double>& xyz = xy.appendCol(z);

	Lasso_Regressor lasso3(0.01, 100000, 1.0e-4);
	//Ridge_Regressor  lasso3(0.1);
	//ElasticNet_Regressor lasso3(0.01, 0.00001, 100000, 1.0e-4);
	lasso3.fit(xyz, w);
	//lasso.x.print();
	Matrix<dnn_double> c3(lasso3.model->coef, 3+1, 1);
	c3.v[0] = c3.v[0] / lasso3.model->var[0];
	c3.v[1] = c3.v[1] / lasso3.model->var[1];
	c3.v[2] = c3.v[2] / lasso3.model->var[2];
	c3.print();
	lasso3.report();
#endif
#endif
}