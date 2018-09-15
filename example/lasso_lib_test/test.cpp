#define _cublas_Init_def extern
#include "../../include/matrix_config.h"
#include "../../include/Matrix.hpp"

#include "../../include/util/lasso_lib.h"
#include "../../include/util/csvreader.h"

int main(int argc, char** argv)
{
	CSVReader csv1("Boston.csv");
	Matrix<dnn_double> df = csv1.toMat();

	df = df.removeCol(0);

	Matrix<dnn_double>& y = df.Col(13);
	y.print("", "%.3f ");

	Matrix<dnn_double>& X = df.removeCol(14,-1);
	X.print("", "%.3f ");
#if 0
	Matrix<dnn_double> means = X.Mean();

	means.print("É ");
	Matrix<dnn_double>É–2 = X.Std(means);
	É–2.print("É–");

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			X(i, j) = (X(i, j)-means(0,j)) / É–2(0,j);
		}
	}
	X.print("", "%.3f ");
#endif

	Lasso_Regressor lasso(1.0, 1000, 0.0001);

	lasso.fit(X, y);

	Matrix<dnn_double>& z = lasso.predict(X);
	lasso.report();

	return 0;
}