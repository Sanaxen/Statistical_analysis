#define _cublas_Init_def extern
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/csvreader.h"
//cublas_init _cublas_Init;

int main(int argc, char** argv)
{
	CSVReader csv1("Boston.csv");
	Matrix<dnn_double> df = csv1.toMat();

	df = df.removeCol(0);

	Matrix<dnn_double>& y = df.Col(13);	//7
	y.print("", "%.3f ");

	Matrix<dnn_double>& X = df.removeCol(13,-1);
	X.print("", "%.3f ");
	printf("***%d\n", X.n);
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


	LassoRegression lasso_my(1.0, 1000, 0.0001);

	lasso_my.fit(X, y);
	lasso_my.report();

/*
scikit-learnÇÃLasso
22.5328063241
[-0.          0.         -0.          0.         -0.          2.71517992
-0.         -0.         -0.         -0.         -1.34423287  0.18020715
-3.54700664]
*/
	return 0;
}