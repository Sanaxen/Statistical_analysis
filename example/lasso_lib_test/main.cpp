#define _cublas_Init_def extern
#include "../../include/matrix_config.h"
#include "../../include/Matrix.hpp"

#include "../../include/util/lasso_lib.h"


void test()
{
	Matrix<dnn_double> X(5, 1);
	Matrix<dnn_double> y(5, 1);
	Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i,0) = i + 1;
		y(i,0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	Lasso_Regressor lasso(0.2, 1000, 0.001);

	lasso.fit(X, y);

	Matrix<dnn_double>& Y = lasso.predict(X_test) - lasso.predict_(X_test).chop(1.0e-6);
	Y.print();
}

void test2()
{
	Matrix<dnn_double> X(5, 1);
	Matrix<dnn_double> y(5, 1);
	Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	Lasso_Regressor lasso(0.2, 1000, 0.001);

	lasso.fit(X, y);

	Matrix<dnn_double>& t = (lasso.predict(X_test) - lasso.predict_(X_test)).chop(1.0e-6);
	t.print();
}

void test3()
{
	Matrix<dnn_double> X(5, 1);
	Matrix<dnn_double> y(5, 1);
	Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	Ridge_Regressor ridge(0.2);

	ridge.fit(X, y);

	Matrix<dnn_double>& t = (ridge.predict(X_test, true) - ridge.predict_(X_test)).chop(1.0e-6);
	t.print();
}

void test4()
{
	Matrix<dnn_double> X(5, 1);
	Matrix<dnn_double> y(5, 1);
	Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	ElasticNet_Regressor elasticNet(0.2, 0.2, 1000, 0.001);

	elasticNet.fit(X, y);

	Matrix<dnn_double>& t = (elasticNet.predict(X_test) - elasticNet.predict_(X_test)).chop(1.0e-6);
	t.print();
}

int main(int argc, char** argv)
{
	double *X = (double *)malloc(sizeof(double) * 5);
	double *y = (double *)malloc(sizeof(double) * 5);
	double *X_test = (double *)malloc(sizeof(double) * 5);
	for (int i = 0; i < 5; ++i) {
		X[i] = i + 1;
		y[i] = i + 1;
		X_test[i] = i + 1 + 5;
	}
	lm_problem *prob = (lm_problem *)malloc(sizeof(lm_problem));
	dmatrix data;
	data.data = X;
	data.row = 5;
	data.col = 1;
	prob->X = data;
	prob->y = y;
	lm_param param;
	param.alg = REG;
	param.regu = L1L2;
	param.e = 0.001;
	param.n_iter = 1000;
	param.lambda1 = 0.2;
	param.lambda2 = 0.2;
	lm_model *model = lm_train(prob, &param);
	dmatrix d;
	d.data = X_test;
	d.row = 5;
	d.col = 1;
	y = lm_predict(model, &d);
	for (int i = 0; i < 5; ++i)
		printf("%.3f ", y[i]);
	printf("\n");

	free_model(model);

	test();
	test2();
	test3();
	test4();
	return 0;
}