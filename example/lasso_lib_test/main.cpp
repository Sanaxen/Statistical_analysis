#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/RegularizationRegression.h"
//#ifdef USE_LIBLM
#include "../../include/statistical/RegularizationRegression_eigen_version.h"
//#endif


void test1()
{
	::Matrix<dnn_double> X(5, 1);
	::Matrix<dnn_double> y(5, 1);
	::Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i,0) = i + 1;
		y(i,0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	LassoRegression lasso(0.2, 1000, 0.001);
	Lasso lasso_(0.2, 1000, 0.001);
	lasso.fit(X, y);
	lasso_.fit(X, y);

	::Matrix<dnn_double>& Y = lasso.predict(X_test);
	::Matrix<dnn_double>& Y_ = lasso_.predict(X_test);
	Y.print("lasso");
	(Y - Y_).print();

	lasso.report();
}

void test2()
{
	::Matrix<dnn_double> X(5, 1);
	::Matrix<dnn_double> y(5, 1);
	::Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	LassoRegression lasso(0.2, 1000, 0.001);
	Lasso lasso_(0.2, 1000, 0.001);
	lasso.fit(X, y);
	lasso_.fit(X, y);

	::Matrix<dnn_double>& t = lasso.predict(X_test);
	::Matrix<dnn_double>& t_ = lasso_.predict(X_test);
	t.print("lasso");
	(t - t_).print();
	lasso.report();
}

void test3()
{
	::Matrix<dnn_double> X(5, 1);
	::Matrix<dnn_double> y(5, 1);
	::Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	RidgeRegression ridge(0.2);
	Ridge ridge_(0.2);

	ridge.fit(X, y);
	ridge_.fit(X, y);

	::Matrix<dnn_double>& t = ridge.predict(X_test);
	::Matrix<dnn_double>& t_ = ridge_.predict(X_test);
	t.print("ridge");
	(t - t_).print();
	ridge.report();
}

void test4()
{
	::Matrix<dnn_double> X(5, 1);
	::Matrix<dnn_double> y(5, 1);
	::Matrix<dnn_double> X_test(5, 1);

	for (int i = 0; i < 5; ++i) {
		X(i, 0) = i + 1;
		y(i, 0) = i + 1;
		X_test(i, 0) = i + 1 + 5;
	}

	ElasticNetRegression elasticNet(0.2, 0.2, 1000, 0.001);
	ElasticNet elasticNet_(0.2, 0.2, 1000, 0.001);

	elasticNet.fit(X, y);
	elasticNet_.fit(X, y);

	::Matrix<dnn_double>& t = elasticNet.predict(X_test);
	::Matrix<dnn_double>& t_ = elasticNet_.predict(X_test);
	t.print("elasticNet");
	(t - t_).print();
	elasticNet.report();
}

int main(int argc, char** argv)
{
	test1();
	test2();
	test3();
	test4();
	return 0;
}