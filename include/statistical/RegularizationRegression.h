#ifndef _LASSOREGRESSOR_H
#define _LASSOREGRESSOR_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#include "../../include/Matrix.hpp"
#include "../../include/util/mathutil.h"


inline double _soft_thresholding_operator(const double x, const double lambda)
{
	double d = 0.0;
	if (x > 0 && lambda < fabs(x))
	{
		d = x - lambda;
	}
	else if (x < 0 && lambda < fabs(x))
	{
		d = x + lambda;
	}
	return d;
}

class RegressionBase
{
public:
	int error;

	dnn_double lambda1 = 0.001;
	dnn_double lambda2 = 0.001;
	int max_iteration = 10000;
	dnn_double tolerance = 0.001;

	int num_iteration = 0;
	Matrix<dnn_double> means;
	Matrix<dnn_double> sigma;
	Matrix<dnn_double> coef;
	bool use_bias = true;

	Matrix<dnn_double> whitening(Matrix<dnn_double>& X)
	{
		Matrix<dnn_double> x = X;
		for (int i = 0; i < X.m; i++)
		{
			for (int j = 0; j < X.n; j++)
			{
				x(i, j) = (X(i, j) - means(0, j)) / sigma(0, j);
			}
		}
		return x;
	}

	int getStatus() const
	{
		return error;
	}

	void report()
	{
		printf("--------------\n");
		printf("     ŒW”     \n");
		printf("--------------\n");
		printf("(intercept)%10.4f\n", coef(0, coef.n - 1));
		for (int i = 0; i < coef.n - 1; i++)
		{
			printf("%10.4f\n", coef(0, i));
		}
		printf("--------------\n");
	}

	virtual int fit(Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{
		return -999;
	}

	virtual Matrix<dnn_double> predict(Matrix<dnn_double>& X)
	{
		Matrix<dnn_double>& x = whitening(X);
		//Matrix<dnn_double> x = X;

		if (use_bias)
		{
			Matrix<dnn_double> bias;
			bias = bias.ones(x.m, 1);

			x = x.appendCol(bias);
		}
		return x * coef.transpose();
	}
};

class LassoRegression:public RegressionBase
{
public:
	dnn_double error_eps = 100000.0;

	LassoRegression( const double lambda1_ = 0.001, const int max_iteration_ = 10000, const dnn_double tolerance_ = 0.001)
	{
		lambda1 = lambda1_;
		max_iteration = max_iteration_;
		tolerance = tolerance_;
		error = 0;
	}


	virtual int fit(Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{
		means = X.Mean();
		sigma = X.Std(means);
		Matrix<dnn_double> train = whitening(X);
		Matrix<dnn_double> beta;

		if (use_bias)
		{
			Matrix<dnn_double> bias;
			bias = bias.ones(train.m, 1);

			train = train.appendCol(bias);

			//train.print("", "%.3f ");
			beta = beta.zeros(1, train.n);
		}
		const int N = train.m;

		int varNum = train.n;

		if (use_bias)
		{
			varNum = train.n - 1;
			beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
			//beta.print();
		}

		error = -1;
		for (size_t iter = 0; iter < max_iteration; ++iter)
		{
			coef = beta;

			for (int k = 0; k < varNum; k++)
			{
				Matrix<dnn_double> tmp_beta = beta;

				Matrix<dnn_double>& x_k = train.Col(k);

				dnn_double z = x_k.transpose()*x_k;

				tmp_beta(0, k) = 0;
				dnn_double p_k = x_k.transpose()*(y - train * tmp_beta.transpose());

				//if (k == 0 && iter == 0)
				//{
				//	printf("%f\n", p_k);
				//	printf("%f\n", lambda*N);
				//}
				double w_k = _soft_thresholding_operator(p_k, lambda1*N) / z;
				beta(0, k) = w_k;

				if (use_bias)
				{
					beta(0, train.n - 1) = 0.0;
					beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
				}
			}

			error_eps = (coef - beta).norm();
			if (error_eps < tolerance)
			{
				num_iteration = iter;
				printf("convergence:%f - iter:%d\n", (coef - beta).norm(), iter);
				error = 0;
				break;
			}
		}
		return error;
	}

};

class ElasticNetRegression :public RegressionBase
{
public:
	dnn_double error_eps = 100000.0;

	ElasticNetRegression(const double lambda1_ = 0.001, const double lambda2_ = 0.001, const int max_iteration_ = 10000, const dnn_double tolerance_ = 0.001)
	{
		lambda1 = lambda1_;
		lambda2 = lambda2_;
		max_iteration = max_iteration_;
		tolerance = tolerance_;
		error = 0;
	}

	virtual int fit(Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{

		means = X.Mean();
		sigma = X.Std(means);
		Matrix<dnn_double> train = whitening(X);
		Matrix<dnn_double> beta;

		if (use_bias)
		{
			Matrix<dnn_double> bias;
			bias = bias.ones(train.m, 1);

			train = train.appendCol(bias);

			//train.print("", "%.3f ");
			beta = beta.zeros(1, train.n);
		}
		const int N = train.m;

		int varNum = train.n;

		if (use_bias)
		{
			varNum = train.n - 1;
			beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
			//beta.print();
		}

		error = -1;
		for (size_t iter = 0; iter < max_iteration; ++iter)
		{
			coef = beta;

			for (int k = 0; k < varNum; k++)
			{
				Matrix<dnn_double> tmp_beta = beta;

				Matrix<dnn_double>& x_k = train.Col(k);

				dnn_double z = x_k.transpose()*x_k + lambda2;

				tmp_beta(0, k) = 0;
				dnn_double p_k = x_k.transpose()*(y - train * tmp_beta.transpose());

				//if (k == 0 && iter == 0)
				//{
				//	printf("%f\n", p_k);
				//	printf("%f\n", lambda*N);
				//}
				double w_k = _soft_thresholding_operator(p_k, lambda1*N) / z;
				beta(0, k) = w_k;

				if (use_bias)
				{
					beta(0, train.n - 1) = 0.0;
					beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
				}
			}

			error_eps = (coef - beta).norm();
			if (error_eps < tolerance)
			{
				num_iteration = iter;
				printf("convergence:%f - iter:%d\n", (coef - beta).norm(), iter);
				error = 0;
				break;
			}
		}
		return error;
	}

};

class RidgeRegression :public RegressionBase
{
public:

	RidgeRegression(const double lambda1_ = 0.001, const int max_iteration_ = 10000, const dnn_double tolerance_ = 0.001)
	{
		lambda1 = lambda1_;
		max_iteration = max_iteration_;
		tolerance = tolerance_;
		error = 0;
	}

	virtual int fit(Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{

		means = X.Mean();
		sigma = X.Std(means);
		Matrix<dnn_double> train = whitening(X);
		Matrix<dnn_double> beta;

		if (use_bias)
		{
			Matrix<dnn_double> bias;
			bias = bias.ones(train.m, 1);

			train = train.appendCol(bias);

			//train.print("", "%.3f ");
			beta = beta.zeros(1, train.n);
		}
		const int N = train.m;

		int varNum = train.n;

		if (use_bias)
		{
			varNum = train.n - 1;
			beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
			//beta.print();
		}

		Matrix<dnn_double> I;
		I = I.unit(train.n, train.n);
		coef = (train.transpose() * train + lambda1*I).inv()*train.transpose()*y;
		coef = coef.transpose();

		return error;
	}
};
#endif
