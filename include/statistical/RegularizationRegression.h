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

	//Global Contrast Normalization (GCN)
	inline Matrix<dnn_double> whitening_( Matrix<dnn_double>& X)
	{
		return X.whitening(means, sigma);
	}

	inline int getStatus() const
	{
		return error;
	}

	void report(Matrix<dnn_double>& A, std::vector<std::string>& header)
	{
		printf("--------------\n");
		printf("     係数     \n");
		printf("--------------\n");
		printf("(intercept)%10.4f\n", coef(0, coef.n - 1));
		for (int i = 0; i < coef.n - 1; i++)
		{
			printf("[%03d]%10.10s %10.4f\n", i+1, header[i+1].c_str(), coef(0, i));
		}
		printf("--------------\n");

		printf("必要な説明変数\n");
		int num = 0;
		for (int i = 0; i < coef.n - 1; i++)
		{
			if (fabs(coef(0, i)) > 1.0e-6)
			{
				num++;
				printf("[%03d]%10.10s %10.4f\n", i+1, header[i+1].c_str(), coef(0, i));
			}
		}
		printf("説明変数:%d -> %d\n", coef.n - 1, num);

		bool war = false;
		Matrix<dnn_double>& cor = A.Cor();
		for (int i = 0; i < cor.m; i++)
		{
			for (int j = i + 1; j < cor.n; j++)
			{
				if (fabs(cor(i, j)) > 0.5 && fabs(cor(i, j)) < 0.6)
				{

					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(多重共線性)の疑いがあります\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.6 && fabs(cor(i, j)) < 0.8)
				{
					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(多重共線性)の疑いがかなりあります\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.8)
				{
					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(多重共線性)の強い疑いがあります\n", cor(i, j));
				}
			}
		}
		if (war)
		{
			printf("\nmulticollinearity(多重共線性)の疑いがある説明変数がある場合は\n");
			printf("どちらか一方を外して再度分析することで、多重共線性を解消する場合があります。\n");
		}

	}

	virtual int fit(Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{
		return -999;
	}

	virtual Matrix<dnn_double> predict(Matrix<dnn_double>& X)
	{
		Matrix<dnn_double>& x = whitening_(X);
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
		Matrix<dnn_double>& train = whitening_(X);
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

				const Matrix<dnn_double>& x_k_T = x_k.transpose();
				const dnn_double z = x_k_T*x_k;

				tmp_beta(0, k) = 0;
				const dnn_double p_k = x_k_T*(y - train * tmp_beta.transpose());

				beta(0, k) = _soft_thresholding_operator(p_k, lambda1*N) / z;

				//if (use_bias)
				//{
				//	beta(0, train.n - 1) = 0.0;
				//	beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
				//}
			}

			if (use_bias)
			{
				beta(0, train.n - 1) = 0.0;
				beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
			}

			num_iteration = iter;
			error_eps = (coef - beta).norm();
			if (error_eps < tolerance)
			{
				printf("convergence:%f - iter:%d\n", error_eps, iter);
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
		Matrix<dnn_double>& train = whitening_(X);
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

				const Matrix<dnn_double>& x_k_T = x_k.transpose();
				const dnn_double z = x_k_T*x_k + lambda2;

				tmp_beta(0, k) = 0;
				const dnn_double p_k = x_k.transpose()*(y - train * tmp_beta.transpose());

				beta(0, k) = _soft_thresholding_operator(p_k, lambda1*N) / z;

				//if (use_bias)
				//{
				//	beta(0, train.n - 1) = 0.0;
				//	beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
				//}
			}

			if (use_bias)
			{
				beta(0, train.n - 1) = 0.0;
				beta(0, train.n - 1) = (y - train * beta.transpose()).Sum() / N;
			}
			num_iteration = iter;
			error_eps = (coef - beta).norm();
			if (error_eps < tolerance)
			{
				printf("convergence:%f - iter:%d\n", error_eps, iter);
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

	virtual int fit( Matrix<dnn_double>& X, Matrix<dnn_double>& y)
	{

		means = X.Mean();
		sigma = X.Std(means);
		Matrix<dnn_double>& train = whitening_(X);
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
