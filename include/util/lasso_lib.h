#ifndef _LASSO_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define _LASSO_H

#include "../../third_party/lasso_lib/src/liblm/c_api.h"

#ifdef _DEBUG
#pragma comment(lib, "../../third_party/lasso_lib/lib/x64/Debug/lasso_lib.lib")
#else
#pragma comment(lib, "../../third_party/lasso_lib/lib/x64/Release/lasso_lib.lib")
#endif

class _Regressor
{
public:
	lm_problem *prob;
	dmatrix data;
	lm_param param;
	lm_model *model;

	Matrix<dnn_double> x;

	_Regressor()
	{
		prob = (lm_problem *)malloc(sizeof(lm_problem));
		model = NULL;
	}
	virtual ~_Regressor()
	{
		free(prob);
		if (model) free_model(model);
	}
	virtual void fit(const Matrix<dnn_double> &X, const Matrix<dnn_double> &y)
	{}

	Matrix<dnn_double> combine_bias(const Matrix<dnn_double> &X)
	{
		Matrix<dnn_double> t(X.m, X.n + 1);
		for (int j = 0; j < X.n; j++)
		{
			for (int i = 0; i < X.m; i++)
			{
				t(i, j) = X(i, j);
			}
			for (int i = 0; i < X.m; i++)
			{
				t(i, j + 1) = 1;
			}
		}
		return t;
	}

	//Match the scale of the model
	Matrix<dnn_double> fit_model_scale(const Matrix<dnn_double> &X, bool bias=false)
	{
		Matrix<dnn_double> xx = X;
		for (int j = 0; j < X.n; j++)
			for (int i = 0; i < X.m; i++)
			{
				xx(i, j) /= model->var[j];
			}

		if ( bias ) xx = combine_bias(xx);
		return xx;
	}

	Matrix<dnn_double> predict_(Matrix<dnn_double>& X)
	{
		dmatrix d;
#ifdef USE_FLOAT
		double* x = new double[X.m*X.n];
		for (int i = 0; i < X.m*X.n; i++) x[i] = X.v[i];
		d.data = x;
#else
		d.data = &X.v[0];
#endif

		d.row = X.m;
		d.col = X.n;
		double* y = lm_predict(model, &d);
#ifdef USE_FLOAT
		float* yy = new float[X.m, X.n];
		for (int i = 0; i < X.m, X.n; i++) yy[i] = y[i];
		Matrix<dnn_double> Y(yy, X.m, X.n);
#else
		Matrix<dnn_double> Y(y, X.m, X.n);
#endif


		free(y);
#ifdef USE_FLOAT
		delete[] x;
		delete[] yy;
#endif
		return Y;
	}

	Matrix<dnn_double> predict(Matrix<dnn_double>& X, bool bias = false)
	{
		Matrix<dnn_double> t;

		if (bias)
		{
			Matrix<dnn_double> t_ = combine_bias(fit_model_scale(X));
			for (int j = 0; j < t_.n - 1; j++)
			{
				for (int i = 0; i < t_.m; i++)
				{
					t_(i, j) *= model->coef[j];
				}
				for (int i = 0; i < t_.m; i++)
				{
					t_(i, j) += model->coef[t_.n - 1];
				}
			}
			t = Matrix<dnn_double>(t_.m, t_.n-1);
			for (int j = 0; j < t.n; j++)
			{
				for (int i = 0; i < t.m; i++)
				{
					t(i, j) = t_(i, j);
				}
			}
		}
		else
		{
			t = (fit_model_scale(X));

			for (int j = 0; j < t.n; j++)
			{
				for (int i = 0; i < t.m; i++)
				{
					t(i, j) *= model->coef[j];
				}
			}
		}

		return t;
	}
};

class Lasso_Regressor :public _Regressor
{
	size_t n_iter_;
	double e_;
	double lambda_;

public:
	Lasso_Regressor(double lambda, size_t n_iter, double e) :
		lambda_(lambda), n_iter_(n_iter), e_(e) {
		_Regressor();
	}
	virtual void fit(const Matrix<dnn_double> &X, const Matrix<dnn_double> &y)
	{
#ifdef USE_FLOAT
		double* x = new double[X.m*X.n];
		for (int i = 0; i < X.m*X.n; i++) x[i] = X.v[i];
		data.data = x;
#else
		data.data = &X.v[0];
#endif
		data.row = X.m;
		data.col = X.n;

		prob->X = data;
#ifdef USE_FLOAT
		double* yy = new double[y.m*y.n];
		for (int i = 0; i < y.m*y.n; i++) yy[i] = y.v[i];
		prob->y = yy;
#else
		prob->y = &y.v[0];
#endif

		param.alg = REG;
		param.regu = L1;
		param.e = e_;
		param.n_iter = n_iter_;
		param.lambda1 = lambda_;
		param.lambda2 = 1;
		model = lm_train(prob, &param);
#ifdef USE_FLOAT
		delete[] x;
		delete[] yy;
#endif

		dnn_double* c = new dnn_double[X.n];
		for (int i = 0; i < X.n; i++) c[i] = model->coef[i];
		x = Matrix<dnn_double>(c, X.n, 1);
		delete[]c;
	}
};

class Ridge_Regressor :public _Regressor
{
	double lambda_;

public:
	Ridge_Regressor(double lambda) :lambda_(lambda) { _Regressor(); }

	virtual void fit(const Matrix<dnn_double> &X, const Matrix<dnn_double> &y)
	{
#ifdef USE_FLOAT
		double* x = new double[X.m*X.n];
		for (int i = 0; i < X.m*X.n; i++) x[i] = X.v[i];
		data.data = x;
#else
		data.data = &X.v[0];
#endif
		data.row = X.m;
		data.col = X.n;

		prob->X = data;
#ifdef USE_FLOAT
		double* yy = new double[y.m*y.n];
		for (int i = 0; i < y.m*y.n; i++) yy[i] = y.v[i];
		prob->y = yy;
#else
		prob->y = &y.v[0];
#endif

		param.alg = REG;
		param.regu = L2;
		param.lambda2 = lambda_;
		model = lm_train(prob, &param);
#ifdef USE_FLOAT
		delete[] x;
		delete[] yy;
#endif
		dnn_double* c = new dnn_double[X.n];
		for (int i = 0; i < X.n; i++) c[i] = model->coef[i];
		x = Matrix<dnn_double>(c, X.n, 1);
		delete[]c;

	}
};

class ElasticNet_Regressor :public _Regressor
{
	size_t n_iter_;
	double e_;
	double lambda1_;
	double lambda2_;

public:
	ElasticNet_Regressor(double lambda1, double lambda2, size_t n_iter, double e) :
		lambda1_(lambda1), lambda2_(lambda2_), n_iter_(n_iter), e_(e) {
		_Regressor();
	}

	virtual void fit(const Matrix<dnn_double> &X, const Matrix<dnn_double> &y)
	{
#ifdef USE_FLOAT
		double* x = new double[X.m*X.n];
		for (int i = 0; i < X.m*X.n; i++) x[i] = X.v[i];
		data.data = x;
#else
		data.data = &X.v[0];
#endif
		data.row = X.m;
		data.col = X.n;

		prob->X = data;
#ifdef USE_FLOAT
		double* yy = new double[y.m*y.n];
		for (int i = 0; i < y.m*y.n; i++) yy[i] = y.v[i];
		prob->y = yy;
#else
		prob->y = &y.v[0];
#endif

		param.alg = REG;
		param.regu = L1L2;
		param.e = e_;
		param.n_iter = n_iter_;
		param.lambda1 = lambda1_;
		param.lambda2 = lambda1_;
		model = lm_train(prob, &param);
#ifdef USE_FLOAT
		delete[] x;
		delete[] yy;
#endif
		dnn_double* c = new dnn_double[X.n];
		for (int i = 0; i < X.n; i++) c[i] = model->coef[i];
		x = Matrix<dnn_double>(c, X.n, 1);
		delete[]c;

	}
};

#endif
