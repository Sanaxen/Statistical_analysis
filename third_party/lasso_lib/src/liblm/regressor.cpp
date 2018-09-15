//
// Created by Doge on 2017/11/14.
//

#include "regressor.h"
#include <iostream>
#include <vector>

class _dummy_dummy_
{
public:
	_dummy_dummy_()
	{
		Eigen::initParallel(); 
	}
};

_dummy_dummy_ _dummy_dummy__;

VectorXd Regressor::predict(const MatrixXd &X) {
	MatrixXd X_test = combine_bias(scalaer->transform(X));
	return X_test * coef_;
}

int LinearRegression::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = (X_train.transpose() * X_train).inverse()*X_train.transpose()*y;
	return 0;
}

inline double soft_thresholding_operator(const double x, const double lambda)
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

//VectorXd X_dot_b(const MatrixXd & X, const VectorXd & b)
//{
//	VectorXd d = VectorXd::Zero(X.rows());
//	{
//		for (int ii = 0; ii < X.rows(); ii++)
//		{
//			double tmp = 0.0;
//			for (int jj = 0; jj < X.cols() - 1; jj++)
//			{
//				tmp += X(ii, jj)*b(jj);
//			}
//			d(ii) = tmp;
//		}
//	}
//	return d;
//}

int Lasso::fit(const MatrixXd & X, const VectorXd & y) {
	error = -1;
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));

	//std::cout << "X_train=" << X_train << std::endl;
	//std::cout << "y=" << y << std::endl;

	VectorXd beta = VectorXd::Zero(X_train.cols());
	int N = X_train.rows();
	
	beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;

	//std::cout << beta << std::endl;

#if 0
	for (size_t iter = 0; iter < n_iter_; ++iter) {
		coef_ = beta;

		for (int k = 0; k < (int)X_train.cols()-1; k++) {
			VectorXd tmp_beta = beta;

			auto z = X_train.col(k).transpose()*X_train.col(k);

			tmp_beta(k) = 0;
			auto r_j = y - X_dot_b(X_train, tmp_beta);
			double p_k = (X_train.col(k).transpose()*r_j);
			double w_k = soft_thresholding_operator(p_k, lambda_*N) / z;
			beta(k) = w_k;

			beta(X_train.cols() - 1) = 0.0;
			beta(X_train.cols() - 1) = (y - X_dot_b(X_train, beta)).sum() / N;
		}

		if ((coef_ - beta).norm() < e_)
		{
			printf("convergence:%f - iter:%d\n", (coef_ - beta).norm(), iter);
			error = 0;
			break;
		}
		coef_ = beta;
	}
#else
	for (size_t iter = 0; iter < n_iter_; ++iter) {
		coef_ = beta;

		for (int k = 0; k < (int)X_train.cols() - 1; k++) {
			VectorXd tmp_beta = beta;

			auto z = X_train.col(k).transpose()*X_train.col(k);

			tmp_beta(k) = 0;
			double p_k = X_train.col(k).transpose() * (y - X_train * tmp_beta);

			double w_k = soft_thresholding_operator(p_k, lambda_*N) / z;
			beta(k) = w_k;

			beta(X_train.cols() - 1) = 0.0;
			beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;
		}

		if ((coef_ - beta).norm() < e_)
		{
			printf("convergence:%f - iter:%d\n", (coef_ - beta).norm(), iter);
			error = 0;
			break;
		}
		coef_ = beta;
	}


	//coef_ = VectorXd::Zero(X_train.cols());
	//for (int k = 0; k < (int)X_train.cols(); k++) {
	//	auto z = X_train.col(k).transpose() * X_train.col(k);
	//	double wk = coef_(k);
	//	coef_(k) = 0;
	//	double p_k = X_train.col(k).transpose() * (y - X_train * coef_);
	//	double w_k = 0.0;
	//	if (p_k < -lambda_ *N / 2.0)
	//		w_k = (p_k + lambda_*N / 2.0) / z;
	//	else if (p_k > lambda_ *N / 2.0)
	//		w_k = (p_k - lambda_*N / 2.0) / z;
	//	else
	//		w_k = 0.0;
	//	tmp(k) = w_k;
	//	coef_(k) = w_k;
	//}
#endif

	return error;
}

int Ridge::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = (X_train.transpose() * X_train + lambda_
                                             * MatrixXd::Identity(X_train.cols(), X_train.cols())).inverse()*X_train.transpose()*y;
	return 0;
}

int ElasticNet::fit(const MatrixXd & X, const VectorXd & y) {
	error = -1;
	MatrixXd X_train = combine_bias(scalaer->fit_transform(X));

	//std::cout << "X_train=" << X_train << std::endl;
	//std::cout << "y=" << y << std::endl;

	VectorXd beta = VectorXd::Zero(X_train.cols());
	int N = X_train.rows();

	beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;

	//std::cout << beta << std::endl;

	for (size_t iter = 0; iter < n_iter_; ++iter) {
		coef_ = beta;

		for (size_t k = 0; k < X_train.cols()-1; ++k) {
			VectorXd tmp_beta = beta;

			auto z = X_train.col(k).transpose()*X_train.col(k) + lambda2_;

			tmp_beta(k) = 0;
			double p_k = X_train.col(k).transpose() * (y - X_train * tmp_beta);

			double w_k = soft_thresholding_operator(p_k, lambda1_*N) / z;
			beta(k) = w_k;

			beta(X_train.cols() - 1) = 0.0;
			beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;
		}

		if ((coef_ - beta).norm() < e_)
		{
			printf("convergence:%f - iter:%d\n", (coef_ - beta).norm(), iter);
			error = 0;
			break;
		}
        //coef_ = tmp;
    }
	return error;
}