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

int Lasso::fit(const MatrixXd & X, const VectorXd & y) {
	error = -1;
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = VectorXd::Zero(X_train.cols());
    for (size_t iter = 0; iter < n_iter_; ++iter) {
        VectorXd tmp = VectorXd::Zero(X_train.cols());
        //assert(z.rows() == tmp.cols());
		
		tmp = coef_;
		int N = X_train.rows();


		for (int k = 0; k < (int)X_train.cols(); k++) {
			auto z = X_train.col(k).transpose() * X_train.col(k);
			double wk = coef_(k);
            coef_(k) = 0;
            double p_k = X_train.col(k).transpose() * (y - X_train * coef_);
			double w_k = 0.0;
			if (p_k < -lambda_ *N / 2.0)
				w_k = (p_k + lambda_*N / 2.0) / z;
			else if (p_k > lambda_ *N / 2.0)
				w_k = (p_k - lambda_*N / 2.0) / z;
            else
                w_k = 0.0;
            //tmp(k) = w_k;
            coef_(k) = w_k;
        }

		//coef_(X_train.cols() - 1) = 0.0;
		//coef_(X_train.cols()-1) = (y - X_train * coef_).sum() / N;
		if ((coef_ - tmp).norm() < e_)
		{
			printf("convergence:%f - iter:%d\n", (coef_ - tmp).norm(), iter);
			error = 0;
			break;
		}
        //coef_ = tmp;
    }
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
    coef_ = VectorXd::Zero(X_train.cols());
    for (size_t iter = 0; iter < n_iter_; ++iter) {
        VectorXd tmp = VectorXd::Zero(X_train.cols());
        //assert(z.rows() == tmp.cols());

		tmp = coef_;
		int N = X_train.rows();

        for (size_t k = 0; k < X_train.cols(); ++k) {
			auto z = X_train.col(k).transpose() * X_train.col(k) + lambda2_;
			double wk = coef_(k);
            coef_(k) = 0;
            double p_k = X_train.col(k).transpose() * (y - X_train * coef_);
            double w_k = 0.0;
            if (p_k < -lambda1_ *N/ 2.0)
                w_k = (p_k + lambda1_ *N / 2.0) / z;
            else if (p_k > lambda1_ *N / 2.0)
                w_k = (p_k - lambda1_ *N / 2.0) / z;
            else
                w_k = 0.0;
            //tmp(k) = w_k;
            coef_(k) = w_k;
        }
		//coef_(N) = 0.0;
		coef_(N) = (y - X_train * coef_).sum() / N;
		if ((coef_ - tmp).norm() < e_)
		{
			printf("convergence:%f - iter:%d\n", (coef_ - tmp).norm(), iter);
			error = 0;
			break;
		}
        //coef_ = tmp;
    }
	return error;
}