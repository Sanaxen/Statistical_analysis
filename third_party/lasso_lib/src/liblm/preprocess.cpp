//
// Created by Doge on 2017/11/14.
//

#include "preprocess.h"
#include <iostream>


void StandardScaler::fit(const MatrixXd & X) {
    mean_ = X.colwise().mean();
	//auto t = X.rowwise() -mean_;
	//var_ = (t.array() * t.array()).matrix().colwise().sum().array().sqrt() / 2.0;
	
	var_ = VectorXd::Zero(X.cols());
	for (int i = 0; i < X.rows(); i++)
		for (int j = 0; j < X.cols(); j++)
			var_[j] += (X(i, j) - mean_[j])*(X(i, j) - mean_[j]);

	for (int i = 0; i < X.cols(); i++)
		var_[i] = sqrt(var_[i] / double(X.rows() - 1));

	//std::cout << "mean=" << mean_ << std::endl;
	//std::cout << "var=" << var_ << std::endl;
}

MatrixXd StandardScaler::transform(const MatrixXd & X) {
    MatrixXd t(X.rowwise() - mean_);
	for (int i = 0; i < X.cols(); ++i)
	{
		t.col(i) /= var_(i);
	}
	return t;
}

MatrixXd StandardScaler::fit_transform(const MatrixXd & X) {
	fit(X);
    return transform(X);
}