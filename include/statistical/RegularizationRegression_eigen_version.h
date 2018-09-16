//
//
// Made a single header && original bug fix by Sanaxen on 2018/09/15.
// base code Created by Doge on 2017/11/14.

#ifndef LIBLM_REGRESSOR_H
#define LIBLM_REGRESSOR_H


#include <Eigen/Dense>
//using namespace Eigen;

inline Eigen::MatrixXd formMatrix_double(::Matrix<dnn_double> & X)
{
	Eigen::MatrixXd xx(X.m, X.n);
	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			xx(i, j) = X(i, j);
		}
	}
	return xx;
}
inline ::Matrix<dnn_double> formEigenVectorXd(Eigen::VectorXd & X)
{
	::Matrix<dnn_double> xx(1, X.size());
	for (int j = 0; j < xx.n; j++)
	{
		xx(0, j) = X(j);
	}
	return xx;
}

inline double __soft_thresholding_operator(const double x, const double lambda)
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

class StandardScaler {
public:
	void fit(const Eigen::MatrixXd & X) {
		mean_ = X.colwise().mean();
		//auto t = X.rowwise() -mean_;
		//var_ = (t.array() * t.array()).matrix().colwise().sum().array().sqrt() / 2.0;

		var_ = Eigen::VectorXd::Zero(X.cols());
		for (int i = 0; i < X.rows(); i++)
			for (int j = 0; j < X.cols(); j++)
				var_[j] += (X(i, j) - mean_[j])*(X(i, j) - mean_[j]);

		for (int i = 0; i < X.cols(); i++)
			var_[i] = sqrt(var_[i] / double(X.rows() - 1));

		//std::cout << "mean=" << mean_ << std::endl;
		//std::cout << "var=" << var_ << std::endl;
	}

	Eigen::MatrixXd transform(const Eigen::MatrixXd & X) {
		Eigen::MatrixXd t(X.rowwise() - mean_);
		for (int i = 0; i < X.cols(); ++i)
		{
			t.col(i) /= var_(i);
		}
		return t;
	}
	Eigen::MatrixXd fit_transform(const Eigen::MatrixXd & X) {
	    fit(X);
	    return transform(X);
	}
	//void set_param(const Eigen::RowVectorXd &mean, const Eigen::RowVectorXd &var)
 //   {
 //       mean_ = mean;
 //       var_ = var;
 //   }
    //Eigen::RowVectorXd get_mean() {
    //    return mean_;
    //}
    //Eigen::RowVectorXd get_var() {
    //    return var_;
    //}
//private:
    Eigen::RowVectorXd mean_;
    Eigen::RowVectorXd var_;
};

//struct LModel {
//    Eigen::RowVectorXd mean;
//    Eigen::RowVectorXd var;
//	Eigen::VectorXd coef;
//};


class LinearModel {
public:
	LinearModel() {
    }
    ~LinearModel() {
    }

	virtual int fit(::Matrix<dnn_double> & X, ::Matrix<dnn_double> y) 
	{
		return -999;
	};

	virtual int fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) 
	{
		return -999;
	};

    virtual Eigen::VectorXd LinearModel::predict(const Eigen::MatrixXd & X) {
	    Eigen::MatrixXd X_test = combine_bias(scalaer.transform(X));
	    return X_test * coef_;
	}


    //void set_model(const Eigen::VectorXd &coef, const Eigen::RowVectorXd &mean, const Eigen::RowVectorXd &var) {
    //    coef_ = coef;
    //    scalaer->set_param(mean, var);
    //}
    //LModel get_model() {
    //    LModel model;
    //    model.coef = coef_;
    //    model.mean = scalaer->get_mean();
    //    model.var = scalaer->get_var();
    //    return model;
    //}
	Eigen::VectorXd coef_;
	StandardScaler scalaer;

protected:
	Eigen::MatrixXd LinearModel::combine_bias(const Eigen::MatrixXd & X) {
	    Eigen::MatrixXd t(X.rows(), X.cols() + 1);
	    t.col(X.cols()) = Eigen::VectorXd::Ones(X.rows());
	    t.leftCols(X.cols()) = X;
	    return t;
	}
};


class Regressor :public LinearModel {
public:
	int error;

	dnn_double lambda1 = 0.001;
	dnn_double lambda2 = 0.001;
	int max_iteration = 10000;
	dnn_double tolerance = 0.001;

	int num_iteration = 0;

    virtual Eigen::VectorXd predict(const Eigen::MatrixXd &X) {
	    Eigen::MatrixXd X_test = combine_bias(scalaer.transform(X));
	    return X_test * coef_;
	}
	virtual ::Matrix<dnn_double> predict(::Matrix<dnn_double> &X) {
		return formEigenVectorXd(predict(formMatrix_double(X)));
	}

	int getStatus() const
	{
		return error;
	}
};


class LinearRegression :public Regressor {
public:
    virtual  int fit(const Eigen::MatrixXd & X, const Eigen::VectorXd & y) {
	    Eigen::MatrixXd X_train = combine_bias(scalaer.fit_transform(X));
	    coef_ = (X_train.transpose() * X_train).inverse()*X_train.transpose()*y;
	}
};


class Lasso :public Regressor {
public:
	dnn_double error_eps = 100000.0;
	Lasso(double lambda, size_t n_iter, double e)
	{
		Regressor();
		lambda1 = lambda;
		max_iteration = n_iter;
		tolerance = e;
    }
	virtual int fit(::Matrix<dnn_double> & X, ::Matrix<dnn_double> y) 
	{
		Eigen::MatrixXd& xx = formMatrix_double(X);
		Eigen::MatrixXd& yy = formMatrix_double(y);

		return fit(xx, yy);
	}

    virtual int fit(const Eigen::MatrixXd & X, const Eigen::VectorXd & y) 
	{
		error = -1;
		Eigen::MatrixXd X_train = combine_bias(scalaer.fit_transform(X));

		//std::cout << "X_train=" << X_train << std::endl;
		//std::cout << "y=" << y << std::endl;

		Eigen::VectorXd beta = Eigen::VectorXd::Zero(X_train.cols());
		int N = X_train.rows();

		beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;

		//std::cout << beta << std::endl;

		for (size_t iter = 0; iter < max_iteration; ++iter) {
			coef_ = beta;

			for (int k = 0; k < (int)X_train.cols() - 1; k++) {
				Eigen::VectorXd tmp_beta = beta;

				auto z = X_train.col(k).transpose()*X_train.col(k);

				tmp_beta(k) = 0;
				double p_k = X_train.col(k).transpose() * (y - X_train * tmp_beta);

				double w_k = __soft_thresholding_operator(p_k, lambda1*N) / z;
				beta(k) = w_k;

				beta(X_train.cols() - 1) = 0.0;
				beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;
			}

			error_eps = (coef_ - beta).norm();
			if (error_eps < tolerance)
			{
				num_iteration = iter;
				printf("convergence:%f - iter:%d\n", (coef_ - beta).norm(), iter);
				error = 0;
				break;
			}
			coef_ = beta;
		}

		return error;
	}
};



class Ridge :public Regressor {
public:
    Ridge(double lambda)
	{ 
		Regressor(); 
		lambda1 = lambda;
	}
    virtual int fit(const Eigen::MatrixXd & X, const Eigen::VectorXd & y) {
	    Eigen::MatrixXd X_train = combine_bias(scalaer.fit_transform(X));
	    coef_ = (X_train.transpose() * X_train + lambda1
	                                             * Eigen::MatrixXd::Identity(X_train.cols(), X_train.cols())).inverse()*X_train.transpose()*y;
		return 0;
	}
	virtual int fit(::Matrix<dnn_double> & X, ::Matrix<dnn_double> y)
	{
		Eigen::MatrixXd& xx = formMatrix_double(X);
		Eigen::MatrixXd& yy = formMatrix_double(y);

		return fit(xx, yy);
	}
};



class ElasticNet :public Regressor {
public:
	dnn_double error_eps = 100000.0;
	ElasticNet(double lambda1_, double lambda2_, size_t n_iter, double e)
	{
        Regressor();
		lambda1 = lambda1_;
		lambda2 = lambda2_;
		max_iteration = n_iter;
		tolerance = e;
		error = 0;
	}
    virtual int fit(const Eigen::MatrixXd & X, const Eigen::VectorXd & y) {
		error = -1;
		Eigen::MatrixXd X_train = combine_bias(scalaer.fit_transform(X));

		//std::cout << "X_train=" << X_train << std::endl;
		//std::cout << "y=" << y << std::endl;

		Eigen::VectorXd beta = Eigen::VectorXd::Zero(X_train.cols());
		int N = X_train.rows();

		beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;

		//std::cout << beta << std::endl;

		for (size_t iter = 0; iter < max_iteration; ++iter) {
			coef_ = beta;

			for (size_t k = 0; k < X_train.cols() - 1; ++k) {
				Eigen::VectorXd tmp_beta = beta;

				auto z = X_train.col(k).transpose()*X_train.col(k) + lambda2;

				tmp_beta(k) = 0;
				double p_k = X_train.col(k).transpose() * (y - X_train * tmp_beta);

				double w_k = __soft_thresholding_operator(p_k, lambda1*N) / z;
				beta(k) = w_k;

				beta(X_train.cols() - 1) = 0.0;
				beta(X_train.cols() - 1) = (y - X_train * beta).sum() / N;
			}

			error_eps = (coef_ - beta).norm();
			if (error_eps < tolerance)
			{
				num_iteration = iter;
				printf("convergence:%f - iter:%d\n", (coef_ - beta).norm(), iter);
				error = 0;
				break;
			}
			//coef_ = tmp;
		}
		return error;
	}
	virtual int fit(::Matrix<dnn_double> & X, ::Matrix<dnn_double> y)
	{
		Eigen::MatrixXd& xx = formMatrix_double(X);
		Eigen::MatrixXd& yy = formMatrix_double(y);

		return fit(xx, yy);
	}
};



#endif //LIBLM_REGRESSOR_H
