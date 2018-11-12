#ifndef FastICA_H_
#define FastICA_H_
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define MAX_ITERATIONS	1000
#define TOLERANCE		0.0001
//where 1 ≤ a1 ≤ 2 is some suitable constant, often taken as a1 = 1. 
#define A1_			1

//#define RAND	RandMT
#define RAND	Rand

class ICA
{
	Matrix<dnn_double> whitening_;
	Matrix<dnn_double> means_;
	
	Matrix<dnn_double> V;
	Matrix<dnn_double> xt;
	//Number of components
	int component;
	int error;
public:
	inline int getStatus() const
	{
		return error;
	}
	//pre-whitening matrix that projects data onto the first compc principal components.
	Matrix<dnn_double> K;

	//estimated un-mixing matrix (see definition in details)
	Matrix<dnn_double> W;

	//estimated mixing matrix
	Matrix<dnn_double> A;

	//estimated source matrix
	Matrix<dnn_double> S;

	ICA(){}

	Matrix<dnn_double>& whitening()
	{
		return whitening_;
	}
	Matrix<dnn_double>& means()
	{
		return means_;
	}

	void set(const int compc)
	{
		component = compc;
	}

	/*
	X:pre-processed data matrix
	K:pre-whitening matrix that projects data onto the first compc principal components.
	W:estimated un-mixing matrix (see definition in details)
	A:estimated mixing matrix
	S:estimated source matrix
	*/
	inline Matrix<dnn_double> fastICA_calc(Matrix<dnn_double>& X, const int max_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
		error = 0;
		const int rows = X.m;
		const int cols = X.n;

		const dnn_double alp = A1_;
		Matrix<dnn_double>& W = Matrix<dnn_double>(rows, rows);
		Matrix<dnn_double>& D = Matrix<dnn_double>(rows, rows);
		Matrix<dnn_double>& d = Matrix<dnn_double>(1, rows);

		W = W.RAND();
		W.print();

		SVDcmp<dnn_double> svd(W);
		d.diag_vec(svd.Sigma);


		Matrix<dnn_double>& Wd = svd.U*svd.V.diag((d.Reciprocal()).v)*svd.U.transpose()*W;

		dnn_double delta = FLT_MAX;
		dnn_double delta_pre = FLT_MAX;
		dnn_double delta_min = FLT_MAX;
		Matrix<dnn_double> Wd_;

		int it = 0;
		const Matrix<dnn_double>& xt = X.transpose() / dnn_double(cols);

		while (delta > tolerance && it < max_iteration)
		{
			// g1(u) = tanh(a1*u)
			Matrix<dnn_double>& tmp = Tanh(alp*(Wd * X));
			tmp = tmp*xt - D.diag(tmp.one_sub_sqr(alp).mean_rows().v)*Wd;

			SVDcmp<dnn_double> svd(tmp);
			d.diag_vec(svd.Sigma);

			W = svd.U*svd.V.diag(d.Reciprocal().v)*svd.U.transpose()*tmp;

			delta = fabs((W*Wd.transpose()).MaxDiag() - 1.0);

			if (it % 1000 == 0)
			{
				printf("delta[%d]:%f\n", it, delta);
			}
			if (delta < delta_min)
			{
				Wd_ = W;
				delta_min = delta;
			}
			if (fabs(delta - delta_pre) < 1.0e-6 && it > 0)
			{
				printf("convergence:%f - iter:%d\n", fabs(delta - delta_pre), it);
				break;
			}
			if (fabs(delta) <= tolerance)
			{
				printf("convergence:%f - iter:%d\n", fabs(delta), it);
			}

			delta_pre = delta;
			Wd = W;
			it++;
		}
		if (it >= max_iteration)
		{
			error = -1;
			printf("iteration:%d >= max_iteration:%d\n", it, max_iteration);
			printf("delta_min:%f\n", delta_min);
			if (delta_min < 0.01)
			{
				Wd = Wd_;
				error = 1;
			}
		}

		return Wd;
	}

	void fit(Matrix<dnn_double>& X, const int max_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{

		const int rows = X.m;
		const int cols = X.n;
		Matrix<dnn_double>& d = Matrix<dnn_double>(1, cols);

		Matrix<dnn_double> xx = X;

#if 10
		//centering
		xx = xx.Centers(means_);
#else
		//Global Contrast Normalization (GCN)
		means_ = xx.Mean();
		Matrix<dnn_double>& sigma = xx.Std(means_);
		xx = xx.whitening(means_, sigma);
#endif

		//whitening
		xt = xx.transpose();
		xx = xx / dnn_double(rows);
		SVDcmp<dnn_double> svd1(xt*xx);
		d.diag_vec(svd1.Sigma);

		const dnn_double epsilon = 1.0e-12;
#if 10
		//xPCAwhite = diag(1. / sqrt(diag(S) + epsilon)) * U' * x;
		// PCA
		V = svd1.V.diag(InvSqrt(d+ epsilon).v)*svd1.U.transpose();
		xx = V*xt;
#else
		//xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
		// ZCA
		V = svd1.U*svd1.V.diag(InvSqrt(d).v)*svd1.U.transpose();
		xx = V*xt;
#endif

		whitening_ = xx;

		//fast ICA
		Matrix<dnn_double>& ica = fastICA_calc(whitening_, max_iteration, tolerance);


		//output
		Matrix<dnn_double>&x = xt.transpose().DeCenters(means_);

		K = V.transpose();
		Matrix<dnn_double>& wrk = ica*V;

		S = (wrk*xt).transpose();

		Matrix<dnn_double>& wrk2 = wrk.transpose();

		V = wrk2*(wrk*wrk2).inv();
		A = V.transpose();

		W = ica.transpose();
	}
};
#endif
