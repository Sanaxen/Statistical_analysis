#ifndef FastICA_H_
#define FastICA_H_
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define MAX_ITERATIONS	1000
#define TOLERANCE		0.0001
//where 1 ≤ a1 ≤ 2 is some suitable constant, often taken as a1 = 1. 
#define A1			1

//#define RAND	RandMT
#define RAND	Rand

class ICA
{
	Matrix<dnn_double> whitening_;
	Matrix<dnn_double> means_;
	
	Matrix<dnn_double> V;
	Matrix<dnn_double> xt;
	int component;
public:
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
		const int rows = X.m;
		const int cols = X.n;

		const dnn_double alp = A1;
		Matrix<dnn_double>& W = Matrix<dnn_double>(rows, rows);
		Matrix<dnn_double>& D = Matrix<dnn_double>(rows, rows);
		Matrix<dnn_double>& d = Matrix<dnn_double>(1, rows);

		W = W.RAND();
		W.print();

		SVDcmp<dnn_double> svd(W);
		d.diag_vec(svd.Sigma);


		Matrix<dnn_double> Wd = svd.U*svd.V.diag((d.Reciprocal()).v)*svd.U.transpose()*W;

		std::vector<dnn_double> lim(max_iteration + 1, 1000);

		int it = 0;
		const Matrix<dnn_double>& xt = X.transpose() / dnn_double(cols);

		while (lim[it] > tolerance && it < max_iteration)
		{
			// g1(u) = tanh(a1*u)
			Matrix<dnn_double>& tmp = Tanh(alp*(Wd * X));
			tmp = tmp*xt - D.diag(tmp.one_sub_sqr(alp).mean_rows().v)*Wd;

			SVDcmp<dnn_double> svd(tmp);
			d.diag_vec(svd.Sigma);

			W = svd.U*svd.V.diag(d.Reciprocal().v)*svd.U.transpose()*tmp;

			lim[it + 1] = fabs((W*Wd.transpose()).MaxDiag() - 1.0);
			Wd = W;
			it++;
		}
		if (it >= max_iteration)
		{
			printf("iteration:%d >= max_iteration:%d\n", it, max_iteration);
		}

		return Wd;
	}

	void fit(Matrix<dnn_double>& X, const int max_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{

		const int rows = X.m;
		const int cols = X.n;
		Matrix<dnn_double>& d = Matrix<dnn_double>(1, cols);

		Matrix<dnn_double> xx = X;

		//centering
		xx = xx.Centers(means_);

		//whitening
		xt = xx.transpose();
		xx = xx / dnn_double(rows);
		SVDcmp<dnn_double> svd1(xt*xx);
		d.diag_vec(svd1.Sigma);

		V = svd1.V.diag(InvSqrt(d).v)*svd1.U.transpose();
		xx = V*xt;

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
