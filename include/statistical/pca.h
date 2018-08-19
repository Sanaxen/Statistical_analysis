#ifndef _PCA_H
#define _PCA_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

//Principal component analysis
class PCA
{
	Matrix<dnn_double> x;
	int variablesNum;
	eigenvalues eig;

	Matrix<dnn_double> whitening_;
	//Variance - covariance matrix	ï™éUã§ï™éUçsóÒ
	Matrix<dnn_double> variance_covariance_;

	bool use_variance_covariance_matrix_;
	int error;
public:
	Matrix<dnn_double> variance;
	Matrix<dnn_double> mean;

	std::vector<dnn_double> component;
	Matrix<dnn_double> coef;

	PCA()
	{
		error = 0;
	}

	int getStatus() const
	{
		return error;
	}
	//Cumulative contribution rate(ó›êœäÒó^ó¶)
	dnn_double contribution_rate(const int id)
	{
		return eig.getRealValue()(0, id) / variance_covariance_.Tr();
	}

	Matrix<dnn_double>& whitening()
	{
		return whitening_;
	}
	Matrix<dnn_double>& variance_covariance()
	{
		return variance_covariance_;
	}

	eigenvalues& getEigen()
	{
		return eig;
	}

	void set(const int variablesNum_)
	{
		variablesNum = variablesNum_;
	}
	

	int fit(Matrix<dnn_double>&xx, bool sort_value=false, bool use_variance_covariance_matrix=true)
	{
		use_variance_covariance_matrix_ = use_variance_covariance_matrix;
		variance = Matrix<dnn_double>().zeros(1, variablesNum);
		mean = Matrix<dnn_double>().zeros(1, variablesNum);

		x = xx;
		Matrix<dnn_double> C(variablesNum, variablesNum);

		coef = Matrix<dnn_double>(variablesNum, variablesNum);
		component.resize(variablesNum, 0.0);
		const int n = x.m;
		//centering && whitening
		for (int j = 0; j < variablesNum; j++)
		{
			dnn_double mean_ = 0.0;
			dnn_double s2 = 0.0;
			for (int i = 0; i < n; i++)
			{
				mean_ += x(i, j);
				s2 += x(i, j) * x(i, j);
			}
			mean_ /= n;
			mean(0, j) = mean_;

			s2 /= n;
			//s2 = n * (s2 - mean_ * mean_) / (n - 1);
			s2 = (s2 - mean_ * mean_);
			s2 = sqrt(s2);

			variance(0, j) = s2;

			if (use_variance_covariance_matrix_)
			{
			for (int i = 0; i < n; i++)
				x(i, j) = (x(i, j) - mean_) / s2;
		}
			else
			{
				for (int i = 0; i < n; i++)
					x(i, j) = (x(i, j) - mean_);
			}
		}
		whitening_ = x;

		// 	Variance - covariance matrix
		for (int j = 0; j < variablesNum; j++)
		{
			for (int i = j; i < variablesNum; i++)
			{
				dnn_double s2 = 0.0;
				for (int k = 0; k < n; k++)
					s2 += x(k, i) * x(k, j);
				//s2 /= (n - 1);
				s2 /= n;
				C(i, j) = s2;
				if (i != j)
					C(j, i) = s2;
			}
		}
		//column_major_Matrix<dnn_double> cmmA;
		//cmmA.set_column_major(C);
		//cmmA.toRow_major(C);

		variance_covariance_ = C;

		eig.set(C);
		eig.calc(sort_value);
		error = eig.getStatus();
		if (error == 0)
		{
			Matrix<dnn_double>& value = eig.getRealValue();
			//value.print("å≈óLíl");
			//std::vector<Matrix<dnn_double>>tmp2 = eig.getRightVector(0);
			//(C*tmp2[0] - value(0,0) * tmp2[0]).print("-check--");

			std::vector<int>& index = eig.getIndex();
			for (int i = 0; i < variablesNum; i++) {
				component[i] = value(0, i);
				std::vector<Matrix<dnn_double>>vec = eig.getRightVector(i);
				for (int j = 0; j < variablesNum; j++)
				{
					coef(i, j) = vec[0](0, j);
				}
			}
			//column_major_Matrix<dnn_double> cmm;
			//cmm.set_column_major(coef);
			//cmm.toRow_major(coef);

		}
		return error;
	}

	Matrix<dnn_double> principal_component()
	{
		Matrix<dnn_double>&w = whitening();
		Matrix<dnn_double> pca_w(w.m, w.n);
		for (int i = 0; i < w.m; i++)
		{
			for (int j = 0; j < w.n; j++)
			{
				double s = 0.0;
				for (int k = 0; k < variablesNum; k++)
				{
					s += w(i, k)*coef(j, k);
				}
				pca_w(i, j) = s;
			}
		}

		return pca_w;
	}

	void Report(int debug = 0)
	{
		mean.print("mean");
		variance.print("variance");

		if (error == 0) {
			for (int i = 0; i < variablesNum; i++) {
				printf("éÂê¨ï™ %f", component[i]);
				printf(" åWêî");
				for (int j = 0; j < variablesNum; j++)
					printf(" %f", coef(i, j));
				printf("\n");
			}
		}
		else
			printf("error:%d\n", stat);

		printf("\neigen value:");
		for (int i = 0; i < variablesNum; i++)
		{
			printf("%.3f ", getEigen().getRealValue()(0,i));
		}
		printf("\n");

		for (int i = 0; i < variablesNum; i++)
		{
			getEigen().getRightVector(i)[0].print("eigen vector");
		}

		if (!use_variance_covariance_matrix_)
		{
			variance_covariance().print("ï™éUã§ï™éUçsóÒ");
		}
		else
		{
			variance_covariance().print("ëää÷çsóÒ");
		}
		if (debug )
		{
			//pca.getEigen().getImageValue().print("Im");
			Matrix<dnn_double>& value = getEigen().getRealValue();
			//value.print();
			for (int i = 0; i < variablesNum; i++)
			{
				std::vector<Matrix<dnn_double>>&tmp2 = getEigen().getRightVector(i);

				//tmp2[1].print("I,m");
				(variance_covariance()*tmp2[0] - value(0, i) * tmp2[0]).chop(1.0e-6).print("check");
			}
		}
		printf("äÒó^ó¶(%%)\n");
		Matrix<dnn_double>&w = whitening();
		for (int j = 0; j < w.n; j++)
		{
			printf("%.3f%%\n", 100 * contribution_rate(j));
		}
		principal_component().print("principal component");
	}
};

#endif
