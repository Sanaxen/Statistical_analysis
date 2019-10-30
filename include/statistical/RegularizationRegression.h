#ifndef _LASSOREGRESSOR_H
#define _LASSOREGRESSOR_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#include "../../include/Matrix.hpp"
#include "../../include/util/mathutil.h"
#include "../../include/util/utf8_printf.hpp"
#include "../../include/util/plot.h"


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
	int y_var_idx;
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

	double AIC;
	double se;

	//Global Contrast Normalization (GCN)
	inline Matrix<dnn_double> whitening_( Matrix<dnn_double>& X)
	{
		return X.whitening(means, sigma);
	}

	inline int getStatus() const
	{
		return error;
	}

	double calc_AIC(Matrix<dnn_double>& A, Matrix<dnn_double>& y, int num = 0)
	{
		if (num == 0)
		{
			for (int i = 0; i < coef.n - 1; i++)
			{
				if (fabs(coef(0, i)) > 1.0e-6)
				{
					num++;
				}
			}
		}
		if (num == 0)
		{
			AIC = 999999999.0;
			se = AIC;
			return AIC;
		}
		Matrix<dnn_double> y_predict = predict(A);
		se = 0.0;
		for (int i = 0; i < A.m; i++)
		{
			se += (y(i, 0) - y_predict(i, 0))*(y(i, 0) - y_predict(i, 0));
		}

		AIC = A.m*(log(2.0*M_PI*se / A.m) + 1) + 2.0*(num + 2.0);
		if (fabs(coef(0, coef.n - 1)) < 1.0e-16)
		{
			AIC = A.m*(log(2.0*M_PI*se / A.m) + 1) + 2.0*(num + 1.0);
		}
		return AIC;
	}

	void report(std::string& filename, Matrix<dnn_double>& A, std::vector<std::string>& header, Matrix<dnn_double>* y = NULL)
	{
		FILE* fp1 = stdout;
		if (filename != "")
		{
			fp1 = fopen(filename.c_str(), "w");
			if (!fp1) return;
		}

		fprintf(fp1, "--------------\n");
		fprintf(fp1, "     ŒW”     \n");
		fprintf(fp1, "--------------\n");
		fprintf(fp1, "(intercept)%10.4f\n", coef(0, coef.n - 1));
		for (int i = 0; i < coef.n - 1; i++)
		{
			fprintf(fp1, "[%03d]%-10.10s %10.4f\n", i+1, header[i+1].c_str(), coef(0, i));
		}
		fprintf(fp1, "--------------\n");

		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d,%s\n", y_var_idx, header[0].c_str());
		std::vector<int> var_indexs;
		fprintf(fp1,"•K—v‚Èà–¾•Ï”\n");
		int num = 0;
		for (int i = 0; i < coef.n - 1; i++)
		{
			int k = i;
			if (k >= y_var_idx)
			{
				k++;
			}
			if (fabs(coef(0, i)) > 1.0e-6)
			{
				var_indexs.push_back(i);
				num++;
				fprintf(fp1, "[%03d]%-10.10s %10.4f\n", i+1, header[i+1].c_str(), coef(0, i));

				if ( fp )fprintf(fp, "%d,%s\n", k, header[i + 1].c_str());
			}
		}
		fclose(fp);
		fprintf(fp1, "à–¾•Ï”:%d -> %d\n", coef.n - 1, num);

		if (y != NULL)
		{
			calc_AIC(A, *y, num);
			fprintf(fp1, "SE:%.3f\n", se);
			fprintf(fp1, "AIC:%.3f\n", AIC);
		}

		bool war = false;
		Matrix<dnn_double>& cor = A.Cor();
		fprintf(fp1, "[‘ŠŠÖŒW”(‘½d‹¤ü«‚Ì‰Â”\«•]‰¿)]\n");
		fprintf(fp1, "    %-10.8s", "");
		for (int j = 0; j < A.n; j++)
		{
			fprintf(fp1, "%-10.8s", header[j + 1].c_str());
		}
		fprintf(fp1, "\n");
		fprintf(fp1, "--------------------------------------------------------------------------------------------\n");
		for (int i = 0; i < A.n; i++)
		{
			fprintf(fp1, "%-10.8s", header[i + 1].c_str());
			for (int j = 0; j < A.n; j++)
			{
				fprintf(fp1, "%10.4f", cor(i, j));
			}
			fprintf(fp1, "\n");
		}
		fprintf(fp1, "--------------------------------------------------------------------------------------------\n");

		for (int i = 0; i < cor.m; i++)
		{
			for (int j = i + 1; j < cor.n; j++)
			{
				bool skip = false;
				for (int k = 0; k < var_indexs.size(); k++)
				{
					if (i == var_indexs[k])
					{
						skip = true;
						break;
					}
				}
				if (skip) continue;
				if (fabs(cor(i, j)) > 0.5 && fabs(cor(i, j)) < 0.6)
				{

					war = true;
					fprintf(fp1, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					fprintf(fp1, " => %10.4f multicollinearity(‘½d‹¤ü«)‚Ì‹^‚¢‚ª‚ ‚è‚Ü‚·\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.6 && fabs(cor(i, j)) < 0.8)
				{
					war = true;
					fprintf(fp1, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					fprintf(fp1, " => %10.4f multicollinearity(‘½d‹¤ü«)‚Ì‹^‚¢‚ª‚©‚È‚è‚ ‚è‚Ü‚·\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.8)
				{
					war = true;
					fprintf(fp1, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					fprintf(fp1, " => %10.4f multicollinearity(‘½d‹¤ü«)‚Ì‹­‚¢‹^‚¢‚ª‚ ‚è‚Ü‚·\n", cor(i, j));
				}
			}
		}
		if (war)
		{
			fprintf(fp1, "\nmulticollinearity(‘½d‹¤ü«)‚Ì‹^‚¢‚ª‚ ‚éà–¾•Ï”‚ª‚ ‚éê‡‚Í\n");
			fprintf(fp1, "‚Ç‚¿‚ç‚©ˆê•û‚ğŠO‚µ‚ÄÄ“x•ªÍ‚·‚é‚±‚Æ‚ÅA‘½d‹¤ü«‚ğ‰ğÁ‚·‚éê‡‚ª‚ ‚è‚Ü‚·B\n");
		}

#ifdef USE_GRAPHVIZ_DOT
		{
			bool background_Transparent = false;
			int size = 20;
			bool sideways = true;
			char* filename = "multicollinearity2.txt";
			char* outformat = "png";

			utf8str utf8;
			FILE* fp = fopen(filename, "w");
			utf8.fprintf(fp, "digraph  {\n");
			if (background_Transparent)
			{
				utf8.fprintf(fp, "digraph[bgcolor=\"#00000000\"];\n");
			}
			utf8.fprintf(fp, "size=\"%d!\"\n", size);
			if (sideways)
			{
				utf8.fprintf(fp, "graph[rankdir=\"LR\"];\n");
			}
			utf8.fprintf(fp, "node [fontname=\"MS UI Gothic\" layout=circo shape=component]\n");

			for (int i = 0; i < cor.m; i++)
			{
				//utf8.fprintf(fp, "\"%s\"[color=blue shape=circle]\n", header[i + 1].c_str());
				for (int j = i + 1; j < cor.n; j++)
				{
					bool skip = false;
					for (int k = 0; k < var_indexs.size(); k++)
					{
						if (i == var_indexs[k])
						{
							skip = true;
							break;
						}
					}
					if (skip) continue;

					std::string hd1 = header[j + 1];
					std::string hd2 = header[i + 1];
					if (hd1.c_str()[0] != '\"')
					{
						hd1 = "\"" + hd1 + "\"";
					}
					if (hd2.c_str()[0] != '\"')
					{
						hd2 = "\"" + hd2 + "\"";
					}
					if (fabs(cor(i, j)) > 0.5 && fabs(cor(i, j)) < 0.6)
					{
						utf8.fprintf(fp, "%s -> %s [label=\"%8.3f\" color=olivedrab1 dir=\"both\"]\n", hd1.c_str(), hd2.c_str(), cor(i, j));
					}
					if (fabs(cor(i, j)) >= 0.6 && fabs(cor(i, j)) < 0.8)
					{
						utf8.fprintf(fp, "%s -> %s [label=\"%8.3f\" color=goldenrod3 penwidth=\"2\" dir=\"both\"]\n", hd1.c_str(), hd2.c_str(), cor(i, j));
					}
					if (fabs(cor(i, j)) >= 0.8)
					{
						utf8.fprintf(fp, "%s -> %s [label=\"%8.3f\" color=red penwidth=\"3\" dir=\"both\"]\n", hd1.c_str(), hd2.c_str(), cor(i, j));
						utf8.fprintf(fp, "%s[color=red penwidth=\"3\"]\n", hd1.c_str());
						utf8.fprintf(fp, "%s[color=red penwidth=\"3\"]\n", hd2.c_str());
					}
				}
			}
			utf8.fprintf(fp, "}\n");
			fclose(fp);

			char cmd[512];
			graphviz_path_::getGraphvizPath();
			std::string path = graphviz_path_::path_;
			if (path != "")
			{
				sprintf(cmd, "\"%s\\dot.exe\" -T%s %s -o multicollinearity2.%s", path.c_str(), outformat, filename, outformat);
			}
			else
			{
				sprintf(cmd, "dot.exe -T%s %s -o multicollinearity2.%s", outformat, filename, outformat);
			}
			system(cmd);
		}
#endif

		Matrix<dnn_double>& vif = cor;
		for (int i = 0; i < A.n; i++)
		{
			for (int j = 0; j < A.n; j++)
			{
				if (cor(i, j) > 1.0 - 1.0e-10)
				{
					vif(i, j) = 1.0 / 1.0e-10;
					continue;
				}
				vif(i, j) = 1.0 / (1.0 - cor(i, j));
			}
		}

		fprintf(fp1, "\n");
		fprintf(fp1, "‘½d‹¤ü«‚Ì[‚³‚ğ’è—Ê‰»‚µ‚½•]‰¿");
		fprintf(fp1, "[•ªUŠg‘åŒW”(variance inflation factor)VIF(‘½d‹¤ü«‚Ì‰Â”\«•]‰¿)]\n");
		fprintf(fp1, "    %-10.10s", "");
		for (int j = 0; j < A.n; j++)
		{
			fprintf(fp1, "%-10.10s", header[j + 1].c_str());
		}
		fprintf(fp1, "\n");
		fprintf(fp1, "--------------------------------------------------------------------------------------------\n");
		for (int i = 0; i < A.n; i++)
		{
			fprintf(fp1, "%-10.8s", header[i + 1].c_str());
			for (int j = 0; j < A.n; j++)
			{
				if (i == j) fprintf(fp1, "%10.8s", "-----");
				else fprintf(fp1, "%10.4f", vif(i, j));
			}
			fprintf(fp1, "\n");
		}
		fprintf(fp1, "--------------------------------------------------------------------------------------------\n");

		bool war2 = false;
		for (int i = 0; i < cor.m; i++)
		{
			for (int j = i + 1; j < cor.n; j++)
			{
				bool skip = false;
				for (int k = 0; k < var_indexs.size(); k++)
				{
					if (i == var_indexs[k])
					{
						skip = true;
						break;
					}
				}
				if (skip) continue;

				if (fabs(cor(i, j)) >= 10)
				{
					war2 = true;
					fprintf(fp1, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					fprintf(fp1, " => %10.4f multicollinearity(‘½d‹¤ü«)‚Ì‹­‚¢‹^‚¢‚ª‚ ‚è‚Ü‚·\n", cor(i, j));
				}
			}
		}
		if (war2)
		{
			fprintf(fp1, "multicollinearity(‘½d‹¤ü«)‚Ì‹^‚¢‚ª‚ ‚éà–¾•Ï”‚ª‚ ‚éê‡‚Í\n");
			fprintf(fp1, "‚Ç‚¿‚ç‚©ˆê•û‚ğŠO‚µ‚ÄÄ“x•ªÍ‚·‚é‚±‚Æ‚ÅA‘½d‹¤ü«‚ğ‰ğÁ‚·‚éê‡‚ª‚ ‚è‚Ü‚·B\n");
		}
		else
		{
			fprintf(fp1, "VIF’l‚©‚ç‚Ímulticollinearity(‘½d‹¤ü«)‚Ì‹^‚¢‚ª‚ ‚éà–¾•Ï”‚Í–³‚³‚»‚¤‚Å‚·\n");
		}

		if (war || war2)
		{
			fprintf(fp1, "multicollinearity(‘½d‹¤ü«)‚ª‰ğÁ‚Å‚«‚È‚¢ê‡‚ÍLasso‰ñ‹A‚à‚µ‚Ä‰º‚³‚¢B\n");
		}
		if (fp == stdout || fp == NULL || filename == "")
		{
			/* empty */
		}
		else
		{
			if (fp) fclose(fp);
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
		printf("LassoRegression\n");
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
		}
			beta = beta.zeros(1, train.n);
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
		printf("ElasticNetRegression\n");
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
		}
			beta = beta.zeros(1, train.n);
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
		printf("RidgeRegression\n");
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
		}
			beta = beta.zeros(1, train.n);
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
