#ifndef _LINERREGRESSOR_H
#define _LINERREGRESSOR_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

//#include "Matrix.hpp"
#include "../../include/Matrix.hpp"
#include "../../include/util/mathutil.h"


class linear_model
{
	int error;
public:
	Matrix<dnn_double> Y;
	Matrix<dnn_double> A;
	Matrix<dnn_double> b;

	inline int getStatus() const
	{
		return error;
	}

	linear_model()
	{
		error = 0;
	}
	int fit(Matrix<dnn_double>& x, Matrix<dnn_double>& y)
	{
		A = Matrix<dnn_double>(x.m, x.n+1);
		Y = y;

		for (int i = 0; i < x.m; i++) A(i, 0) = 1.0;
		for (int i = 0; i < x.m; i++)
		{
			for (int j = 0; i < x.n; j++)
			{
				A(i, j + 1) = x(i, j);
			}
		}
		linear_east_square les;
		
		les.fit(A, Y);
		error = les.getStatus();

		if (error == 0)
		{
			b = les.coef;
		}
	}

	Matrix<dnn_double> residual()
	{
		Matrix<dnn_double>& prerict = A*b;
		return Sqr(Y - prerict);
	}
	inline dnn_double RSS()
	{
		return SumAll(residual());
	}
};


class multiple_regression
{
	int error;
	int variablesNum;
public:
	inline int getStatus() const
	{
		return error;
	}
	multiple_regression()
	{}
	Matrix<dnn_double>A;
	Matrix<dnn_double>B;
	//偏回帰係数を求める
	linear_east_square les;


	std::vector<dnn_double> mean_x;
	dnn_double mean_y;
	Matrix<dnn_double> Sa;
	Matrix<dnn_double> Sb;
	dnn_double bias;
	std::vector<dnn_double> y_predict;

	dnn_double se;
	dnn_double sr;
	dnn_double st;
	dnn_double r2;
	dnn_double r_2;
	dnn_double ve;
	dnn_double vr;
	dnn_double AIC;
	dnn_double F_value;

	std::vector<dnn_double> se_ii;
	Matrix<dnn_double> Sa_1;

	std::vector<dnn_double> t_value;
	std::vector<dnn_double> p_value;

	void set(const int variablesNum_)
	{
		variablesNum = variablesNum_;
	}

	int fit(Matrix<dnn_double>A_, Matrix<dnn_double>B_)
	{
		A = A_;
		B = B_;

		if (A.m - A.n - 1 <= 0)
		{
			error = -1;
			return error;
		}

		//重回帰の正規方程式を生成する
		mean_x.resize(A.n);
		for (int j = 0; j < A.n; j++)
		{
			dnn_double sum = 0.0;
			for (int i = 0; i < A.m; i++)
			{
				sum += A(i, j);
			}
			sum /= A.m;
			mean_x[j] = sum;
		}
		mean_y = 0.0;
		for (int i = 0; i < B.m; i++)
		{
			mean_y += B(i, 0);
		}
		mean_y /= B.m;

		Sa = Matrix<dnn_double>(A.n, A.n);
		for (int ii = 0; ii < A.n; ii++)
		{
			for (int jj = 0; jj < A.n; jj++)
			{
				dnn_double sum = 0.0;
				for (int k = 0; k < A.m; k++)
				{
					sum += (A(k, ii) - mean_x[ii])*(A(k, jj) - mean_x[jj]);
				}
				Sa(ii, jj) = sum;
			}
		}

		Sb = Matrix<dnn_double>(A.n, 1);
		for (int ii = 0; ii < A.n; ii++)
		{
			for (int jj = 0; jj < B.n; jj++)
			{
				dnn_double sum = 0.0;
				for (int k = 0; k < A.m; k++)
				{
					sum += (A(k, ii) - mean_x[ii])*(B(k, jj) - mean_y);
				}
				Sb(ii, jj) = sum;
			}
		}
		//Sa.print("Sa");
		//Sb.print("Sb");


		//printf("error=%d\n", les.fit(Sa, Sb));
		//les.x.print("x");

		error = les.fit2(Sa, Sb);
		printf("error=%d\n", error);
		//les.x.print("x");
		if (error != 0) return error;

		bias = mean_y;
		for (int i = 0; i < A.n; i++) bias -= les.coef(i, 0)*mean_x[i];
		//printf("bias:%f\n", bias);

		y_predict.clear();
		for (int i = 0; i < A.m; i++)
		{
			dnn_double y = bias;
			for (int j = 0; j < A.n; j++)
			{
				y += A(i, j)*les.coef(j, 0);
			}
			y_predict.push_back(y);
		}

		se = 0.0;
		for (int i = 0; i < A.m; i++)
		{
			se += (B(i, 0) - y_predict[i])*(B(i, 0) - y_predict[i]);
		}

		sr = 0.0;
		for (int i = 0; i < A.m; i++)
		{
			sr += (y_predict[i] - mean_y)*(y_predict[i] - mean_y);
		}
		//printf("SR(変動平方和):%f\n", sr);

		st = sr + se;
		//printf("ST(SE+SR):%f\n", st);

		r2 = sr / (sr + se);
		//printf("R^2(決定係数(寄与率)):%f\n", r2);
		//printf("重相関係数:%f\n", sqrt(r2));

		const dnn_double φT = A.m - 1;
		const dnn_double φR = 2;
		const dnn_double φe = A.m - A.n - 1;

		dnn_double Syy = 0.0;
		for (int i = 0; i < A.m; i++)
		{
			Syy += (B(i, 0) - mean_y)*(B(i, 0) - mean_y);
		}
		//printf("Syy:%f\n", Syy);

		r_2 = 1.0 - (se / φe) / (Syy / φT);
		//printf("自由度調整済寄与率(補正):%f\n", r_2);

		ve = se / (A.m - A.n - 1);
		vr = sr / A.n;
		//printf("不偏分散 VE:%f VR:%f\n", ve, vr);
		//printf("F値:%f\n", vr / ve);
		F_value = vr / ve;	//= (r2/A.n) / ((1.0 - r2)/(A.m-A.n-1)));


		Sa_1 = Sa.inv();

		se_ii.clear();

		dnn_double s = 0;
		for (int i = 0; i < A.n; i++)
		{
			s = sqrt(Sa_1(i, i)*ve);
			se_ii.push_back(s);
		}

		s = 0.0;
		for (int i = 0; i < A.n; i++)
		{
			for (int j = 0; j < A.n; j++)
			{
				s += mean_x[i] * mean_x[j] * Sa_1(i, j);
			}
		}
		se_ii.push_back(sqrt((1.0 / A.m + s)*ve));

		//printf("標準誤差(bias):%f\n", se_ii[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("標準誤差:%f\n", se_ii[i]);
		//}

		AIC = A.m*(log(2.0*M_PI*se / A.m) + 1) +  2.0*(A.n + 2.0);
		if (fabs(bias) < 1.0e-16)
		{
			AIC = A.m*(log(2.0*M_PI*se / A.m) + 1) + 2.0*(A.n + 1.0);
		}
		t_value.clear();
		for (int i = 0; i < A.n; i++)
		{
			t_value.push_back(les.coef(i, 0) / se_ii[i]);
		}
		t_value.push_back(bias / se_ii[A.n]);

		//printf("t値(bias):%f\n", t_value[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("t値:%f\n", t_value[i]);
		//}

		p_value.clear();
		Student_t_distribution t_distribution(A.m - A.n - 1);
		for (int i = 0; i < A.n + 1; i++)
		{
			double tt;
			double pvalue;

			if (t_value[i] < 0)
			{
				pvalue = t_distribution.distribution(t_value[i], &tt)*2.0;
			}
			else
			{
				pvalue = t_distribution.distribution(-t_value[i], &tt)*2.0;
			}
			p_value.push_back(pvalue);
		}
		//printf("p値(bias):%f\n", p_value[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("p値:%f\n", p_value[i]);
		//}
		return error;
	}

	void report(double α = 0.05)
	{
		printf("--------------------------------------------------------------------\n");
		printf("SE(残差)                :%f\n", se);
		printf("SR(変動平方和)          :%f\n", sr);
		printf("ST(SE+SR)               :%f\n", st);
		printf("R^2(決定係数(寄与率))   :%f\n", r2);
		printf("重相関係数              :%f\n", sqrt(r2));
		printf("自由度調整済寄与率(補正):%f\n", r_2);
		printf("不偏分散(平均平方)       VE:%f VR:%f\n", ve, vr);
		printf("AIC                     :%f\n\n", AIC);

		F_distribution f_distribution(A.n - 1, A.m - A.n);
		double f_pdf = f_distribution.p_value(α);

		if (f_distribution.status != 0)
		{
			printf("f_distribution status:%d\n", f_distribution.status);
		}
		printf("F値:%f > F(%.2f)_%d,%d=[%.2f]", F_value, α, A.n - 1, A.m - A.n, f_pdf);
		if (F_value > f_pdf)
		{
			printf("=>予測に有効であると結論できる\n");
		}
		else
		{
			printf("=>予測に有効とは言えないと結論できる\n");
		}

		Student_t_distribution t_distribution(A.m - A.n - 1);

		printf("\n係数・定数項の推定(信頼幅:%.2f%%)\n", 100 * (1 - α));
		//printf("(bias)%f ", bias - t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		//printf("%f\n", bias + t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("%f ", les.x(i, 0) - t_distribution.p_value(α / 2.0)*se_ii[i]);
		//	printf("%f\n", les.x(i, 0) + t_distribution.p_value(α / 2.0)*se_ii[i]);
		//}

		printf("--------------------------------------------------------------------\n");
		printf("     係数     標準誤差    t値      p値    下限(%.1f%%)  上限(%.1f%%)\n", 100 * (1 - α), 100 * (1 - α));
		printf("--------------------------------------------------------------------\n");
		printf("%10.4f", bias);
		printf("%10.4f", se_ii[A.n]);
		printf("%10.4f", t_value[A.n]);
		printf("%10.4f", p_value[A.n]);
		printf("%10.4f", bias - t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		printf("%10.4f\n", bias + t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		for (int i = 0; i < A.n; i++)
		{
			printf("%10.4f", les.coef(i, 0));
			printf("%10.4f", se_ii[i]);
			printf("%10.4f", t_value[i]);
			printf("%10.4f", p_value[i]);
			printf("%10.4f", les.coef(i, 0) - t_distribution.p_value(α / 2.0)*se_ii[i]);
			printf("%10.4f\n", les.coef(i, 0) + t_distribution.p_value(α / 2.0)*se_ii[i]);
		}
		printf("--------------------------------------------------------------------\n");

		printf("※p値が小さいほど係数はゼロでは無いと考えられる．\n");
		printf("※AICの値は小さいほど当てはまりがよいとされている\n");

	}
};
#endif
