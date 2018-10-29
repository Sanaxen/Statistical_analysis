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
	//�Ή�A�W�������߂�
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

		//�d��A�̐��K�������𐶐�����
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

		error = les.fit(Sa, Sb);
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
		//printf("SR(�ϓ������a):%f\n", sr);

		st = sr + se;
		//printf("ST(SE+SR):%f\n", st);

		r2 = sr / (sr + se);
		//printf("R^2(����W��(��^��)):%f\n", r2);
		//printf("�d���֌W��:%f\n", sqrt(r2));

		const dnn_double ��T = A.m - 1;
		const dnn_double ��R = 2;
		const dnn_double ��e = A.m - A.n - 1;

		dnn_double Syy = 0.0;
		for (int i = 0; i < A.m; i++)
		{
			Syy += (B(i, 0) - mean_y)*(B(i, 0) - mean_y);
		}
		//printf("Syy:%f\n", Syy);

		r_2 = 1.0 - (se / ��e) / (Syy / ��T);
		//printf("���R�x�����ϊ�^��(�␳):%f\n", r_2);

		ve = se / (A.m - A.n - 1);
		vr = sr / A.n;
		//printf("�s�Ε��U VE:%f VR:%f\n", ve, vr);
		//printf("F�l:%f\n", vr / ve);
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

		//printf("�W���덷(bias):%f\n", se_ii[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("�W���덷:%f\n", se_ii[i]);
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

		//printf("t�l(bias):%f\n", t_value[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("t�l:%f\n", t_value[i]);
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
		//printf("p�l(bias):%f\n", p_value[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("p�l:%f\n", p_value[i]);
		//}
		return error;
	}

	double predict(Matrix<dnn_double>& x)
	{
		dnn_double y = bias;
		for (int j = 0; j < x.n; j++)
		{
			y += x(0, j)*les.coef(j, 0);
		}
		return y;
	}

	void report(std::vector<std::string>& header, double �� = 0.05)
	{
		printf("--------------------------------------------------------------------\n");
		printf("SE(�c��)                :%f\n", se);
		printf("SR(�ϓ������a)          :%f\n", sr);
		printf("ST(SE+SR)               :%f\n", st);
		printf("R^2(����W��(��^��))   :%f\n", r2);
		printf("�d���֌W��              :%f\n", sqrt(r2));
		printf("���R�x�����ϊ�^��(�␳):%f\n", r_2);
		printf("�s�Ε��U(���ϕ���)       VE:%f VR:%f\n", ve, vr);
		printf("AIC                     :%f\n\n", AIC);

		F_distribution f_distribution(A.n - 1, A.m - A.n);
		double f_pdf = f_distribution.p_value(��);

		if (f_distribution.status != 0)
		{
			printf("f_distribution status:%d\n", f_distribution.status);
		}
		printf("F�l:%f > F(%.2f)_%d,%d=[%.2f]", F_value, ��, A.n - 1, A.m - A.n, f_pdf);
		if (F_value > f_pdf)
		{
			printf("=>�\���ɗL���ł���ƌ��_�ł���\n");
		}
		else
		{
			printf("=>�\���ɗL���Ƃ͌����Ȃ��ƌ��_�ł���\n");
		}

		Student_t_distribution t_distribution(A.m - A.n - 1);

		printf("\n�W���E�萔���̐���(�M����:%.2f%%)\n", 100 * (1 - ��));
		//printf("(bias)%f ", bias - t_distribution.p_value(�� / 2.0)*se_ii[A.n]);
		//printf("%f\n", bias + t_distribution.p_value(�� / 2.0)*se_ii[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("%f ", les.x(i, 0) - t_distribution.p_value(�� / 2.0)*se_ii[i]);
		//	printf("%f\n", les.x(i, 0) + t_distribution.p_value(�� / 2.0)*se_ii[i]);
		//}

		printf("---------------------------------------------------------------------------------------------\n");
		printf("               �W��     �W���덷    t�l      p�l    ����(%.1f%%)  ���(%.1f%%)  �W����0�̉\��\n", 100 * (1 - ��), 100 * (1 - ��));
		printf("---------------------------------------------------------------------------------------------\n");
		printf("           %10.4f", bias);
		printf("%10.4f", se_ii[A.n]);
		printf("%10.4f", t_value[A.n]);
		printf("%10.4f", p_value[A.n]);

		double min_c = bias - t_distribution.p_value(�� / 2.0)*se_ii[A.n];
		double max_c = bias + t_distribution.p_value(�� / 2.0)*se_ii[A.n];
		printf("%10.4f", min_c);
		printf("%10.4f", max_c);
		if (min_c*max_c < 0.0) printf("          True\n");
		else printf("          False\n");

		for (int i = 0; i < A.n; i++)
		{
			printf("%-10.8s %10.4f", header[i + 1].c_str(), les.coef(i, 0));
			printf("%10.4f", se_ii[i]);
			printf("%10.4f", t_value[i]);
			printf("%10.4f", p_value[i]);

			double min_c = les.coef(i, 0) - t_distribution.p_value(�� / 2.0)*se_ii[i];
			double max_c = les.coef(i, 0) + t_distribution.p_value(�� / 2.0)*se_ii[i];

			printf("%10.4f", min_c);
			printf("%10.4f", max_c);
			if (min_c*max_c < 0.0) printf("          True\n");
			else printf("          False\n");
		}
		printf("--------------------------------------------------------------------------------------------\n");

		printf("��p�l���������قǌW���̓[���ł͖����ƍl������D\n");
		printf("��AIC�̒l�͏������قǓ��Ă͂܂肪�悢�Ƃ���Ă���\n");
		printf("���f�[�^�_�O�Ő��`�̊֌W�����藧�ۏ؂͂���܂���B\n\n");

		if (r2 <= 0.6)
		{
			printf("����W����0.6�ȉ��Ȃ̂ŖړI�ϐ������܂������ł��Ă��Ȃ������m��܂���B\n");
		}
		if (r2 > 0.6 && r2 < 0.7)
		{
			printf("����W����0.6�𒴂��Ă���̂ŖړI�ϐ����r�I�悭�����ł��Ă��܂��B\n");
		}
		if (r2 >= 0.7 && r2 < 0.9)
		{
			printf("����W����0.7�ȏ�Ȃ̂ŖړI�ϐ���ǂ������ł��Ă��܂��B\n");
		}
		if (r2 >= 0.9)
		{
			printf("����W����0.9�ȏ�Ȃ̂ŖړI�ϐ�����ɗǂ������ł��Ă��܂��B\n");
		}
		printf("������W���͐����ϐ���ǉ�����΁C���ꂪ���ł���(���Ӗ��Ȃ��̂ł�)�K����������_�ɒ��ӂ��ĉ�����\n");
		printf("\n");

		bool war = false;
		Matrix<dnn_double>& cor = A.Cor();

		printf("[���֌W��(���d�������̉\���]��)]\n");
		printf("    %-10.8s", "");
		for (int j = 0; j < A.n; j++)
		{
			printf("%-10.8s", header[j + 1].c_str());
		}
		printf("\n");
		printf("--------------------------------------------------------------------------------------------\n");
		for (int i = 0; i < A.n; i++)
		{
			printf("%-10.8s", header[i + 1].c_str());
			for (int j = 0; j < A.n; j++)
			{
				printf("%10.4f", cor(i, j));
			}
			printf("\n");
		}
		printf("--------------------------------------------------------------------------------------------\n");

		for (int i = 0; i < cor.m; i++)
		{
			for (int j = i + 1; j < cor.n; j++)
			{
				if (fabs(cor(i, j)) > 0.5 && fabs(cor(i, j)) < 0.6)
				{

					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(���d������)�̋^��������܂�\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.6 && fabs(cor(i, j)) < 0.8)
				{
					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(���d������)�̋^�������Ȃ肠��܂�\n", cor(i, j));
				}
				if (fabs(cor(i, j)) >= 0.8)
				{
					war = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(���d������)�̋����^��������܂�\n", cor(i, j));
				}
			}
		}
		if (war)
		{
			printf("multicollinearity(���d������)�̋^������������ϐ�������ꍇ��\n");
			printf("�ǂ��炩������O���čēx���͂��邱�ƂŁA���d����������������ꍇ������܂��B\n");
		}
		else
		{
			printf("multicollinearity(���d������)�̋^������������ϐ��͖��������ł�\n");
		}


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

		printf("\n");
		printf("���d�������̐[�������ʉ������]��");
		printf("[���U�g��W��(variance inflation factor)VIF(���d�������̉\���]��)]\n");
		printf("    %-10.8s", "");
		for (int j = 0; j < A.n; j++)
		{
			printf("%-10.8s", header[j + 1].c_str());
		}
		printf("\n");
		printf("--------------------------------------------------------------------------------------------\n");
		for (int i = 0; i < A.n; i++)
		{
			printf("%-10.8s", header[i + 1].c_str());
			for (int j = 0; j < A.n; j++)
			{
				if (i == j) printf("%10.8s", "-----");
				else printf("%10.4f", vif(i, j));
			}
			printf("\n");
		}
		printf("--------------------------------------------------------------------------------------------\n");

		bool war2 = false;
		for (int i = 0; i < cor.m; i++)
		{
			for (int j = i + 1; j < cor.n; j++)
			{
				if (fabs(cor(i, j)) >= 10)
				{
					war2 = true;
					printf("%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
					printf(" => %10.4f multicollinearity(���d������)�̋����^��������܂�\n", cor(i, j));
				}
			}
		}
		if (war2)
		{
			printf("multicollinearity(���d������)�̋^������������ϐ�������ꍇ��\n");
			printf("�ǂ��炩������O���čēx���͂��邱�ƂŁA���d����������������ꍇ������܂��B\n");
		}
		else
		{
			printf("VIF�l�����multicollinearity(���d������)�̋^������������ϐ��͖��������ł�\n");
		}

		if (war || war2)
		{
			printf("multicollinearity(���d������)�������ł��Ȃ��ꍇ��Lasso��A�������ĉ������B\n");
		}
	}
};
#endif
