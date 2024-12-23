#ifndef _LINERREGRESSOR_H
#define _LINERREGRESSOR_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

//#include "Matrix.hpp"
#include "../../include/Matrix.hpp"
#include "../../include/util/mathutil.h"
#include "../../include/util/utf8_printf.hpp"
#include "../../include/util/graphviz_util.h"

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

	int fit(Matrix<dnn_double>A_, Matrix<dnn_double>B_, bool test = false, const std::vector<std::string>& header = {})
	{
		A = A_;
		B = B_;

		if (A.m - A.n - 1 <= 0)
		{
			printf("データ数より説明変数の方が多すぎます\n");
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

		if (!test)
		{
			error = les.fit(Sa, Sb);
			printf("error=%d\n", error);
		}
		else
		{
			error = 0;
		}
		//les.x.print("x");
		if (error != 0) return error;

		if (!test)
		{
			bias = mean_y;
			for (int i = 0; i < A.n; i++) bias -= les.coef(i, 0)*mean_x[i];
			//printf("bias:%f\n", bias);
		}

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
		if (test)
		{
			FILE* fp = fopen("predict.csv", "w");
			if (fp)
			{
				for (int j = 0; j < A.n; j++)
				{
					fprintf(fp, "%s,", header[j+1].c_str());
				}
				fprintf(fp, "predict_value:%s\n", header[0].c_str());

				for (int i = 0; i < A.m; i++)
				{
					for (int j = 0; j < A.n; j++)
					{
						fprintf(fp, "%.3f,", A(i, j));
					}
					fprintf(fp, "%.3f\n", y_predict[i]);
				}
				fclose(fp);
			}
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

	double predict(Matrix<dnn_double>& x)
	{
		dnn_double y = bias;
		for (int j = 0; j < x.n; j++)
		{
			y += x(0, j)*les.coef(j, 0);
		}
		return y;
	}

	void report(std::string& filename, std::vector<std::string>& header, double α = 0.05)
	{
		FILE* fp = stdout;
		if (filename != "")
		{
			fp = fopen(filename.c_str(), "w");
			if (!fp) return;
		}
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "SE(残差)                :%.4f\n", se);
		fprintf(fp, "SR(変動平方和)          :%.4f\n", sr);
		fprintf(fp, "ST(SE+SR)               :%.4f\n", st);
		fprintf(fp, "R^2(決定係数(寄与率))   :%.4f\n", r2);
		fprintf(fp, "重相関係数              :%.4f\n", sqrt(r2));
		fprintf(fp, "自由度調整済寄与率(補正):%.4f\n", r_2);
		fprintf(fp, "不偏分散(平均平方)       VE:%.4f VR:%.4f\n", ve, vr);
		fprintf(fp, "AIC                     :%.4f\n\n", AIC);

		F_distribution f_distribution(A.n - 1, A.m - A.n);
		double f_pdf = f_distribution.p_value(α);

		if (f_distribution.status != 0)
		{
			fprintf(fp, "f_distribution status:%d\n", f_distribution.status);
		}
		fprintf(fp, "F値:%f > F(%.2f)_%d,%d=[%.2f]", F_value, α, A.n - 1, A.m - A.n, f_pdf);
		if (F_value > f_pdf)
		{
			fprintf(fp, "=>予測に有効であると結論できる\n");
		}
		else
		{
			fprintf(fp, "=>予測に有効とは言えないと結論できる\n");
		}

		Student_t_distribution t_distribution(A.m - A.n - 1);

		fprintf(fp, "\n係数・定数項の推定(信頼幅:%.2f%%)\n", 100 * (1 - α));
		//printf("(bias)%f ", bias - t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		//printf("%f\n", bias + t_distribution.p_value(α / 2.0)*se_ii[A.n]);
		//for (int i = 0; i < A.n; i++)
		//{
		//	printf("%f ", les.x(i, 0) - t_distribution.p_value(α / 2.0)*se_ii[i]);
		//	printf("%f\n", les.x(i, 0) + t_distribution.p_value(α / 2.0)*se_ii[i]);
		//}

		fprintf(fp, "---------------------------------------------------------------------------------------------\n");
		fprintf(fp, "               係数     標準誤差    t値      p値    下限(%.1f%%)  上限(%.1f%%)  係数が0の可能性\n", 100 * (1 - α), 100 * (1 - α));
		fprintf(fp, "---------------------------------------------------------------------------------------------\n");
		fprintf(fp, "           %10.4f", bias);
		fprintf(fp, " %10.4f", se_ii[A.n]);
		fprintf(fp, " %10.4f", t_value[A.n]);
		fprintf(fp, " %10.4f", p_value[A.n]);

		double min_c = bias - t_distribution.p_value(α / 2.0)*se_ii[A.n];
		double max_c = bias + t_distribution.p_value(α / 2.0)*se_ii[A.n];
		fprintf(fp, " %10.4f", min_c);
		fprintf(fp, " %10.4f", max_c);
		if (min_c*max_c < 0.0) fprintf(fp, "          ○\n");
		else fprintf(fp, "          ×\n");

		for (int i = 0; i < A.n; i++)
		{
			fprintf(fp, " %-10.8s %10.4f", header[i + 1].c_str(), les.coef(i, 0));
			fprintf(fp, " %10.4f", se_ii[i]);
			fprintf(fp, " %10.4f", t_value[i]);
			fprintf(fp, " %10.4f", p_value[i]);

			double min_c = les.coef(i, 0) - t_distribution.p_value(α / 2.0)*se_ii[i];
			double max_c = les.coef(i, 0) + t_distribution.p_value(α / 2.0)*se_ii[i];

			fprintf(fp, " %10.4f", min_c);
			fprintf(fp, " %10.4f", max_c);
			if (min_c*max_c < 0.0) fprintf(fp, "          ○\n");
			else fprintf(fp, "          ×\n");
		}
		fprintf(fp, "--------------------------------------------------------------------------------------------\n");

		fprintf(fp, "※p値が小さいほど係数はゼロでは無いと考えられる．\n");
		fprintf(fp, "※AICの値は小さいほど当てはまりがよいとされている\n");
		fprintf(fp, "※データ点外で線形の関係が成り立つ保証はありません。\n\n");

		if (r2 <= 0.6)
		{
			fprintf(fp, "決定係数は0.6以下なので目的変数をうまく説明できていないかも知れません。\n");
		}
		if (r2 > 0.6 && r2 < 0.7)
		{
			fprintf(fp, "決定係数は0.6を超えているので目的変数を比較的よく説明できています。\n");
		}
		if (r2 >= 0.7 && r2 < 0.9)
		{
			fprintf(fp, "決定係数は0.7以上なので目的変数を良く説明できています。\n");
		}
		if (r2 >= 0.9)
		{
			fprintf(fp, "決定係数は0.9以上なので目的変数を非常に良く説明できています。\n");
		}
		fprintf(fp, "※決定係数は説明変数を追加すれば，それが何であれ(無意味なものでも)必ず増加する点に注意して下さい\n");
		fprintf(fp, "\n");

		if (A.n > 1)
		{
			bool war = false;
			Matrix<dnn_double>& cor = A.Cor();

			fprintf(fp, "[相関係数(多重共線性の可能性評価)]\n");
			fprintf(fp, "    %-10.8s", "");
			for (int j = 0; j < A.n; j++)
			{
				fprintf(fp, "%-10.8s", header[j + 1].c_str());
			}
			fprintf(fp, "\n");
			fprintf(fp, "--------------------------------------------------------------------------------------------\n");
			for (int i = 0; i < A.n; i++)
			{
				fprintf(fp, "%-10.8s", header[i + 1].c_str());
				for (int j = 0; j < A.n; j++)
				{
					fprintf(fp, " %10.4f", cor(i, j));
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "--------------------------------------------------------------------------------------------\n");

			for (int i = 0; i < cor.m; i++)
			{
				for (int j = i + 1; j < cor.n; j++)
				{
					if (fabs(cor(i, j)) > 0.5 && fabs(cor(i, j)) < 0.6)
					{

						war = true;
						fprintf(fp, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
						fprintf(fp, " => %10.4f multicollinearity(多重共線性)の疑いがあります\n", cor(i, j));
					}
					if (fabs(cor(i, j)) >= 0.6 && fabs(cor(i, j)) < 0.8)
					{
						war = true;
						fprintf(fp, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
						fprintf(fp, " => %10.4f multicollinearity(多重共線性)の疑いがかなりあります\n", cor(i, j));
					}
					if (fabs(cor(i, j)) >= 0.8)
					{
						war = true;
						fprintf(fp, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
						fprintf(fp, " => %10.4f multicollinearity(多重共線性)の強い疑いがあります\n", cor(i, j));
					}
				}
			}
			if (war)
			{
				fprintf(fp, "multicollinearity(多重共線性)の疑いがある説明変数がある場合は\n");
				fprintf(fp, "どちらか一方を外して再度分析することで、多重共線性を解消する場合があります。\n");
			}
			else
			{
				fprintf(fp, "multicollinearity(多重共線性)の疑いがある説明変数は無さそうです\n");
			}

#ifdef USE_GRAPHVIZ_DOT
			{
				bool background_Transparent = false;
				int size = 20;
				bool sideways = true;
				char* filename = "multicollinearity.txt";
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
					sprintf(cmd, "\"%s\\dot.exe\" -T%s %s -o multicollinearity.%s", path.c_str(), outformat, filename, outformat);
				else
					sprintf(cmd, "dot.exe -T%s %s -o multicollinearity.%s", outformat, filename, outformat);
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

			fprintf(fp, "\n");
			fprintf(fp, "多重共線性の深刻さを定量化した評価");
			fprintf(fp, "[分散拡大係数(variance inflation factor)VIF(多重共線性の可能性評価)]\n");
			fprintf(fp, "    %-10.8s", "");
			for (int j = 0; j < A.n; j++)
			{
				fprintf(fp, "%-10.8s", header[j + 1].c_str());
			}
			fprintf(fp, "\n");
			fprintf(fp, "--------------------------------------------------------------------------------------------\n");
			for (int i = 0; i < A.n; i++)
			{
				fprintf(fp, "%-10.8s", header[i + 1].c_str());
				for (int j = 0; j < A.n; j++)
				{
					if (i == j) fprintf(fp, "%10.8s", "-----");
					else fprintf(fp, "%10.4f", vif(i, j));
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "--------------------------------------------------------------------------------------------\n");

			bool war2 = false;
			for (int i = 0; i < cor.m; i++)
			{
				for (int j = i + 1; j < cor.n; j++)
				{
					if (fabs(cor(i, j)) >= 10)
					{
						war2 = true;
						fprintf(fp, "%-10.10s & %-10.10s", header[i + 1].c_str(), header[j + 1].c_str());
						fprintf(fp, " => %10.4f multicollinearity(多重共線性)の強い疑いがあります\n", cor(i, j));
					}
				}
			}
			if (war2)
			{
				fprintf(fp, "multicollinearity(多重共線性)の疑いがある説明変数がある場合は\n");
				fprintf(fp, "どちらか一方を外して再度分析することで、多重共線性を解消する場合があります。\n");
			}
			else
			{
				fprintf(fp, "VIF値からはmulticollinearity(多重共線性)の疑いがある説明変数は無さそうです\n");
			}

			if (war || war2)
			{
				fprintf(fp, "multicollinearity(多重共線性)が解消できない場合はLasso回帰も試して下さい。\n");
			}
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
};
#endif
