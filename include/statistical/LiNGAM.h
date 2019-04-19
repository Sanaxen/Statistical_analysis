#ifndef __LINGAM_H__
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define __LINGAM_H__

#include "../../include/Matrix.hpp"
#include "../../include/statistical/fastICA.h"
#include "../../include/hungarian-algorithm/Hungarian.h"
#include "../../include/statistical/LinearRegression.h"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/utf8_printf.hpp"

//#define USE_EIGEN

#ifdef USE_EIGEN
#include "../../include/statistical/RegularizationRegression_eigen_version.h"
#endif

template<class T>
class VectorIndex
{
public:
	T dat;
	T abs_dat;
	int id;
	bool zero_changed;

	VectorIndex()
	{
		zero_changed = false;
	}
	bool operator<(const VectorIndex& right) const {
		return abs_dat < right.abs_dat;
	}
};

/*
S.Shimizu, P.O.Hoyer, A.Hyvarinen and A.Kerminen.
A linear non-Gaussian acyclic model for causal discovery (2006)
https://qiita.com/m__k/items/3090090429bd1b0db13e
*/
class Lingam
{
	int error;
	int variableNum;

	/*
	S.Shimizu, P.O.Hoyer, A.Hyvarinen and A.Kerminen.
	A linear non-Gaussian acyclic model for causal discovery (2006)
	5.2 Permuting B to Get a Causal Order
	Algorithm B: Testing for DAGness, and returning a causal order if true
	*/
	bool AlgorithmB(Matrix<dnn_double>& Bhat, std::vector<int>& p)
	{
		printf("\nAlgorithmB start\n");
		Bhat.print("start");

		for (int i = 0; i < Bhat.m; i++)
		{
			bool skipp = false;
			for (int k = 0; k < p.size(); k++)
			{
				if (p[k] == i)
				{
					skipp = true;
					break;
				}
			}
			if (skipp) continue;

			double sum = 0.0;
			for (int j = 0; j < Bhat.n; j++)
			{
				bool skipp = false;
				for (int k = 0; k < p.size(); k++)
				{
					if (p[k] == j)
					{
						skipp = true;
						break;
					}
				}
				if (skipp) continue;

				sum += fabs(Bhat(i, j));
			}
			if (fabs(sum) < 1.0e-16)
			{
				p.push_back(i);
			}
		}
		for (int x = 0; x < p.size(); x++)
			std::cout << x << "," << p[x] << "\t";
		printf("\n");
		fflush(stdout);

		printf("AlgorithmB end\n");
		return (p.size() == Bhat.m);
	}

	/*
	S.Shimizu, P.O.Hoyer, A.Hyvarinen and A.Kerminen.
	A linear non-Gaussian acyclic model for causal discovery (2006)
	5.2 Permuting B to Get a Causal Order
	Algorithm C: Finding a permutation of B by iterative pruning and testing
	*/
	Matrix<dnn_double> AlgorithmC(Matrix<dnn_double>& b_est_tmp, int n)
	{
		Matrix<dnn_double> b_est = b_est_tmp;
		const int N = int(n * (n + 1) / 2) - 1;

		dnn_double min_val = 0.0;
		std::vector<VectorIndex<dnn_double>> tmp;
		for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
		{
			VectorIndex<dnn_double> d;
			d.dat = b_est_tmp.v[i];
			d.abs_dat = fabs(b_est_tmp.v[i]);
			d.id = i;
			tmp.push_back(d);
		}
		//��Βl�����������Ƀ\�[�g
		std::sort(tmp.begin(), tmp.end());
		//b_est_tmp �����̐�Βl������������ n(n+1)/2 �� 0 �ƒu��������
		int nn = 0;
		for (int i = 0; i < tmp.size(); i++)
		{
			if (nn >= N) break;
			if (tmp[i].zero_changed) continue;
			printf("[%d]%f ", i, tmp[i].dat);
			tmp[i].dat = 0.0;
			tmp[i].abs_dat = 0.0;
			tmp[i].zero_changed = true;
			nn++;
		}

		printf("\nAlgorithmC start\n");
		int c = 0;
		std::vector<int> p;
		p.clear();

		while (p.size() < n)
		{
			c++;
			Matrix<dnn_double> B = b_est_tmp;
			bool stat = AlgorithmB(B, p);
			if (stat)
			{
				replacement = p;
				break;
			}

			for (int i = 0; i < tmp.size(); i++)
			{
				if (tmp[i].zero_changed) continue;
				printf("[%d]%f->0.0\n", i, tmp[i].dat);
				tmp[i].dat = 0.0;
				tmp[i].abs_dat = 0.0;
				tmp[i].zero_changed = true;
				break;
			}
			//b_est_tmp �����ɖ߂�
			for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
			{
				b_est_tmp.v[tmp[i].id] = tmp[i].dat;
			}
		}
		for (int x = 0; x < replacement.size(); x++)
			std::cout << x << "," << replacement[x] << "\t";
		printf("\nreplacement.size()=%d\n", replacement.size());

		std::vector<int> &r = replacement;

		//b_opt = b_opt[r, :]
		Matrix<dnn_double> b_opt = (Substitution(r)*b_est);
		//b_opt = b_opt[:, r]
		b_opt = (b_opt)*Substitution(r).transpose();
		b_opt.print_e();

		//b_csl = np.tril(b_opt, -1)
		Matrix<dnn_double> b_csl = b_opt;
		for (int i = 0; i < b_csl.m; i++)
		{
			for (int j = i; j < b_csl.n; j++)
			{
				b_csl(i, j) = 0.0;
			}
		}
		b_csl.print_e();

		if (0)
		{
			//���בւ������ɖ߂�
			//b_csl[r, :] = deepcopy(b_csl)
			b_csl = (Substitution(r).transpose()*b_csl);
			//b_csl.print_e();

			//b_csl[:, r] = deepcopy(b_csl)
			b_csl = (b_csl*Substitution(r));
			//b_csl.print_e();

			for (int i = 0; i < B.n; i++)
			{
				r[i] = i;
			}
		}
		printf("AlgorithmC end\n");
		fflush(stdout);
		return b_csl;
	}

public:

	vector<int> replacement;
	Matrix<dnn_double> B;
	Matrix<dnn_double> input;

	Lingam() {
		error = -999;
	}

	void set(int variableNum_)
	{
		variableNum = variableNum_;
	}

	int remove_redundancy(const dnn_double alpha = 0.01, const size_t max_ica_iteration = 1000000, const dnn_double tolerance = TOLERANCE)
	{
		error = 0;
		Matrix<dnn_double> xs = input;
		//xs.print();
		Matrix<dnn_double> X = xs.Col(replacement[0]);
		Matrix<dnn_double> Y = xs.Col(replacement[1]);
		for (int i = 1; i < B.m; i++)
		{
			//X.print();
			//Y.print();
			size_t n_iter = max_ica_iteration;
#ifdef USE_EIGEN
			Lasso lasso(alpha, n_iter, tolerance);
#else
			LassoRegression lasso(alpha, n_iter, tolerance);
#endif
			lasso.fit(X, Y);
			while (lasso.getStatus() != 0)
			{
				error = -1;
				//return error;

				//n_iter *= 2;
				//lasso.fit(X, Y, n_iter, tolerance);
				printf("n_iter=%d error_eps=%f\n", lasso.num_iteration,lasso.error_eps);
				break;
			}

#ifdef USE_EIGEN
			Matrix<dnn_double>& c = formEigenVectorXd(lasso.coef_);
			for (int k = 0; k < i; k++)
			{
				c.v[k] = c.v[k] / lasso.scalaer.var_[k];
				B(i, k) = c.v[k];
			}
#else
			const Matrix<dnn_double>& c = lasso.coef;
			for (int k = 0; k < i; k++)
			{
				B(i, k) = c.v[k] / lasso.sigma(0, k);
			}
#endif
			if (i == B.m) break;
			//c.print();
			X = X.appendCol(Y);
			Y = xs.Col(replacement[i + 1]);
		}
		B.print_e("remove_redundancy");
		return error;
	}

	std::vector<std::string> linear_regression_var;

	void digraph(const std::vector<std::string>& column_names, std::vector<std::string> y_var, std::vector<int>& residual_flag, const char* filename, bool sideways = false, int size=30, char* outformat="png", bool background_Transparent=false)
	{
		Matrix<dnn_double> B_tmp = B.chop(0.001);
		B_tmp.print_e("remove 0.001");

		std::vector<std::string> item;
		item.resize(B.n);

#if 0
		//for (int i = 0; i < B.n; i++)
		//{
		//	item[i] = column_names[replacement[i]];
		//}

		//Matrix<dnn_double> XCor = input;
		//XCor = (XCor)*Substitution(replacement);
		//XCor = XCor.Cor();
#else
		for (int i = 0; i < B.n; i++)
		{
			item[i] = column_names[i];
		}
		Matrix<dnn_double> XCor = input.Cor();
#endif

		utf8str utf8;
		FILE* fp = fopen(filename, "w");
		utf8.fprintf(fp, "digraph {\n");
		if (background_Transparent)
		{
			utf8.fprintf(fp, "graph[bgcolor=\"#00000000\"];\n");
		}
		utf8.fprintf(fp, "size=\"%d!\"\n", size);
		if (sideways)
		{
			utf8.fprintf(fp, "graph[rankdir=\"LR\"];\n");
		}
		utf8.fprintf(fp, "node [fontname=\"MS UI Gothic\" layout=circo shape=note]\n");

		for (int i = 0; i < B_tmp.n; i++)
		{
			std::string item1 = item[i];
			if (item1.c_str()[0] != '\"')
			{
				item1 = "\"" + item1 + "\"";
			}

			if (residual_flag[i])
			{
				utf8.fprintf(fp, "%s [fillcolor=lightgray, style=\"filled\"]\n", item1.c_str());
				utf8.fprintf(fp, "%s[color=lightgray shape=rectangle]\n", item1.c_str());
			}
			else
			{
				utf8.fprintf(fp, "%s[color=blue shape=note]\n", item1.c_str());
			}
			
			for (int j = 0; j < B_tmp.n; j++)
			{
				if (B_tmp(i, j) != 0.0)
				{
					std::string item1 = item[i];
					std::string item2 = item[j];
					if (item1.c_str()[0] != '\"')
					{
						item1 = "\"" + item1 + "\"";
					}
					if (item2.c_str()[0] != '\"')
					{
						item2 = "\"" + item2 + "\"";
					}
					bool out_line = false;
					bool in_line = false;
					for (int k = 0; k < y_var.size(); k++)
					{
						std::string x = y_var[k];
						if (x.c_str()[0] != '\"')
						{
							x = "\"" + x + "\"";
						}

						if (item1 == x)
						{
							in_line = true;
						}
						if (item2 == x)
						{
							out_line = true;
						}
					}

					char* style = "";
					if (residual_flag[i])
					{
						style = "style=\"dotted\"";
					}
					if (out_line)
					{
						utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=red penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
					}
					else
						if (in_line)
						{
							linear_regression_var.push_back(item[j]);
							utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
						}
						else
						{
							utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=black %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i,j), style);
						}
				}
			}
		}
		for (int i = 0; i < y_var.size(); i++)
		{
			//����2[fillcolor="#ccddff", style="filled"];
			std::string item1 = y_var[i];
			if (item1.c_str()[0] != '\"')
			{
				item1 = "\"" + item1 + "\"";
			}
			utf8.fprintf(fp, "%s [fillcolor=\"#ccddff\", style=\"filled\"]\n", item1.c_str());
		}

		utf8.fprintf(fp, "}\n");
		fclose(fp);
#ifdef USE_GRAPHVIZ_DOT
		char cmd[512];
		sprintf(cmd, "dot.exe -T%s %s -o Digraph.%s", outformat, filename, outformat);
		system(cmd);
#endif
	}

	void report(const std::vector<std::string>& column_names)
	{
		printf("=======	Cause - and-effect diagram =======\n");
		for (int i = 0; i < B.n; i++)
		{
			for (int j = 0; j < B.n; j++)
			{
				if (B(i, j) != 0.0)
				{
					printf("%s --[%6.3f]--> %s\n", column_names[j].c_str(), B(i, j), column_names[i].c_str());
				}
			}
		}
		printf("------------------------------------------\n");
	}
	void before_sorting()
	{
		std::vector<int> &r = replacement;
		Matrix<dnn_double> b_csl = B;

		//���בւ������ɖ߂�
		//b_csl[r, :] = deepcopy(b_csl)
		b_csl = (Substitution(r).transpose()*b_csl);
		//b_csl.print_e();

		//b_csl[:, r] = deepcopy(b_csl)
		b_csl = (b_csl*Substitution(r));
		//b_csl.print_e();

		for (int i = 0; i < B.n; i++)
		{
			r[i] = i;
		}
		B = b_csl;
	}

	int fit(Matrix<dnn_double>& X, const int max_ica_iteration= MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
		error = 0;

		input = X;
		Matrix<dnn_double> xs = X;

		ICA ica;
		ica.set(variableNum);
		ica.fit(xs, max_ica_iteration, tolerance);
		(ica.A.transpose()).inv().print_e();
		error = ica.getStatus();


		Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv();
		Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal();

		HungarianAlgorithm HungAlgo;
		vector<int> replace;

		double cost = HungAlgo.Solve(W_ica_, replace);

		for ( int x = 0; x < W_ica_.m; x++)
			std::cout << x << "," << replace[x] << "\t";
		printf("\n");

		Matrix<dnn_double>& ixs = toMatrix(replace);
		ixs.print();
		Substitution(replace).print("Substitution matrix");

		//P^-1*Wica
		Matrix<dnn_double>& W_ica_perm = (Substitution(replace).inv()*W_ica);
		W_ica_perm.print_e("Replacement->W_ica_perm");

		//D^-1
		Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
		Matrix<dnn_double> D2(diag_vector(D));
		(D2.Reciprocal()).print_e("1/D");

		//W_ica_perm_D=I - D^-1*(P^-1*Wica)
		Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal()));

		W_ica_perm_D.print_e("W_ica_perm_D");

		//B=I - D^-1*(P^-1*Wica)
		Matrix<dnn_double>& b_est = Matrix<dnn_double>().unit(W_ica_perm_D.m, W_ica_perm_D.n) - W_ica_perm_D;
		b_est.print_e("b_est");

#if 10
		//https://www.cs.helsinki.fi/u/ahyvarin/papers/JMLR06.pdf
		const int n = W_ica_perm_D.m;
		Matrix<dnn_double> b_est_tmp = b_est;
		b_est = AlgorithmC(b_est_tmp, n);
#else
		//All case inspection
		std::vector<std::vector<int>> replacement_list;
		std::vector<int> v(W_ica_perm_D.m);
		std::iota(v.begin(), v.end(), 0);       // v �� 0, 1, 2, ... N-1 ��ݒ�
		do {
			std::vector<int> replacement_case;

			for (auto x : v) replacement_case.push_back(x);
			replacement_list.push_back(replacement_case);
			//for (auto x : v) cout << x << " "; cout << "\n";    // v �̗v�f��\��
		} while (next_permutation(v.begin(), v.end()));     // ���̏���𐶐�

		const int n = W_ica_perm_D.m;
		const int N = int(n * (n + 1) / 2) - 1;

		//for (int i = 0; i < b_est.m*b_est.n; i++)
		//{
		//	if (fabs(b_est.v[i]) < 1.0e-8) b_est.v[i] = 0.0;
		//}

		Matrix<dnn_double> b_est_tmp = b_est;

		dnn_double min_val = 0.0;
		std::vector<VectorIndex<dnn_double>> tmp;
		for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
		{
			VectorIndex<dnn_double> d;
			d.dat = b_est_tmp.v[i];
			d.abs_dat = fabs(b_est_tmp.v[i]);
			d.id = i;
			tmp.push_back(d);
		}
		//��Βl�����������Ƀ\�[�g
		std::sort(tmp.begin(), tmp.end());

		int N_ = N;
		bool tri_ng = false;
		do
		{
			//b_est_tmp �����̐�Βl������������ n(n+1)/2 �� 0 �ƒu��������
			int nn = 0;
			for (int i = 0; i < tmp.size(); i++)
			{
				if (nn >= N_) break;
				if (tmp[i].zero_changed) continue;
				//printf("[%d]%f ", i, tmp[i].dat);
				tmp[i].dat = 0.0;
				tmp[i].abs_dat = 0.0;
				tmp[i].zero_changed = true;
				nn++;
			}
			//printf("\n");
			N_ = 1;	//���͎��ɐ�Βl�������������� 0 �ƒu���čĂъm�F

					//b_est_tmp �����ɖ߂�
			for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
			{
				b_est_tmp.v[tmp[i].id] = tmp[i].dat;
			}
			b_est_tmp.print_e();
			if (b_est_tmp.isZero(1.0e-8))
			{
				error = -1;
				break;
			}

			tri_ng = false;
			//�s����ёւ��ĉ��O�p�s��ɂł��邩���ׂ�B
			for (int k = 0; k < replacement_list.size(); k++)
			{
				//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v �̗v�f��\��

				Matrix<dnn_double> tmp = Substitution(replacement_list[k])*b_est_tmp;
				//�����s�񂩃`�F�b�N
				for (int i = 0; i < tmp.m; i++)
				{
					for (int j = i; j < tmp.n; j++)
					{
						if (tmp(i, j) != 0.0)
						{
							tri_ng = true;
							break;
						}
					}
					if (tri_ng) break;
				}

				if (!tri_ng)
				{
					b_est = Substitution(replacement_list[k])*b_est;
					for (int i = 0; i < b_est.m; i++)
					{
						for (int j = i; j < b_est.n; j++)
						{
							b_est(i, j) = 0.0;
						}
					}
					replacement = replacement_list[k];
					//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v �̗v�f��\��
					break;
				}
			}
		} while (tri_ng);
#endif

		for (int x = 0; x < replacement.size(); x++)
			std::cout << x << "," << replacement[x] << "\t";
		printf("\n");
		b_est.print_e();
		fflush(stdout);

		if ( error == 0 ) B = b_est;
		
		return error;
	}
};
#endif
