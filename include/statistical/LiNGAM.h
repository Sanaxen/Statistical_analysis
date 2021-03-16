#ifndef __LINGAM_H__
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define NOMINMAX
#define __LINGAM_H__
#include <chrono>

#include "../../include/Matrix.hpp"
#include "../../include/statistical/fastICA.h"
#include "../../include/hungarian-algorithm/Hungarian.h"
#include "../../include/statistical/LinearRegression.h"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/utf8_printf.hpp"
#include "../../include/util/csvreader.h"

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

/* Mutual information*/
//https://qiita.com/hyt-sasaki/items/ffaab049e46f800f7cbf
class MutualInformation
{
	void gridtabel(Matrix<dnn_double>& M1, Matrix<dnn_double>& M2)
	{
		double max1, min1;
		double max2, min2;

		max1 = M1.Max();
		min1 = M1.Min();

		max2 = M2.Max();
		min2 = M2.Min();

#if 0
		double dx = (max1 - min1) / grid;
		double dy = (max2 - min2) / grid;
#else
		max1 = (max1 > max2) ? max1 : max2;
		min1 = (min1 < min2) ? min1 : min2;
		max2 = max1;
		min2 = min1;
		double dx = (max1 - min1) / grid;
		double dy = (max2 - min2) / grid;
#endif

		//printf("dx:%f dy:%f\n", dx, dy);
		std::vector<dnn_double> table1;
		std::vector<dnn_double> table2;
		Matrix<dnn_double>table12 = Matrix<dnn_double>().zeros(grid, grid);

		table1.resize(grid, 0);
		table2.resize(grid, 0);

#pragma omp parallel for
		for (int k = 0; k < M1.m*M1.n; k++)
		{
			for (int i = 0; i < grid; i++)
			{
				double c = 0;
				if (i == grid - 1) c = 0.000001;
				if (M1.v[k] >= min1 + dx * i && M1.v[k] < min1 + dx * (i + 1))
				{
#pragma omp critical
					{
						table1[i] += 1;
					}
				}
			}

			for (int i = 0; i < grid; i++)
			{
				double c = 0;
				if (i == grid - 1) c = 0.000001;
				if (M2.v[k] >= min2 + dy * i && M2.v[k] < min2 + dy * (i + 1)+c)
				{
#pragma omp critical
					{
						table2[i] += 1;
					}
				}
			}
			for (int i = 0; i < grid; i++)
			{
				for (int j = 0; j < grid; j++)
				{
					double c = 0;
					if (i == grid - 1) c = 0.000001;
					double d = 0;
					if (j == grid - 1) d = 0.000001;

					if (M1.v[k] >= min1 + dx * j && M1.v[k] < min1 + dx * (j + 1)+d &&
						M2.v[k] >= min2 + dy * i && M2.v[k] < min2 + dy * (i + 1)+c)
					{
#pragma omp critical
						{
							table12(i, j) += 1;
						}
					}
				}
			}
		}

		//for (int i = 0; i < table1.size(); i++)
		//{
		//	printf("%d, ", (int)table1[i]);
		//}
		//printf("\n");
		//printf("dx:%f\n", dx);

		probability1.resize(grid,0);
		probability2.resize(grid,0);
		probability12 = Matrix<dnn_double>().zeros(grid, grid);
		probability1_2 = Matrix<dnn_double>().zeros(grid, grid);

		double s1 = 0;
		double s2 = 0;
		for (int i = 0; i < grid; i++)
		{
			probability1[i] = table1[i];
			probability2[i] = table2[i];
			s1 += probability1[i];
			s2 += probability2[i];
		}
		double s12 = 0;
		for (int i = 0; i < grid; i++)
		{
			for (int j = 0; j < grid; j++)
			{
				probability12(i, j) = table12(i, j);
				s12 += probability12(i, j);
			}
		}

		if (s1 == 0.0) s1 = 1;
		if (s2 == 0.0) s2 = 1;
		if (s12 == 0.0) s12 = 1;
		for (int i = 0; i < grid; i++)
		{
			probability1[i] /= s1;
			probability2[i] /= s2;
		}
		probability12 = probability12 / s12;

		s12 = 0;
		for (int i = 0; i < grid; i++)
		{
			for (int j = 0; j < grid; j++)
			{
				probability1_2(i, j) = probability1[i]* probability2[j];
				s12 += probability1_2(i, j);
			}
		}
		//probability1_2 = probability1_2 / s12;

		//dump
		//Matrix<dnn_double> tmp1(probability1);
		//Matrix<dnn_double> tmp2(probability2);

		//tmp1.print_csv("p1.csv");
		//tmp2.print_csv("p2.csv");
		//probability12.print_csv("p12.csv");		//p(x,y)
		//probability1_2.print_csv("p1_2.csv");	//p(x)*p(y)
	}

public:
	int grid;
	std::vector<dnn_double> probability1;	//p(x)
	std::vector<dnn_double> probability2;	//p(y)
	Matrix<dnn_double> probability12;		//p(x,y)
	Matrix<dnn_double> probability1_2;		//p(x)*p(y)


	MutualInformation(Matrix<dnn_double>& Col1, Matrix<dnn_double>& Col2, int grid_ = 30)
	{
		grid = grid_;

		gridtabel(Col1, Col2);
	}

	double Information()
	{
		double I = 0.0;

		Matrix<dnn_double> zz = Matrix<dnn_double>().zeros(grid, grid);

#pragma omp parallel for
		for (int i = 0; i < grid; i++)
		{
			for (int j = 0; j < grid; j++)
			{
				if (probability1_2(i,j) < 1.0e-32 || probability12(i, j) < 1.0e-32)
				{
					//continue;
				}
				else
				{
					double z = probability12(i, j) / probability1_2(i,j);
					if (z > 0)
					{
						zz(i, j) = probability12(i, j)*log(z);
					}
				}
			}
		}
		I = zz.Sum();
		return I;
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
		if (logmsg)
		{
			printf("\nAlgorithmB start\n");
			Bhat.print("start");
		}

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
		if (logmsg)
		{
			for (int x = 0; x < p.size(); x++)
				std::cout << x << "," << p[x] << "\t";
			printf("\n");
			fflush(stdout);

			printf("AlgorithmB end\n");
		}
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
		//絶対値が小さい順にソート
		std::sort(tmp.begin(), tmp.end());
		//b_est_tmp 成分の絶対値が小さい順に n(n+1)/2 個を 0 と置き換える
		int nn = 0;
		for (int i = 0; i < tmp.size(); i++)
		{
			if (nn >= N) break;
			if (tmp[i].zero_changed) continue;
			if (logmsg)
			{
				printf("[%d]%f ", i, tmp[i].dat);
			}
			tmp[i].dat = 0.0;
			tmp[i].abs_dat = 0.0;
			tmp[i].zero_changed = true;
			nn++;
		}

		if (logmsg)
		{
			printf("\nAlgorithmC start\n");
		}
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
				if (logmsg)
				{
					printf("[%d]%f->0.0\n", i, tmp[i].dat);
				}
				tmp[i].dat = 0.0;
				tmp[i].abs_dat = 0.0;
				tmp[i].zero_changed = true;
				break;
			}
			//b_est_tmp 成分に戻す
			for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
			{
				b_est_tmp.v[tmp[i].id] = tmp[i].dat;
			}
		}
		if (logmsg)
		{
			for (int x = 0; x < replacement.size(); x++)
				std::cout << x << "," << replacement[x] << "\t";
			printf("\nreplacement.size()=%d\n", (int)replacement.size());
		}
		std::vector<int> &r = replacement;

		//b_opt = b_opt[r, :]
		Matrix<dnn_double> b_opt = (Substitution(r)*b_est);
		//b_opt = b_opt[:, r]
		b_opt = (b_opt)*Substitution(r).transpose();
		if (logmsg)
		{
			b_opt.print_e();
		}
		//b_csl = np.tril(b_opt, -1)
		Matrix<dnn_double> b_csl = b_opt;
		for (int i = 0; i < b_csl.m; i++)
		{
			for (int j = i; j < b_csl.n; j++)
			{
				b_csl(i, j) = 0.0;
			}
		}
		if (logmsg)
		{
			b_csl.print_e();
		}

		if (0)
		{
			//並べ替えを元に戻す
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
		if (logmsg)
		{
			printf("AlgorithmC end\n");
			fflush(stdout);
		}
		return b_csl;
	}

	std::mt19937 mt;

public:

	bool logmsg = true;
	int confounding_factors_sampling = 1000;
	double confounding_factors_upper = 0.90;

	vector<int> replacement;
	Matrix<dnn_double> B;
	Matrix<dnn_double> B_pre_sort;
	Matrix<dnn_double> input;
	Matrix<dnn_double> modification_input;
	Matrix<dnn_double> mutual_information;
	Matrix<dnn_double> Mu;
	Matrix<dnn_double> residual_error;
	Matrix<dnn_double> residual_error_info;
	bool mutual_information_values = false;
	int confounding_factors = 0;

	Lingam() {
		error = -999;
	}

	void set(int variableNum_, std::mt19937 mt_)
	{
		mt = mt_;
		variableNum = variableNum_;
	}

	void save(std::string& filename)
	{
		FILE* lock = fopen("lingam.lock", "w");
		if (lock == NULL)
		{
			printf("lingam_lock:fatal error.\n");
			return;
		}

		FILE* fp = fopen((filename + ".replacement").c_str(), "w");
		if (fp)
		{
			for (int i = 0; i < replacement.size(); i++)
			{
				fprintf(fp, "%d\n", replacement[i]);
			}
			fclose(fp);
		}
		fp = fopen((filename + ".option").c_str(), "w");
		if (fp)
		{
			fprintf(fp, "confounding_factors:%d\n", confounding_factors);
			fclose(fp);
		}

		try
		{
			B.print_csv((char*)(filename + ".B.csv").c_str());
			B_pre_sort.print_csv((char*)(filename + ".B_pre_sort.csv").c_str());
			input.print_csv((char*)(filename + ".input.csv").c_str());
			modification_input.print_csv((char*)(filename + ".modification_input.csv").c_str());
			mutual_information.print_csv((char*)(filename + ".mutual_information.csv").c_str());
			Mu.print_csv((char*)(filename + ".mu.csv").c_str());

			residual_error_info.print_csv((char*)(filename + ".residual_error_info.csv").c_str());
			residual_error.print_csv((char*)(filename + ".residual_error.csv").c_str());
		}
		catch (std::exception& e)
		{
			printf("LiNGAM save exception:%s\n", e.what());
		}
		catch (...)
		{
			printf("LiNGAM save exception\n");
		}
		fclose(lock);
	}

	bool load(std::string& filename)
	{
		char buf[256];
		FILE* fp = fopen((filename + ".replacement").c_str(), "r");
		
		if (fp == NULL)
		{
			return false;
		}
		if (fp)
		{
			replacement.clear();
			while (fgets(buf, 256, fp) != NULL)
			{
				int id = 0;
				sscanf(buf, "%d", &id);
				replacement.push_back(id);
			}
			fclose(fp);
		}
		fp = fopen((filename + ".option").c_str(), "r");
		if (fp)
		{
			while (fgets(buf, 256, fp) != NULL)
			{
				if (strstr(buf, "confounding_factors:"))
				{
					sscanf(buf, "confounding_factors:%d\n", &confounding_factors);
				}
			}
			fclose(fp);
		}

		try
		{
			CSVReader csv1((filename + ".B.csv"), ',', false);
			B = csv1.toMat();
			printf("laod B\n"); fflush(stdout);

			CSVReader csv2((filename + ".B_pre_sort.csv"), ',', false);
			B_pre_sort = csv2.toMat();
			printf("load B_pre_sort\n"); fflush(stdout);

			CSVReader csv3((filename + ".input.csv"), ',', false);
			input = csv3.toMat();
			printf("load input\n"); fflush(stdout);

			CSVReader csv4((filename + ".modification_input.csv"), ',', false);
			modification_input = csv4.toMat();
			printf("load modification_input\n"); fflush(stdout);

			CSVReader csv5((filename + ".mutual_information.csv"), ',', false);
			mutual_information = csv5.toMat();
			printf("load mutual_information\n"); fflush(stdout);

			CSVReader csv6((filename + ".mu.csv"), ',', false);
			Mu = csv6.toMat();
			printf("load μ\n"); fflush(stdout);

			CSVReader csv7((filename + ".residual_error_info.csv"), ',', false);
			residual_error_info = csv7.toMat();
			printf("residual_error_info\n"); fflush(stdout);

			CSVReader csv8((filename + ".residual_error.csv"), ',', false);
			residual_error = csv8.toMat();
			printf("residual_error\n"); fflush(stdout);
		}
		catch (std::exception& e)
		{
			printf("LiNGAM load exception:%s\n", e.what());
		}
		catch (...)
		{
			printf("LiNGAM load exception\n");
		}
		return true;
	}


	int remove_redundancy(const dnn_double alpha = 0.01, const size_t max_ica_iteration = 1000000, const dnn_double tolerance = TOLERANCE)
	{
		printf("remove_redundancy:lasso start\n"); fflush(stdout);
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
				printf("n_iter=%d error_eps=%f\n", lasso.num_iteration, lasso.error_eps);
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
			if (lasso.getStatus() == -2)
			{
				error = -1;
				//break;
			}
			else
			{
				const Matrix<dnn_double>& c = lasso.coef;
				for (int k = 0; k < i; k++)
				{
					B(i, k) = c.v[k] / lasso.sigma(0, k);
				}
			}
#endif
			if (i == B.m-1) break;
			//c.print();
			X = X.appendCol(Y);
			Y = xs.Col(replacement[i + 1]);
		}
		B.print_e("remove_redundancy");
		printf("remove_redundancy:lasso end\n\n"); fflush(stdout);
		return error;
	}

	std::vector<std::string> linear_regression_var;

	std::string line_colors[10] =
	{
		"#000000",
		"#000000",
		"#000000",
		"#000000",
		"#696969",
		"#808080",
		"#a9a9a9",
		"#c0c0c0",
		"#d3d3d3",
		"#dcdcdc"
	};

	void diagram(const std::vector<std::string>& column_names, std::vector<std::string> y_var, std::vector<int>& residual_flag, const char* filename, bool sideways = false, int size=30, char* outformat="png", bool background_Transparent=false, double mutual_information_cut = 0)
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

		auto& mutual_information_tmp = mutual_information / mutual_information.Max();

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
						if (mutual_information_values)
						{
							utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\" color=red penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), style);
						}
						else
						{
							utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=red penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
						}
					}
					else
						if (in_line)
						{
							linear_regression_var.push_back(item[j]);
							if (mutual_information_values)
							{
								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), style);
							}
							else
							{
								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
							}
						}
						else
						{
							if (mutual_information_values)
							{

								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\" color=\"%s\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), line_colors[9 - (int)(9 * mutual_information_tmp(i, j))], style);
							}
							else
							{
								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=black %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i,j), style);
							}
						}
				}
			}
		}
		for (int i = 0; i < y_var.size(); i++)
		{
			//項目2[fillcolor="#ccddff", style="filled"];
			std::string item1 = y_var[i];
			if (item1.c_str()[0] != '\"')
			{
				item1 = "\"" + item1 + "\"";
			}
			utf8.fprintf(fp, "%s [fillcolor=\"#ccddff\", style=\"filled\"]\n", item1.c_str());
		}
		//for (int i = 0; i < B_tmp.n; i++)
		//{
		//	for (int j = 0; j < B_tmp.n; j++)
		//	{
		//		if (mutual_information(i, j) < mutual_information_cut)
		//		{
		//			std::string item1 = item[i];
		//			if (item1.c_str()[0] != '\"')
		//			{
		//				item1 = "\"" + item1 + "\"";
		//			}
		//			utf8.fprintf(fp, "%s [fillcolor=\"#ccddff\", style=\"filled\"]\n", item1.c_str());
		//		}
		//	}
		//}


		utf8.fprintf(fp, "}\n");
		fclose(fp);
#ifdef USE_GRAPHVIZ_DOT
		char cmd[512];
		graphviz_path_::getGraphvizPath();
		std::string path = graphviz_path_::path_;
		if (path != "")
		{
			if (path.c_str()[0] == '\"')
			{
				char tmp[1024];
				char tmp2[1024];
				strcpy(tmp, path.c_str());
				strcpy(tmp2, tmp+1);
				if (tmp2[strlen(tmp2) - 1] == '\"')
				{
					tmp2[strlen(tmp2) - 1] = '\0';
				}
				path = tmp2;
			}
			sprintf(cmd, "\"%s\\dot.exe\" -T%s %s -o Digraph.%s", path.c_str(), outformat, filename, outformat);
		}
		else
		{
			sprintf(cmd, "dot.exe -T%s %s -o Digraph.%s", outformat, filename, outformat);
		}

		if (B.n < 20)
		{
			system(cmd);
			printf("%s\n", cmd);
		}
		else
		{
			FILE* fp = fopen("Digraph.bat", "r");
			if (fp)
			{
				fclose(fp);
				remove("Digraph.bat");
			}
			if ((fp = fopen("Digraph.bat", "w")) != NULL)
			{
				fprintf(fp, "%s\n", cmd);
				fclose(fp);
			}
		}
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

		//並べ替えを元に戻す
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

	void before_sorting(Matrix<dnn_double>& b)
	{
		std::vector<int> r = replacement;
		Matrix<dnn_double> b_csl = b;

		//並べ替えを元に戻す
		//b_csl[r, :] = deepcopy(b_csl)
		b_csl = (Substitution(r).transpose()*b_csl);
		//b_csl.print_e();

		//b_csl[:, r] = deepcopy(b_csl)
		b_csl = (b_csl*Substitution(r));
		//b_csl.print_e();
		b = b_csl;
	}

	Matrix<dnn_double> before_sorting_(Matrix<dnn_double>& b)
	{
		std::vector<int> r = replacement;
		Matrix<dnn_double> b_csl = b;

		//並べ替えを元に戻す
		//b_csl[r, :] = deepcopy(b_csl)
		b_csl = (Substitution(r).transpose()*b_csl);
		//b_csl.print_e();

		//b_csl[:, r] = deepcopy(b_csl)
		b_csl = (b_csl*Substitution(r));
		//b_csl.print_e();
		return b_csl;
	}

	int fit(Matrix<dnn_double>& X, const int max_ica_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
		remove("confounding_factors.txt");
		error = 0;

		residual_error = Matrix<dnn_double>(X.m, X.n);
		residual_error_info = Matrix<dnn_double>(X.n, X.n);

		Mu = Matrix<dnn_double>().zeros(X.m, X.n);
		input = X;
		modification_input = X;

		Matrix<dnn_double> xs = X;

		ICA ica;
		ica.set(variableNum);
		ica.fit(xs, max_ica_iteration, tolerance);
		(ica.A.transpose()).inv().print_e();
		error = ica.getStatus();
		if (error != 0)
		{
			printf("ERROR:ICA\n");
			return error;
		}

		Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv();
		Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal();

		HungarianAlgorithm HungAlgo;
		vector<int> replace;

		double cost = HungAlgo.Solve(W_ica_, replace);

		for (int x = 0; x < W_ica_.m; x++)
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
		std::iota(v.begin(), v.end(), 0);       // v に 0, 1, 2, ... N-1 を設定
		do {
			std::vector<int> replacement_case;

			for (auto x : v) replacement_case.push_back(x);
			replacement_list.push_back(replacement_case);
			//for (auto x : v) cout << x << " "; cout << "\n";    // v の要素を表示
		} while (next_permutation(v.begin(), v.end()));     // 次の順列を生成

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
		//絶対値が小さい順にソート
		std::sort(tmp.begin(), tmp.end());

		int N_ = N;
		bool tri_ng = false;
		do
		{
			//b_est_tmp 成分の絶対値が小さい順に n(n+1)/2 個を 0 と置き換える
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
			N_ = 1;	//次は次に絶対値が小さい成分を 0 と置いて再び確認

					//b_est_tmp 成分に戻す
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
			//行を並び替えて下三角行列にできるか調べる。
			for (int k = 0; k < replacement_list.size(); k++)
			{
				//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v の要素を表示

				Matrix<dnn_double> tmp = Substitution(replacement_list[k])*b_est_tmp;
				//下半行列かチェック
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
					//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v の要素を表示
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

		if (error == 0) B = b_est;

		calc_mutual_information(X, mutual_information);
		{
			auto& b = before_sorting_(B);
			for (int j = 0; j < xs.m; j++)
			{
				Matrix<dnn_double> x(xs.n, 1);
				Matrix<dnn_double> y(xs.n, 1);
				for (int i = 0; i < xs.n; i++)
				{
					x(i, 0) = xs(j, i);
					y(i, 0) = X(j, i);
				}
				Matrix<dnn_double>& rr = y - b * x;
				for (int i = 0; i < xs.n; i++)
				{
					residual_error(j, i) = rr(0, i);
				}
			}
			//Evaluation of independence
			calc_mutual_information( residual_error, residual_error_info);
		}
		
		double c_factors = residual_error_info.Max();
		residual_error_info.print_csv("confounding_factors_info.csv");
		if (c_factors > confounding_factors_upper)
		{
			FILE* fp = fopen("confounding_factors.txt", "w");
			if (fp)
			{
				fprintf(fp, "Confounding factors may be present\n");
				fclose(fp);
			}
		}

		B_pre_sort = this->before_sorting_(B);

		if (error)
		{
			printf("WARNING:No valid path was found.\n");
		}

		return error;
	}

	void calc_mutual_information( Matrix<dnn_double>& X, Matrix<dnn_double>& info)
	{
		info = Matrix<dnn_double>().zeros(X.n, X.n);
#pragma omp parallel for
		for (int j = 0; j < X.n; j++)
		{
			for (int i = 0; i < X.n; i++)
			{
				info(i, j) = 0;
				if (i == j)
				{
					continue;
				}

				Matrix<dnn_double>& x = X.Col(replacement[i]);
				Matrix<dnn_double>& y = X.Col(replacement[j]);

				MutualInformation I(x, y);
				double tmp = I.Information();
				info(i, j) = tmp;
			}
		}
	}

	int early_stopping = 0;
	double prior_knowledge_rate = 1.0;
	std::vector<int> prior_knowledge;
	double distribution_rate = 1.0;
	double temperature_alp = 0.95;
	double rho = 3.0;
	//double mu_max_value = 10.0;
	int fit2(Matrix<dnn_double>& X, const int max_ica_iteration= MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
		printf("distribution_rate:%f\n", distribution_rate);
		printf("rho:%f\n", rho);

		remove("confounding_factors.txt");

		CONSOLE_SCREEN_BUFFER_INFO csbi;
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hStdout, &csbi);

		std::chrono::system_clock::time_point  start, end;
		double elapsed;
		double elapsed_ave = 0.0;

		logmsg = false;
		error = 0;

		Mu = Matrix<dnn_double>().zeros(X.m, X.n);
		Matrix<dnn_double> B_best_sv;
		auto replacement_best = replacement;
		residual_error = Matrix<dnn_double>(X.m, X.n);
		residual_error_info = Matrix<dnn_double>(X.n, X.n);
		
		auto residual_error_best = residual_error;
		auto residual_error_info_best = residual_error_info;

		input = X;
		double min_value = Abs(X).Min();
		if (min_value < 1.0e-3) min_value = 1.0e-3;

		//std::random_device seed_gen;
		//std::default_random_engine engine(seed_gen());
		std::default_random_engine engine(123456789);
		//std::student_t_distribution<> dist(12.0);
		std::uniform_real_distribution<> acceptance_rate(0.0, 1.0);
		//std::uniform_real_distribution<> rate_cf(-1.0/ mu_max_value, 1.0/ mu_max_value);
		std::uniform_real_distribution<> knowledge_rate(0.0, 1.0);
		std::uniform_real_distribution<> noise(-min_value*0.1, min_value*0.1);

		std::uniform_real_distribution<> condition(0.0, 1.0);

		int no_accept_count = 0;
		int update_count = 0;
		int reject = 0;
		int accept = 0;
		double temperature = 0.0;

		double best_residual = 999999999.0;
		double best_independ = 999999999.0;
		double best_min_value = 999999999.0;
		Matrix<dnn_double>μ(X.m, X.n);
		std::uniform_int_distribution<> var_select(0, X.n);

		double abs_residual_errormax = -1;

		//μ = μ.zeros(μ.m, μ.n);
		μ = μ.Rand()*rho;
		//std::student_t_distribution<> dist(6.0);
		//std::normal_distribution<> dist(0.0, 20.0);
		//std::uniform_real_distribution<> dist(-mu_max_value, mu_max_value);

		double t_p = 12.0*rho;
		if (t_p <= 6.0) t_p = 7.0;
		std::uniform_real_distribution<> student_t_dist(6.0, t_p);
		std::vector<int> dist_t_param(X.n);
		for (int i = 0; i < X.n; i++)
		{
			dist_t_param[i] = student_t_dist(engine);
		}

		const char* loss = "lingam_loss.dat";
		try
		{
			std::ofstream ofs(loss, std::ios::out);
			ofs.close();
		}
		catch (...)
		{
		}

		for (int kk = 0; kk < confounding_factors_sampling; kk++)
		{
			start = std::chrono::system_clock::now();

			//parameter_search mode!!
			if (early_stopping < 0 )
			{
				if (update_count > 2 || kk > abs(early_stopping))
				{
					break;
				}
			}

			no_accept_count++;
			if (early_stopping > 0 && no_accept_count > early_stopping)
			{
				printf("early_stopping!\n");
				error = 1;
				break;
			}
			Matrix<dnn_double> xs = X;


			//if (kk > 0)
			{
				int var = var_select(engine);
				//printf("var:%d\n", var);

				if (accept)
				{
					for (int j = 0; j < X.m; j++)
					{
						double c = 1;
						if (accept < 32)
						{
							c = 1.0 / pow(2.0, accept);
						}
						else
						{
							c = 1.0e-10;
						}
						μ(j, var) += noise(engine) * c;
					}
				}
				else
				{
					dist_t_param[var] = student_t_dist(engine);

					auto& dist = std::student_t_distribution<>(dist_t_param[var]);

					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ7
					const double rate = distribution_rate * dist(engine);
					for (int j = 0; j < X.m; j++)
					{
						μ(j, var) = rate + noise(engine);
					}
				}

#pragma omp parallel for
				for (int j = 0; j < X.m; j++)
				{
					for (int i = 0; i < X.n; i++)
					{
						xs(j, i) -= μ(j, i);
					}
				}
			}

			ICA ica;
			ica.logmsg = false;
			ica.set(variableNum);
			try
			{
				ica.fit(xs, max_ica_iteration, tolerance);
				//(ica.A.transpose()).inv().print_e();
				error = ica.getStatus();
				if (error != 0)
				{
					continue;
				}
			}
			catch (...)
			{
				//error = -99;
				continue;
			}

			try
			{
				Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv();
				Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal();

				HungarianAlgorithm HungAlgo;
				vector<int> replace;

				double cost = HungAlgo.Solve(W_ica_, replace);

				//for (int x = 0; x < W_ica_.m; x++)
				//	std::cout << x << "," << replace[x] << "\t";
				//printf("\n");

				Matrix<dnn_double>& ixs = toMatrix(replace);
				//ixs.print();
				//Substitution(replace).print("Substitution matrix");

				//P^-1*Wica
				Matrix<dnn_double>& W_ica_perm = (Substitution(replace).inv()*W_ica);
				//W_ica_perm.print_e("Replacement->W_ica_perm");

				//D^-1
				Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
				Matrix<dnn_double> D2(diag_vector(D));
				//(D2.Reciprocal()).print_e("1/D");

				//W_ica_perm_D=I - D^-1*(P^-1*Wica)
				Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal()));

				//W_ica_perm_D.print_e("W_ica_perm_D");

				//B=I - D^-1*(P^-1*Wica)
				Matrix<dnn_double>& b_est = Matrix<dnn_double>().unit(W_ica_perm_D.m, W_ica_perm_D.n) - W_ica_perm_D;
				//b_est.print_e("b_est");

				//https://www.cs.helsinki.fi/u/ahyvarin/papers/JMLR06.pdf
				const int n = W_ica_perm_D.m;
				Matrix<dnn_double> b_est_tmp = b_est;
				b_est = AlgorithmC(b_est_tmp, n);

				//for (int x = 0; x < replacement.size(); x++)
				//	std::cout << x << "," << replacement[x] << "\t";
				//printf("\n");
				//b_est.print_e();
				//fflush(stdout);

				if (error == 0)
				{
					B = b_est;
				}
				else
				{
					printf("--------------------\n");
					continue;
				}
			}
			catch (std::exception& e)
			{
				printf("Exception:%s\n", e.what());
				continue;
			}
			catch (...)
			{
				printf("Exception\n");
				continue;
			}

			//float r = 0.0;
			{
				auto& b = before_sorting_(B);
				//b.print("b");
#pragma omp parallel for
				for (int j = 0; j < xs.m; j++)
				{
					Matrix<dnn_double> μT(xs.n, 1);
					Matrix<dnn_double> x(xs.n, 1);
					Matrix<dnn_double> y(xs.n, 1);
					for (int i = 0; i < xs.n; i++)
					{
						//x(i, 0) = xs(j, replacement[i]);
						//y(i, 0) = X(j, replacement[i]);
						x(i, 0) = xs(j, i);
						y(i, 0) = X(j, i);
						μT(i, 0) = μ(j, i);
					}
					Matrix<dnn_double>& rr = y - b * y - μT;
					for (int i = 0; i < xs.n; i++)
					{
						//r += rr(0, i)*rr(0, i);
						residual_error(j, i) = rr(0, i);
					}
				}
				//r /= (xs.m*xs.n);

				//Evaluation of independence
				calc_mutual_information( residual_error, residual_error_info);
			}
			
			double  abs_residual_errormax_cur = Abs(residual_error).Max();
			if (abs_residual_errormax < 0)
			{
				abs_residual_errormax = abs_residual_errormax_cur;
			}
			if (abs_residual_errormax < 1.0e-10)
			{
				abs_residual_errormax = 1.0e-10;
			}

			//residual_error.print("residual_error");
			//printf("residual_error max:%f\n", Abs(residual_error).Max());

			double info = residual_error_info.Max();
			double residual = abs_residual_errormax_cur / abs_residual_errormax;

			double value = std::max(info, residual);


			//printf("%d info:%f residual:%f -> value:%f (best_min_value:%f)\n", kk, info, residual, value, best_min_value);

			bool accept_ = false;

			if (best_min_value > value)
			{
				accept_ = true;
			}
			if (kk == 0) accept_ = true;

			//if (!accept_ && best_residual == residual)
			//{
			//	if (best_min_value > info )
			//	{
			//		accept_ = true;
			//	}
			//}


			double rt = (double)0.1*kk / (double)confounding_factors_sampling;
			temperature = pow(temperature_alp, rt);

			if (!accept_)
			{
				double alp = acceptance_rate(engine);

				double th = -1.0;
				if (best_min_value < value)
				{
					if (value - best_min_value > 1.5)
					{
						th = exp(-1.5 / temperature);
					}
					else
					{
						th = exp((best_min_value - value) / temperature);
					}
				}
				//printf("temperature:%f  alp:%f th:%f\n", temperature, alp, th );
				if (alp < th)
				{
					accept_ = true;
				}
			}

			if (kk > 0 && accept_ && prior_knowledge.size())
			{
				auto& b = before_sorting_(B);

				for (int k = 0; k < prior_knowledge.size() / 2; k++)
				{
					bool ng_edge = false;
					int ii, jj;
					if (prior_knowledge[2 * k] < 0)
					{
						ii = abs(prior_knowledge[2 * k]) - 1;
						jj = abs(prior_knowledge[2 * k + 1]) - 1;
						ng_edge = true;
					}
					else
					{
						ii = prior_knowledge[2 * k] - 1;
						jj = prior_knowledge[2 * k + 1] - 1;
					}
					if (ng_edge)
					{
						if (fabs(b(ii, jj)) > 0.001)
						{
							accept_ = false;
							break;
						}
						//printf("%d -> %d NG\n", jj, ii);
					}
					else
					{
						if (fabs(b(ii, jj)) > 0.001 && fabs(b(jj, ii)) < 0.001)
						{
							accept_ = false;
							break;
							//printf("%d -> %d NG\n", jj, ii);
						}
						if (fabs(b(ii, jj)) > 0.001 || fabs(b(jj, ii)) < 0.001)
						{
							accept_ = false;
							break;
							//printf("%d -> %d NG\n", jj, ii);
						}
					}
					if (!accept_)
					{
						if (prior_knowledge_rate < knowledge_rate(engine))
						{
							accept_ = true;
						}
					}
				}
				// ※ b( i, j ) = j -> i
				////b.print("b");
				////printf("b(0,1):%f b(0,2):%f\n", b(0, 1), b(0, 2));
				//// # -> 0 NG
				//if (fabs(b(0, 1)) > 0.001 || fabs(b(0, 2)) > 0.001 )
				//{
				//	accept_ = false;
				//}
				////2 -> 1 NG   1 -> 2 OK
				//if (fabs(b(2, 1)) < fabs(b(1, 2)))
				//{
				//	accept_ = false;
				//}

				//if (!accept_ && a)
				//{
				//	b.print("NG-b");
				//}
			}

			bool best_update = false;
			if (accept_)
			{
				//printf("+\n");
				if (best_min_value >= value)
				{
					best_min_value = value;
					best_residual = residual;
					best_independ = info;

					best_update = true;
				}
				//{
				//	auto& b = before_sorting_(B);
				//	b.print("accept-b");
				//}
				accept++;
				reject = 0;
				if (best_update)
				{		
					no_accept_count = 0;
					update_count++;
					char buf[256];
					sprintf(buf, "@[%d/%d] %f (ind:%f,err:%f)accept:%d", kk, confounding_factors_sampling - 1, best_min_value, info, residual, accept);

					if (1)
					{
						DWORD  bytesWritten = 0;
						SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY | BACKGROUND_BLUE | BACKGROUND_INTENSITY);
						WriteFile(hStdout, buf, strlen(buf), &bytesWritten, NULL);
						SetConsoleTextAttribute(hStdout, csbi.wAttributes);
						printf("\n");
					}
					else
					{
						printf(buf);
					}
					Mu = μ;
					B_best_sv = B;
					B_pre_sort = B;
					modification_input = xs;
					replacement_best = replacement;
					residual_error_best = residual_error;
					residual_error_info_best = residual_error_info;


					calc_mutual_information(X, mutual_information);
					save(std::string("lingam.model"));
					if (info + residual < 0.000001)
					{
						printf("convergence!\n");
						break;
					}
				}
				fflush(stdout);
			}
			else
			{
				accept = 0;
				reject++;
			}
			try
			{
				std::ofstream ofs(loss, std::ios::app);
				ofs << best_residual << "," << best_independ << std::endl;
				ofs.close();
			}
			catch (...)
			{
			}

			//if ( confounding_factors_sampling >= 4000 && reject > confounding_factors_sampling / 4)
			//{
			//	break;
			//}

			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//printf("[%d/%d]%8.3f[msec]\n", kk, confounding_factors_sampling - 1, elapsed);
			
			elapsed_ave += elapsed;
			double t1 = (confounding_factors_sampling - 1 - kk)*(elapsed_ave*0.001 / (kk + 1));
			if (kk != 0 && elapsed_ave*0.001 / (kk + 1) < 1 && kk % 20)
			{
				continue;
			}

			if (t1 < 60)
			{
				printf("[%d/%d]Total elapsed time:%.3f[sec] Estimated end time:%.3f[sec] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave*0.001, t1, accept);
				fflush(stdout);
				continue;
			}
			t1 /= 60.0;
			if (t1 < 60)
			{
				printf("[%d/%d]Total elapsed time:%.3f[min] Estimated end time:%.3f[min] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave*0.001 / 60.0, t1, reject);
				fflush(stdout);
				continue;
			}
			t1 /= 60.0;
			if (t1 < 24)
			{
				printf("[%d/%d]Total elapsed time:%.3f[h] Estimated end time:%.3f[h] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave*0.001 / 60.0 / 60.0, t1, reject);
				fflush(stdout);
				continue;
			}
			t1 /= 365;
			if (t1 < 365)
			{
				printf("[%d/%d]Total elapsed time:%.3f[days] Estimated end time:%.3f[days] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave*0.001 /60 / 60 / 24.0 / 365.0, t1, reject);
				fflush(stdout);
				continue;
			}
			printf("[%d/%d]Total elapsed time:%.3f[years] Estimated end time:%.3f[years] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave*0.001 /60 / 60 / 24.0 / 365.0, t1, reject);
			fflush(stdout);
		}

		residual_error = residual_error_best;
		residual_error_info = residual_error_info_best;
		replacement = replacement_best;
		B = B_best_sv;

		calc_mutual_information(X, mutual_information);

		double c_factors = residual_error_info.Max();
		residual_error_info.print_csv("confounding_factors_info.csv");
		if (c_factors > confounding_factors_upper)
		{
			FILE* fp = fopen("confounding_factors.txt", "w");
			if (fp)
			{
				fprintf(fp, "Confounding factors may be present\n");
				fclose(fp);
			}
		}
		B_pre_sort = this->before_sorting_(B);
		if (update_count < 2)
		{
			error = 1;
			printf("WARNING:No valid path was found.\n");
		}
		return error;
	}
};
#endif
