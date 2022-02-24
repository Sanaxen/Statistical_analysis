#ifndef __LINGAM_H__
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define NOMINMAX
#define __LINGAM_H__
#include <chrono>
#include <signal.h>

#include "../../include/Matrix.hpp"
#include "../../include/statistical/fastICA.h"
#include "../../include/hungarian-algorithm/Hungarian.h"
#include "../../include/statistical/LinearRegression.h"
#include "../../include/statistical/RegularizationRegression.h"
#include "../../include/util/utf8_printf.hpp"
#include "../../include/util/csvreader.h"
#include "../../include/util/file_util.hpp"

#include "../../include/util/swilk.h"
#include "../../include/util/Generalized_Gaussian.hpp"
#include "../../include/statistical/Independence.h"

//#define USE_EIGEN

#ifdef USE_EIGEN
#include "../../include/statistical/RegularizationRegression_eigen_version.h"
#endif

extern int use_libtorch;
namespace tiny_dnn
{
	typedef std::vector<float_t> vec_t;
	typedef std::vector<vec_t> tensor_t;
};
inline void MatrixToTensor(Matrix<dnn_double>& X, tiny_dnn::tensor_t& T, int read_max = -1)
{
	size_t rd_max = read_max < 0 ? X.m : min(read_max, X.m);
	for (int i = 0; i < rd_max; i++)
	{
		tiny_dnn::vec_t x;
		for (int j = 0; j < X.n; j++)
		{
			x.push_back(X(i, j));
		}
		T.push_back(x);
	}
}

inline void TensorToMatrix(tiny_dnn::tensor_t& T, Matrix<dnn_double>& X)
{
	X = Matrix<dnn_double>(T.size(), T[0].size());
	for (int i = 0; i < T.size(); i++)
	{
		for (int j = 0; j < T[i].size(); j++)
		{
			X(i, j) = T[i][j];
		}
	}
}

inline void normalizeZ(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
{
	mean = std::vector<float_t>(X[0].size(), 0.0);
	sigma = std::vector<float_t>(X[0].size(), 0.0);

	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			mean[k] += X[i][k];
		}
	}
	for (int k = 0; k < X[0].size(); k++)
	{
		mean[k] /= X.size();
	}
	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			sigma[k] += (X[i][k] - mean[k]) * (X[i][k] - mean[k]);
		}
	}
	for (int k = 0; k < X[0].size(); k++)
	{
		sigma[k] /= (X.size() - 1);
		sigma[k] = sqrt(sigma[k]);
	}


	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			X[i][k] = (X[i][k] - mean[k]) / (sigma[k] + 1.0e-10);
		}
	}
}

#ifdef USE_LIBTORCH

#define _LIBRARY_EXPORTS __declspec(dllimport)

extern "C" _LIBRARY_EXPORTS int cuda_is_available();
extern "C" _LIBRARY_EXPORTS int torch_train_init(void);
extern "C" _LIBRARY_EXPORTS void torch_setDevice(const char* device_name);
extern "C" _LIBRARY_EXPORTS void torch_delete_model();
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x);
extern "C" _LIBRARY_EXPORTS float torch_get_Loss(int batch);

extern "C" _LIBRARY_EXPORTS void torch_params(
	int n_train_epochs_,
	int n_minibatch_,
	int input_size_,

	int n_layers_,
	int dropout_,
	int n_hidden_size_,
	int fc_hidden_size_,
	float learning_rate_,

	float clip_gradients_,
	int use_cnn_,
	int use_add_bn_,
	int use_cnn_add_bn_,
	int residual_,
	int padding_prm_,

	int classification_,
	char* weight_init_type_,
	char* activation_fnc_,
	int early_stopping_,
	char* opt_type_,
	bool batch_shuffle_,
	int test_mode_
);
extern "C" _LIBRARY_EXPORTS void torch_train_fc(
	std::vector<tiny_dnn::vec_t>&train_images_,
	std::vector<tiny_dnn::vec_t>&train_labels_,
	int n_minibatch,
	int n_train_epochs,
	char* regression,
	std::function <void(void)> on_enumerate_minibatch,
	std::function <void(void)> on_enumerate_epoch
);
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

/* シグナル受信/処理 */
inline void SigHandler_lingam(int p_signame)
{
	static int sig_catch = 0;
	if (sig_catch)
	{
		printf("割り込みです。終了します\n");
		fflush(stdout);

		FILE* fp = fopen("confounding_factors.txt", "w");
		if (fp)
		{
			fprintf(fp, "%f\n", -99.0);
			fclose(fp);
		}
		exit(0);
	}
	sig_catch++;
	return;
}

inline bool Shapiro_wilk_test(Matrix<dnn_double>& xs, std::vector<std::string>& header_names, std::vector<int>& shapiro_wilk_test, std::vector<std::string>& shapiro_wilk_values_input)
{
	bool error = false;
	
	shapiro_wilk_test.resize(xs.n, 0);
	{
		printf("shapiro_wilk test(0.05) start\n");
		shapiro_wilk shapiro;
		for (int i = 0; i < xs.n; i++)
		{

			Matrix<dnn_double> tmp = xs.Col(i);
			int stat = shapiro.test(tmp);
			if (stat == 0)
			{
				char buf[256];
				sprintf(buf, "w:%-4.4f p_value:%-.16g", shapiro.get_w(), shapiro.p_value());
				shapiro_wilk_values_input.push_back(buf);
				//printf("%s\n", buf);

				printf("[%-20.20s]w:%-8.3f p_value:%-10.16f\n", header_names[i].c_str(), shapiro.get_w(), shapiro.p_value());
				if (shapiro.p_value() > 0.05)
				{
					shapiro_wilk_test[i] = 1;
					error = true;
				}
			}
			else
			{
				printf("error shapiro.test=%d\n", stat);
			}
		}
		printf("shapiro_wilk test end\n\n");
	}
	return error;
}


class lingam_reg
{
public:
	void calc_r(Matrix<dnn_double>&x, Matrix<dnn_double>&y, Matrix<dnn_double>&xr, Matrix<dnn_double>&yr)
	{
		
		auto& x_mean = x.Mean();
		auto& y_mean = y.Mean();
		auto& xy_mean = x.hadamard(y).Mean();

		xr = (x - xy_mean.v[0] - x_mean.v[0]*y_mean.v[0]) / y.Var(y_mean).hadamard(y).v[0];
		yr = (y - xy_mean.v[0] - x_mean.v[0]*y_mean.v[0]) / x.Var(x_mean).hadamard(x).v[0];

		return;
	}


	bool dir(Matrix<dnn_double>&x, Matrix<dnn_double>&y, double& diff)
	{

		Matrix<dnn_double> xr;
		Matrix<dnn_double> yr;
		calc_r(x, y, xr, yr);

		auto& xr_mean = xr.Mean();
		auto& yr_mean = yr.Mean();
		double m = _entropy(_normalize(x)) + _entropy(_normalize(xr) / xr.Std(xr_mean))
			- _entropy(_normalize(y)) - _entropy(_normalize(yr) / yr.Std(yr_mean));
		
		diff = m;
		if (m >= 0)
		{
			return true;
		}
		return false;
	}
};

#define _SAMPLING 0
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
	Matrix<dnn_double> intercept;
	Matrix<dnn_double> residual_error;
	Matrix<dnn_double> residual_error_independ;
	bool mutual_information_values = false;
	int confounding_factors = 0;
	int bins = 30;
	bool use_intercept = false;

	Matrix<dnn_double> b_probability;
	bool use_bootstrap = false;
	int bootstrap_sample = 1000;
	bool use_hsic = false;
	bool use_gpu = false;

	Lingam() {
		error = -999;
	}

	void set(int variableNum_, std::mt19937 mt_)
	{
		mt = mt_;
		variableNum = variableNum_;
	}

	void save(std::string& filename, bool loss_dat = false)
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
			fprintf(fp, "use_bootstrap:%d\n", use_bootstrap?1:0);
			fclose(fp);
		}
		fp = fopen((filename + ".loss").c_str(), "w");
		if (fp)
		{
			fprintf(fp, "loss:%f\n", loss_value);
			fclose(fp);
		}
		copyfile("select_variables.dat", (filename + ".select_variables.dat").c_str());
		if (loss_dat)
		{
			copyfile("lingam_loss.dat", (filename + ".lingam_loss.dat").c_str());
		}
		try
		{
			B.print_csv((char*)(filename + ".B.csv").c_str());
			B_pre_sort.print_csv((char*)(filename + ".B_pre_sort.csv").c_str());
			input.print_csv((char*)(filename + ".input.csv").c_str());
			modification_input.print_csv((char*)(filename + ".modification_input.csv").c_str());
			mutual_information.print_csv((char*)(filename + ".mutual_information.csv").c_str());
			Mu.print_csv((char*)(filename + ".mu.csv").c_str());
			intercept.print_csv((char*)(filename + ".intercept.csv").c_str());

			residual_error_independ.print_csv((char*)(filename + ".residual_error_independ.csv").c_str());
			residual_error.print_csv((char*)(filename + ".residual_error.csv").c_str());

			if (use_bootstrap)
			{
				b_probability.print_csv((char*)(filename + ".b_probability.csv").c_str());
			}
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
		remove("lingam.lock");
	}

	bool load(std::string& filename, bool loss_data = false)
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
				if (strstr(buf, "use_bootstrap:"))
				{
					int dmy;
					sscanf(buf, "use_bootstrap:%d\n", &dmy);
					use_bootstrap = (dmy != 0) ? true : false;
				}
			}
			fclose(fp);
		}
		fp = fopen((filename + ".loss").c_str(), "r");
		if (fp)
		{
			while (fgets(buf, 256, fp) != NULL)
			{
				if (strstr(buf, "loss:"))
				{
					sscanf(buf, "loss:%lf\n", &loss_value);
				}
			}
			fclose(fp);
		}
		copyfile((filename + ".select_variables.dat").c_str(), "select_variables.dat");
		if (loss_data)
		{
			copyfile((filename + ".lingam_loss.dat").c_str(), "lingam_loss.dat");
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

			CSVReader csv7((filename + ".residual_error_independ.csv"), ',', false);
			residual_error_independ = csv7.toMat();
			printf("residual_error_independ\n"); fflush(stdout);

			CSVReader csv8((filename + ".residual_error.csv"), ',', false);
			residual_error = csv8.toMat();
			printf("residual_error\n"); fflush(stdout);

			CSVReader csv9((filename + ".intercept.csv"), ',', false);
			intercept = csv9.toMat();
			printf("intercept\n"); fflush(stdout);

			if (use_bootstrap)
			{
				CSVReader csv10((filename + ".b_probability.csv"), ',', false);
				b_probability = csv10.toMat();
				printf("b_probability\n"); fflush(stdout);
			}
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


	int remove_redundancy(const dnn_double alpha = 0.01, const size_t max_ica_iteration = 1000000, const dnn_double tolerance = TOLERANCE, int use_adaptive_lasso = 1)
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
			lasso.use_bias = false;
#endif
			if (use_adaptive_lasso)
			{
				//adaptive
				lasso.adaptiv_fit(X, Y);
			}
			else
			{
				lasso.fit(X, Y);
			}
			
			//lasso.fit(X, Y);
			if (lasso.getStatus() != 0)
			{
				error = -1;
				//return error;

				//n_iter *= 2;
				//lasso.fit(X, Y, n_iter, tolerance);
				printf("n_iter=%d error_eps=%f\n", lasso.num_iteration, lasso.error_eps);
				//break;
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
			//else
			{
				const Matrix<dnn_double>& c = lasso.coef;
				for (int k = 0; k < i; k++)
				{
					B(i, k) = c.v[k] / (lasso.sigma(0, k) + 1.0e-10);
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

	void diagram(const std::vector<std::string>& column_names, std::vector<std::string> y_var, std::vector<int>& residual_flag, const char* filename, bool sideways = false, int size=30, char* outformat="png", bool background_Transparent=false, double mutual_information_cut = 0, bool view_confounding_factors = false)
	{
		printf("confounding_factors_upper:%f\n", confounding_factors_upper);
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
		//mutual_information.print("#");

		double mi_max = mutual_information.Max();
		if (mi_max == 0.0) mi_max = 1.0;
		if (mi_max < 1.0e-10) mi_max = 1.0e-10;
		auto& mutual_information_tmp = mutual_information / mi_max;

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

		int confounding_factors_count = 0;
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
				if (this->confounding_factors)
				{
					utf8.fprintf(fp, "%s [fillcolor=\"#FFDF79\", style=\"filled\"]\n", item1.c_str());
				}
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
						if (b_probability.n != 0)
						{

							if (mutual_information_values)
							{
								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\\n%8.1f%%\" color=red penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), b_probability(i,j)*100.0, style);
							}
							else
							{
								utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\\n%8.1f%%\" color=red penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), b_probability(i, j) * 100.0, style);
							}
						}
						else
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
					}
					else
						if (in_line)
						{
							linear_regression_var.push_back(item[j]);

							if (b_probability.n != 0)
							{
								if (mutual_information_values)
								{
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\\n%8.1f%%\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), b_probability(i,j) * 100.0, style);
								}
								else
								{
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\\n%8.1f%%\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), b_probability(i, j) * 100.0, style);
								}
							}
							else
							{
								if (mutual_information_values)
								{
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), style);
								}
								else
								{
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=blue penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
								}
							}
						}
						else
						{
							if (b_probability.n != 0)
							{
								if (mutual_information_values)
								{

									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)%8.3f\\n%8.1f%%\" color=\"%s\" %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), mutual_information(i, j), b_probability(i,j) * 100.0, line_colors[9 - (int)(9 * mutual_information_tmp(i, j))], style);
								}
								else
								{
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\\n%8.1f%%\" color=black %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), b_probability(i, j) * 100.0, style);
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
									utf8.fprintf(fp, "%s-> %s [label=\"%8.3f(%8.3f)\" color=black %s]\n", item2.c_str(), item1.c_str(), B_tmp(i, j), XCor(i, j), style);
								}
							}
						}
				}
				else
				{
					if (i < j && view_confounding_factors && B_tmp(j, i) == 0.0)
					{
						if (mutual_information(i, j) > confounding_factors_upper )
						{
							confounding_factors_count++;
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
							std::string item3;
							char item[80];
							sprintf(item, "\"Unknown(%d)\"", confounding_factors_count);
							char* style = "style=\"dotted\"";

							utf8.fprintf(fp, "%s [fillcolor=\"#f5f5dc\", style=\"filled\"]\n", item);
							//utf8.fprintf(fp, "%s-> %s [dir=\"both\" label=\"(%8.3f)%8.3f\" color=black penwidth=\"2\" %s]\n", item2.c_str(), item1.c_str(), XCor(i, j), mutual_information(i, j), style);

							//utf8.fprintf(fp, "%s-> %s [label=\"---(%8.3f)%8.3f\" color=black %s]\n", item, item2.c_str(), XCor(i, j), mutual_information(i, j), style);
							//utf8.fprintf(fp, "%s-> %s [label=\"---(%8.3f)%8.3f\" color=black %s]\n", item, item1.c_str(), XCor(i, j), mutual_information(i, j), style);
							utf8.fprintf(fp, "%s-> %s [color=black %s]\n", item, item2.c_str(), style);
							utf8.fprintf(fp, "%s-> %s [color=black %s]\n", item, item1.c_str(), style);
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

	Matrix<dnn_double> inv_before_sorting_(Matrix<dnn_double>& y)
	{
		std::vector<int> r = replacement;
		Matrix<dnn_double> x;
		
		x = (Substitution(r).transpose().inv())*y*Substitution(r).inv();

		return x;
	}

	int fit(Matrix<dnn_double>& X, const int max_ica_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
		remove("confounding_factors.txt");
		error = 0;

		residual_error = Matrix<dnn_double>(X.m, X.n);
		residual_error_independ = Matrix<dnn_double>(X.n, X.n);
		intercept = Matrix<dnn_double>().zeros(X.n, 1);

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

		int inv_error = 0;
		Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv(&inv_error);
		if (inv_error != 0)
		{
			error = -1;
			printf("ERROR:W_ica\n");
			return error;
		}
		int error_ = 0;
		Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal(&error_);
		if (error_ != 0)
		{
			error = -1;
			printf("ERROR:W_ica_\n");
			return error;
		}
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
		inv_error = 0;
		Matrix<dnn_double>& W_ica_perm = (Substitution(replace).inv(&inv_error)*W_ica);
		W_ica_perm.print_e("Replacement->W_ica_perm");
		if (inv_error != 0)
		{
			error = -1;
			printf("ERROR:P^-1*Wica\n");
			return error;
		}
		//D^-1
		Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
		Matrix<dnn_double> D2(diag_vector(D));
		(D2.Reciprocal()).print_e("1/D");

		//W_ica_perm_D=I - D^-1*(P^-1*Wica)
		Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal(&error_)));
		if (error_ != 0)
		{
			error = -1;
			printf("ERROR:W_ica_perm_D\n");
			return error;
		}

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

		calc_mutual_information(X, mutual_information, bins);
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
			calc_mutual_information( residual_error, residual_error_independ, bins);
		}
		
		double c_factors = residual_error_independ.Max();
		residual_error_independ.print_csv("confounding_factors_independ.csv");
		FILE* fp = fopen("confounding_factors.txt", "w");
		if (fp)
		{
			fprintf(fp, "%f\n", c_factors);
			fclose(fp);
		}

		B_pre_sort = this->before_sorting_(B);

		if (error)
		{
			printf("WARNING:No valid path was found.\n");
		}

		return error;
	}

	void calc_mutual_information_r(Matrix<dnn_double>& X, Matrix<dnn_double>& rerr, Matrix<dnn_double>& info, int bins = 30, bool nonlinner_cor = false)
	{
		info = info.zeros(X.n, rerr.n);
		printf("info %d x %d\n", info.m, info.n);
#pragma omp parallel for
		for (int j = 0; j < X.n; j++)
		{
			for (int i = 0; i < rerr.n; i++)
			{
				info(i, j) = 0;
				if (i == j)
				{
					continue;
				}

				Matrix<dnn_double>& x = X.Col(i);
				Matrix<dnn_double>& y = rerr.Col(j);

				double tmp;
				if (nonlinner_cor)
				{
					if (use_hsic)
					{
						HSIC hsic;
						//tmp = fabs(hsic.value_(x, y, 30, 50));
						tmp = fabs(hsic.value(x, y, 200));
					}
					else
					{
						tmp = independ_test(x, y);
					}
				}
				else
				{
					MutualInformation I(x, y, bins);
					tmp = I.Information();
				}
				info(i, j) = tmp;
			}
		}
	}
	void calc_mutual_information( Matrix<dnn_double>& X, Matrix<dnn_double>& info, int bins = 30, bool nonlinner_cor = false)
	{
		info = info.zeros(X.n, X.n);
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

				double tmp;
				if (nonlinner_cor)
				{
					if (use_hsic)
					{
						HSIC hsic;
						//tmp = fabs(hsic.value_(x, y, 30, 50));
						tmp = fabs(hsic.value(x, y, 200));
					}
					else
					{
						tmp = independ_test(x, y);
					}
				}
				else
				{
					MutualInformation I(x, y, bins);
					tmp = I.Information();
				}
				info(i, j) = tmp;
			}
		}
	}

	double res_error(Matrix<dnn_double>& xs, Matrix<dnn_double>& μ, Matrix<dnn_double>& b)
	{
		double rerr = 0;
		//b.print("b");
#pragma omp parallel for
		for (int j = 0; j < xs.m; j++)
		{
			Matrix<dnn_double> μT(xs.n, 1);
			Matrix<dnn_double> x(xs.n, 1);
			Matrix<dnn_double> y(xs.n, 1);
			for (int i = 0; i < xs.n; i++)
			{
				x(i, 0) = xs(j, i);
				μT(i, 0) = μ(j, i);
			}
			Matrix<dnn_double>& rr = y - b * y - μT;
			for (int i = 0; i < xs.n; i++)
			{
				rerr += rr(0, i);
			}
		}
		return rerr;
	}

	double dir_change(Matrix<dnn_double>& x, Matrix<dnn_double>& μ)
	{
		Matrix<dnn_double> B_sv = B;
		std::vector<int> replacement_sv = replacement;
		
		Matrix<dnn_double>& b = this->before_sorting_(B);
		double rerr = res_error(x, μ, b);

		double edge_dir = 0;
		int dir_chk = 0;
		do {
			dir_chk++;
			int num = 0;
			int ok = 0;
			for (int i = 0; i < x.n; i++)
			{
				for (int j = i + 1; j < x.n; j++)
				{
					if (fabs(b(i, j)) < 0.001)
					{
						continue;
					}

					lingam_reg reg;

					num++;
					double d;
					bool r = reg.dir(x.Col(i), x.Col(j), d);
					if (fabs(d) > 10.0)
					{
						if (!r)
						{
							b(j, i) = b(i, j);
							b(i, j) = 0;
							//printf("NG %d -> %d (%f)\n", i, j, d);
						}
						else
						{
							//printf("OK %d -> %d (%f)\n", i, j,d);
							ok++;
						}
					}
				}
			}
			edge_dir = (double)ok / (double)num;
			printf("%d/%d (%f)\n", ok, num, 100.0*edge_dir);

			if (edge_dir > 0.7)
			{
				this->inv_before_sorting_(b);
				Matrix<dnn_double> b_est_tmp = b;
				B = AlgorithmC(b_est_tmp, x.n);
			}
			else
			{
				break;
			}
		} while (dir_chk < 3);

		b = this->before_sorting_(B);
		double rerr2 = res_error(x, μ, b);

		if (rerr2 > rerr)
		{
			B_sv = B_sv;
			replacement = replacement_sv;
			return 0;
		}
		return edge_dir;
	}

	void dir_change_(Matrix<dnn_double>& x, Matrix<dnn_double>& μ)
	{
		std::default_random_engine engine(123456789);
		std::uniform_real_distribution<> rnd(0.0, 1.0);

		Matrix<dnn_double> B_sv = B;
		std::vector<int> replacement_sv = replacement;

		Matrix<dnn_double>& b = this->before_sorting_(B);

		bool change = false;
		for (int i = 0; i < x.n; i++)
		{
			for (int j = i + 1; j < x.n; j++)
			{
				if (fabs(b(i, j)) < 0.0001)
				{
					continue;
				}

				if (rnd(engine) < 0.8)
				{
					continue;
				}

				b(j, i) = b(i, j);
				b(i, j) = 0;
				change = true;
				break;
			}
			if (change) break;
		}

		this->inv_before_sorting_(b);
		Matrix<dnn_double> b_est_tmp = b;
		B = AlgorithmC(b_est_tmp, x.n);

		b = this->before_sorting_(B);
	}

	int early_stopping = 0;
	double prior_knowledge_rate = 1.0;
	std::vector<int> prior_knowledge;
	double distribution_rate = 0.1;
	double temperature_alp = 0.95;
	double rho = 3.0;
	double loss_value = 0.0;
	//double mu_max_value = 10.0;
	int loss_function = 0;
	
	inline void fput_loss(const char* loss, double best_residual, double best_independ, double best_min_value)
	{
		try
		{
			std::ofstream ofs(loss, std::ios::app);
			if (
				best_residual >= 2.0 ||
				best_independ >= 2.0 ||
				best_min_value >= 2.0)
			{
				ofs << 2 << "," << 2 << "," << 2 << std::endl;
			}
			else
			{
				ofs << best_residual << "," << best_independ << "," << best_min_value << std::endl;
			}
			ofs.flush();
			ofs.close();
		}
		catch (...)
		{
			printf("----\n");
			//getchar();
		}
	}
	int fit2(Matrix<dnn_double>& X_, const int max_ica_iteration= MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
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

		Matrix<double> X = X_;

		int bootstrapN = 0;

		if (use_bootstrap)
		{
#if _SAMPLING
			int m = bootstrap_sample;
			X = Matrix<double>(m, X_.n);
#endif
			Matrix<double> b(X.n, X.n);
			b_probability = b.zeros(X.n, X.n);
		}
		else
		{
			b_probability = Matrix<double>();
		}
		std::uniform_int_distribution<> bootstrap(0, X.m - 1);

		Mu = Matrix<dnn_double>().zeros(X.m, X.n);
		Matrix<dnn_double> B_best_sv;
		auto replacement_best = replacement;
		residual_error = Matrix<dnn_double>(X.m, X.n);
		residual_error_independ = Matrix<dnn_double>(X.n, X.n);
		
		auto residual_error_best = residual_error;
		auto residual_error_independ_best = residual_error_independ;

		input = X_;
		double min_value =Abs(X_).Min();
		if (min_value < 1.0e-3) min_value = 1.0e-3;
		double max_value = Abs(X_).Max();
		if (max_value < 1.0e-3) max_value = 1.0e-3;

		//std::random_device seed_gen;
		//std::default_random_engine engine(seed_gen());
		std::default_random_engine engine(123456789);
		//std::student_t_distribution<> dist(12.0);
		std::uniform_real_distribution<> acceptance_rate(0.0, 1.0);
		//std::uniform_real_distribution<> rate_cf(-1.0/ mu_max_value, 1.0/ mu_max_value);
		std::uniform_real_distribution<> knowledge_rate(0.0, 1.0);
		//std::uniform_real_distribution<> noise(-min_value*0.1, min_value*0.1);
		std::uniform_real_distribution<> noise(-0.1, 0.1);
		std::normal_distribution<> intercept_noise(0, 6.0);

		std::uniform_real_distribution<> mu_zero(0.0, 1.0);
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
		Matrix<dnn_double>μ_sv;
		std::uniform_int_distribution<> var_select(0, X.n-1);

		intercept = Matrix<dnn_double>().zeros(X.n, 1);
		Matrix<dnn_double> intercept_best;

		double abs_residual_errormax = -1;

		//μ = μ.zeros(μ.m, μ.n);
		μ = μ.Rand()*rho;
		μ_sv = μ;
		//std::student_t_distribution<> dist(6.0);
		//std::normal_distribution<> dist(0.0, 20.0);
		//std::uniform_real_distribution<> dist(-mu_max_value, mu_max_value);

		double t_p = 12.0*rho;
		if (t_p <= 6.0) t_p = 7.0;
		std::uniform_real_distribution<> student_t_dist(6.0, t_p);
		std::vector<int> dist_t_param(X.n);
		std::vector<int> dist_t_param_best;
		std::vector<int> dist_t_param_sv;
		for (int i = 0; i < X.n; i++)
		{
			dist_t_param[i] = student_t_dist(engine);
		}
		dist_t_param_best = dist_t_param;
		dist_t_param_sv = dist_t_param;

		const char* loss = "lingam_loss.dat";
		try
		{
			std::ofstream ofs(loss, std::ios::out);
			ofs.close();
		}
		catch (...)
		{
		}

		std::uniform_real_distribution<> gg_rho(1, 30);
		std::uniform_real_distribution<> gg_beta(1, 30);
		std::uniform_int_distribution<> gg_seed(1, -1395630315);
		//printf("*************** %d\n", (int)123456789123456789);

		double gg_rho_param = 1;
		double gg_beta_param = 1;

		double start_delta = -1.0;
		double start_value = -1.0;
		double start_independ = -1.0;
		double start_residual = -1.0;
		double weight1 = 1.0;
		double weight2 = 1.5;
		int neighborhood_search = 0;
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

			if (use_bootstrap)
			{
#if _SAMPLING
				for (int i = 0; i < X.m; i++)
				{
					const int row_id = bootstrap(mt);
					for (int j = 0; j < X.n; j++)
					{
						X(i, j) = X_(row_id, j);
					}
				}
#endif
			}
			Matrix<dnn_double> xs = X;


			//if (kk > 0)
			{
				int var = var_select(engine);
				//printf("var:%d\n", var);

				if (accept)
				{
#if 10
					double c = 1;
					if (accept < 32)
					{
						c = 1.0 / pow(2.0, accept);
					}
					else
					{
						c = 1.0e-10;
					}
					for (int j = 0; j < X.m; j++)
					{
						μ(j, var) += noise(engine) * c;
					}
					//intercept(var, 0) += intercept_noise(engine)*c;
#else
					dist_t_param[var] += 0.001;

					auto& dist = std::student_t_distribution<>(dist_t_param[var]);

					gg_rho_param += 0.0001;
					gg_beta_param += 0.0001;
					gg_random gg(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ7
					for (int j = 0; j < X.m; j++)
					{
						const double rate = distribution_rate * gg.rand();
						μ(j, var) = rate /*+ noise(engine)*/;
					}
#endif

				}
				else
				{
					dist_t_param[var] = student_t_dist(engine);

					//auto& dist = std::student_t_distribution<>(dist_t_param[var]);
					auto& dist = std::normal_distribution<>(0.0, dist_t_param[var]);

#if 10
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ
					
					//Average distribution（center)
					double rate = distribution_rate * dist(engine);
					
					if (10)
					{
						gg_rho_param = gg_rho(engine);
						gg_beta_param = gg_beta(engine);
					}
					gg_random gg = gg_random(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));

					for (int j = 0; j < X.m; j++)
					{
						if (10)
						{
							//Generalized_Gaussian distribution
							μ(j, var) = rate + distribution_rate * gg.rand();
						}
						else
						{
							//Uniform distribution
							μ(j, var) = rate + distribution_rate * noise(engine);
						}
					}
					//if (use_intercept)
					//{
					//	intercept(var, 0) = intercept_noise(engine);
					//	//intercept(var, 0) = 0;
					//}
#else
					gg_rho_param = gg_rho(engine);
					gg_beta_param = gg_beta(engine);
					gg_random gg(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ7
					for (int j = 0; j < X.m; j++)
					{
						const double rate = distribution_rate * gg.rand();
						μ(j, var) = rate/* + distribution_rate * noise(engine)*/;
					}
#endif
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

			//intercept.print("intercept");

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
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
			}
			catch (...)
			{
				//error = -99;
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}

			try
			{
				int inv_error = 0;

				Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv(&inv_error);
				if (inv_error != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
				int error_ = 0;
				Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal(&error_);
				if (error_ != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

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
				inv_error = 0;
				Matrix<dnn_double>& W_ica_perm = (Substitution(replace).inv(&inv_error)*W_ica);
				//W_ica_perm.print_e("Replacement->W_ica_perm");
				if (inv_error != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

				//D^-1
				Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
				Matrix<dnn_double> D2(diag_vector(D));
				//(D2.Reciprocal()).print_e("1/D");

				error_ = 0;
				//W_ica_perm_D=I - D^-1*(P^-1*Wica)
				Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal(&error)));
				if (error_ != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

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
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
			}
			catch (std::exception& e)
			{
				printf("Exception:%s\n", e.what());
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}
			catch (...)
			{
				printf("Exception\n");
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}

			//{
			//	auto& b = before_sorting_(B);
			//	b.print("b");
			//	auto& c = inv_before_sorting_(b);

			//	(B - c).print("B-c");
			//}

			//dir_change_(xs, μ);

			//float r = 0.0;
			bool cond = true;
			{
				auto& b = before_sorting_(B);

				if (kk > 0 && prior_knowledge.size())
				{
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
								cond = false;
								break;
							}
							//printf("%d -> %d NG\n", jj, ii);
						}
						else
						{
							if (fabs(b(ii, jj)) > 0.001 && fabs(b(jj, ii)) < 0.001)
							{
								cond = false;
								break;
								//printf("%d -> %d NG\n", jj, ii);
							}
							if (fabs(b(ii, jj)) > 0.001 || fabs(b(jj, ii)) < 0.001)
							{
								cond = false;
								break;
								//printf("%d -> %d NG\n", jj, ii);
							}
						}
						if (!cond)
						{
							if (prior_knowledge_rate < knowledge_rate(engine))
							{
								cond = true;
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

				if (use_intercept)
				{
					//切片の推定
					for (int i = 0; i < xs.n; i++)
					{
						intercept(i, 0) = Median(μ.Col(i));
						//intercept(i, 0) = μ.Col(i).mean();
					}
				}

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
						if (use_intercept)
						{
							//切片の分離
							μT(i, 0) = μ(j, i) - intercept(i, 0);
						}
						// y = B*y + (μ- intercept)+ e
					}
					// y = B*y + (μ- intercept)+intercept + e
					// y = B*y + μ + e
					Matrix<dnn_double>& rr = y - b * y - μT - intercept;
					for (int i = 0; i < xs.n; i++)
					{
						//r += rr(0, i)*rr(0, i);
						residual_error(j, i) = rr(0, i);
					}
				}
				//r /= (xs.m*xs.n);

				//Evaluation of independence
				calc_mutual_information( residual_error, residual_error_independ, bins);
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

			double independ = residual_error_independ.Max();
			double residual = abs_residual_errormax_cur / abs_residual_errormax;
			
			//printf("abs_residual_errormax:%f residual:%f\n", abs_residual_errormax, residual);

			//{
			//	double w = weight1 + weight2;
			//	weight1 = weight1 / w;
			//	weight2 = weight2 / w;
			//}

			double w_tmp = sqrt(best_independ + best_residual);
			weight1 = best_residual / w_tmp;
			weight2 = best_independ / w_tmp;
			if (kk == 0)
			{
				double w_tmp = sqrt(independ + residual);
				weight1 = residual / w_tmp;
				weight2 = independ / w_tmp;
			}

			double value;
			if (loss_function == 0)
			{
				value = max(weight1*residual, weight2*independ) + 0.0001*(weight1 * residual + weight2 * independ);
			}
			else
			{
				value = log(1 + fabs(residual - independ) + weight1 *residual + weight2*independ);

			}

			if (start_value < 0)
			{
				start_independ = independ;
				start_residual = residual;
				start_value = value;
				loss_value = value;
				start_delta = fabs(residual - independ);
			}
			//printf("value:%f (%f) %f\n", value, log(1 + fabs(residual - independ)), weight1 *residual + weight2 * independ);
			bool accept_ = false;

			if (best_residual > residual && best_independ > independ)
			{
				accept_ = true;
			}else
			if (best_min_value > value)
			{
				accept_ = true;
			}

			if (kk == 0) accept_ = true;


			double rt = (double)kk / (double)confounding_factors_sampling;
			temperature = pow(temperature_alp, rt);

			if (!accept_ && cond)
			{
				double alp = acceptance_rate(engine);

				double th = -1.0;
				if (best_min_value < value)
				{
					th = exp((best_min_value - value) / temperature);
				}
				{
					char buf[256];
					sprintf(buf, "[%d][%f] %f:%f temperature:%f  alp:%f < th:%f %s", kk, fabs((best_min_value - value)), best_min_value, value, temperature, alp, th, (alp < th) ? "true" : "false");
					if (1)
					{
						DWORD  bytesWritten = 0;
						if (alp < th)
						{
							SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY );
						}
						else
						{
							SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
						}
						WriteFile(hStdout, buf, strlen(buf), &bytesWritten, NULL);
						SetConsoleTextAttribute(hStdout, csbi.wAttributes);
						WriteFile(hStdout, "\n", 1, &bytesWritten, NULL);
					}
					else
					{
						printf(buf); printf("\n");
					}
				}
				if (alp < th)
				{
					accept_ = true;
				}
			}

			if (!cond) accept_ = false;
			if ( independ < 1.0e-4)  accept_ = false;

			bool best_update = false;
			if (accept_)
			{
				if (use_bootstrap)
				{
					auto b = before_sorting_(B);
					for (int j = 0; j < xs.n; j++)
					{
						for (int i = 0; i < xs.n; i++)
						{
							if (fabs(b(i, j)) > 0.01)
							{
								b_probability(i, j) += 1;
							}
						}
					}
					bootstrapN++;
				}

				//printf("+\n");
				if (best_min_value >= value|| (best_residual > residual || best_independ > independ))
				{
					double d1 = (residual - best_residual);
					double d2 = (independ - best_independ);
					if (best_residual > residual && best_independ < independ || best_residual < residual && best_independ > independ )
					{
						if (fabs(d1)+fabs(d2) < 0.1*(1.0 - rt))
						{
							best_update = true;
						}
					}
					else
					{
						if (best_residual > residual && best_independ > independ)
						{
							best_update = true;
						}
					}
				}

				if (loss_function == 0)
				{
					if (best_min_value < value)
					{
						best_update = false;
					}
				}
				if (best_update || kk == 0)
				{
					best_min_value = value;
					best_residual = residual;
					best_independ = independ;
					loss_value = value;
				}
				//{
				//	auto& b = before_sorting_(B);
				//	b.print("accept-b");
				//}

				μ_sv = μ;
				dist_t_param_sv = dist_t_param;
				neighborhood_search = xs.n;

				accept++;
				reject = 0;
				if (best_update)
				{		
					no_accept_count = 0;
					update_count++;
					char buf[256];
					sprintf(buf, "@[%d/%d] %f (ind:%f,err:%f)accept:%d", kk, confounding_factors_sampling - 1, best_min_value, independ, residual, accept);

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

					//intercept_best.print("intercept_best");
					//μ.print("μ");
					Mu = μ;
					B_best_sv = B;
					B_pre_sort = B;
					modification_input = xs;
					replacement_best = replacement;
					residual_error_best = residual_error;
					residual_error_independ_best = residual_error_independ;
					
					intercept_best = intercept;
					dist_t_param_best = dist_t_param;

					calc_mutual_information(X, mutual_information, bins);
					
					if (use_bootstrap)
					{
						auto tmp = b_probability;
						b_probability /= bootstrapN;
						b_probability *= 0.99;

						save(std::string("lingam.model"));

						b_probability = tmp;
					}
					else
					{
						save(std::string("lingam.model"));
					}
					if (independ + residual < 0.000001)
					{
						printf("convergence!\n");
						break;
					}
				}

				//for (int i = 0; i < xs.n; i++)
				//{
				//	double residual_error_mean = residual_error.Col(i).mean();
				//	intercept(i, 0) += residual_error_mean/xs.n;
				//}
				fflush(stdout);
			}
			else
			{
				accept = 0;
				reject++;
				//neighborhood_search--;
				//if (neighborhood_search <= 0)
				//{
				//	neighborhood_search = 0;
				//}
				//else
				//{
				//	μ = Mu;
				//	dist_t_param = dist_t_param_best;
				//	intercept = intercept_best;
				//	accept = 1;
				//	reject = 0;
				//	printf("------------------\n");
				//}
				//if (acceptance_rate(engine) > 0.95)
				//{
				//	//μ = Mu;
				//	//dist_t_param = dist_t_param_best;
				//	//intercept = intercept_best;
				//	accept = 1;
				//	reject = 0;
				//	printf("------------------\n");
				//}
			}
			fput_loss(loss, best_residual, best_independ, best_min_value);

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

		intercept = intercept_best;
		residual_error = residual_error_best;
		residual_error_independ = residual_error_independ_best;
		replacement = replacement_best;
		B = B_best_sv;

		if (use_bootstrap)
		{
			b_probability /= bootstrapN;
			b_probability *= 0.99;
		}

		calc_mutual_information(X, mutual_information, bins);

		double c_factors = residual_error_independ.Max();
		residual_error_independ.print_csv("confounding_factors_info.csv");
		FILE* fp = fopen("confounding_factors.txt", "w");
		if (fp)
		{
			fprintf(fp, "%f\n", c_factors);
			fclose(fp);
		}
		B_pre_sort = this->before_sorting_(B);
		if (update_count < 2)
		{
			error = 1;
			printf("WARNING:No valid path was found.\n");
		}
		return error;
	}

	int fit3(Matrix<dnn_double>& X_, const int max_ica_iteration = MAX_ITERATIONS, const dnn_double tolerance = TOLERANCE)
	{
#ifdef USE_LIBTORCH
		if ( use_gpu) torch_setDevice("gpu:0");
#endif
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

		Matrix<double> X = X_;

		int bootstrapN = 0;

		if (use_bootstrap)
		{
#if _SAMPLING
			int m = bootstrap_sample;
			X = Matrix<double>(m, X_.n);
#endif
			Matrix<double> b(X.n, X.n);
			b_probability = b.zeros(X.n, X.n);
		}
		else
		{
			b_probability = Matrix<double>();
		}
		std::uniform_int_distribution<> bootstrap(0, X.m - 1);

		Mu = Matrix<dnn_double>().zeros(X.m, X.n);
		Matrix<dnn_double> B_best_sv;
		auto replacement_best = replacement;
		residual_error = Matrix<dnn_double>(X.m, X.n);
		residual_error_independ = Matrix<dnn_double>(X.n, X.n);

		auto residual_error_best = residual_error;
		auto residual_error_independ_best = residual_error_independ;

		input = X_;
		double min_value = Abs(X_).Min();
		if (min_value < 1.0e-3) min_value = 1.0e-3;
		double max_value = Abs(X_).Max();
		if (max_value < 1.0e-3) max_value = 1.0e-3;

		//std::random_device seed_gen;
		//std::default_random_engine engine(seed_gen());
		std::default_random_engine engine(123456789);
		//std::student_t_distribution<> dist(12.0);
		std::uniform_real_distribution<> acceptance_rate(0.0, 1.0);
		//std::uniform_real_distribution<> rate_cf(-1.0/ mu_max_value, 1.0/ mu_max_value);
		std::uniform_real_distribution<> knowledge_rate(0.0, 1.0);
		//std::uniform_real_distribution<> noise(-min_value*0.1, min_value*0.1);
		std::uniform_real_distribution<> noise(-0.1, 0.1);
		std::normal_distribution<> intercept_noise(0, 6.0);

		std::uniform_real_distribution<> mu_zero(0.0, 1.0);
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
		Matrix<dnn_double>μ_sv;
		std::uniform_int_distribution<> var_select(0, X.n - 1);

		intercept = Matrix<dnn_double>().zeros(X.n, 1);
		Matrix<dnn_double> intercept_best;

		double abs_residual_errormax = -1;
		double abs_independ_max = -1;

		//μ = μ.zeros(μ.m, μ.n);
		μ = μ.Rand() * rho;
		μ_sv = μ;
		//std::student_t_distribution<> dist(6.0);
		//std::normal_distribution<> dist(0.0, 20.0);
		//std::uniform_real_distribution<> dist(-mu_max_value, mu_max_value);

		double t_p = 12.0 * rho;
		if (t_p <= 6.0) t_p = 7.0;
		std::uniform_real_distribution<> student_t_dist(6.0, t_p);
		std::vector<int> dist_t_param(X.n);
		std::vector<int> dist_t_param_best;
		std::vector<int> dist_t_param_sv;
		for (int i = 0; i < X.n; i++)
		{
			dist_t_param[i] = student_t_dist(engine);
		}
		dist_t_param_best = dist_t_param;
		dist_t_param_sv = dist_t_param;

		const char* loss = "lingam_loss.dat";
		try
		{
			std::ofstream ofs(loss, std::ios::out);
			ofs.close();
		}
		catch (...)
		{
		}

		std::uniform_real_distribution<> gg_rho(1, 30);
		std::uniform_real_distribution<> gg_beta(1, 30);
		std::uniform_int_distribution<> gg_seed(1, -1395630315);
		//printf("*************** %d\n", (int)123456789123456789);

		double gg_rho_param = 1;
		double gg_beta_param = 1;

		double start_delta = -1.0;
		double start_value = -1.0;
		double start_independ = -1.0;
		double start_residual = -1.0;
		double weight1 = 1.0;
		double weight2 = 1.5;
		int neighborhood_search = 0;
		for (int kk = 0; kk < confounding_factors_sampling; kk++)
		{
			start = std::chrono::system_clock::now();

			//parameter_search mode!!
			if (early_stopping < 0)
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

			if (use_bootstrap)
			{
#if _SAMPLING
				for (int i = 0; i < X.m; i++)
				{
					const int row_id = bootstrap(mt);
					for (int j = 0; j < X.n; j++)
					{
						X(i, j) = X_(row_id, j);
					}
				}
#endif
			}
			Matrix<dnn_double> xs = X;


			//if (kk > 0)
			{
				int var = var_select(engine);
				//printf("var:%d\n", var);

				if (accept)
				{
#if 10
					double c = 1;
					if (accept < 32)
					{
						c = 1.0 / pow(2.0, accept);
					}
					else
					{
						c = 1.0e-10;
					}
					for (int j = 0; j < X.m; j++)
					{
						μ(j, var) += noise(engine) * c;
					}
					//intercept(var, 0) += intercept_noise(engine)*c;
#else
					dist_t_param[var] += 0.001;

					auto& dist = std::student_t_distribution<>(dist_t_param[var]);

					gg_rho_param += 0.0001;
					gg_beta_param += 0.0001;
					gg_random gg(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ7
					for (int j = 0; j < X.m; j++)
					{
						const double rate = distribution_rate * gg.rand();
						μ(j, var) = rate /*+ noise(engine)*/;
					}
#endif

				}
				else
				{
					dist_t_param[var] = student_t_dist(engine);

					//auto& dist = std::student_t_distribution<>(dist_t_param[var]);
					auto& dist = std::normal_distribution<>(0.0, dist_t_param[var]);

#if 10
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ

					//Average distribution（center)
					double rate = distribution_rate * dist(engine);

					if (10)
					{
						gg_rho_param = gg_rho(engine);
						gg_beta_param = gg_beta(engine);
					}
					gg_random gg = gg_random(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));

					for (int j = 0; j < X.m; j++)
					{
						if (10)
						{
							//Generalized_Gaussian distribution
							μ(j, var) = rate + distribution_rate * gg.rand();
						}
						else
						{
							//Uniform distribution
							μ(j, var) = rate + distribution_rate * noise(engine);
						}
					}
					//if (use_intercept)
					//{
					//	intercept(var, 0) = intercept_noise(engine);
					//	//intercept(var, 0) = 0;
					//}
#else
					gg_rho_param = gg_rho(engine);
					gg_beta_param = gg_beta(engine);
					gg_random gg(gg_rho_param, gg_beta_param, 0);
					gg.seed(gg_seed(engine));
					//#pragma omp parallel for <-- 乱数の順序が変わってしまうから並列化したらダメ7
					for (int j = 0; j < X.m; j++)
					{
						const double rate = distribution_rate * gg.rand();
						μ(j, var) = rate/* + distribution_rate * noise(engine)*/;
					}
#endif
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

			//intercept.print("intercept");

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
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
			}
			catch (...)
			{
				//error = -99;
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}

			try
			{
				int inv_error = 0;

				Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv(&inv_error);
				if (inv_error != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
				int error_ = 0;
				Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal(&error_);
				if (error_ != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

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
				inv_error = 0;
				Matrix<dnn_double>& W_ica_perm = (Substitution(replace).inv(&inv_error) * W_ica);
				//W_ica_perm.print_e("Replacement->W_ica_perm");
				if (inv_error != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

				//D^-1
				Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
				Matrix<dnn_double> D2(diag_vector(D));
				//(D2.Reciprocal()).print_e("1/D");

				error_ = 0;
				//W_ica_perm_D=I - D^-1*(P^-1*Wica)
				Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal(&error)));
				if (error_ != 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}

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
					fput_loss(loss, best_residual, best_independ, best_min_value);
					continue;
				}
			}
			catch (std::exception& e)
			{
				printf("Exception:%s\n", e.what());
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}
			catch (...)
			{
				printf("Exception\n");
				fput_loss(loss, best_residual, best_independ, best_min_value);
				continue;
			}

			//{
			//	auto& b = before_sorting_(B);
			//	b.print("b");
			//	auto& c = inv_before_sorting_(b);

			//	(B - c).print("B-c");
			//}

			//dir_change_(xs, μ);
			double error_dag = 0.0;

			//float r = 0.0;
			bool cond = true;
			{
				auto& b = before_sorting_(B);

				if (kk > 0 && prior_knowledge.size())
				{
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
								cond = false;
								break;
							}
							//printf("%d -> %d NG\n", jj, ii);
						}
						else
						{
							if (fabs(b(ii, jj)) > 0.001 && fabs(b(jj, ii)) < 0.001)
							{
								cond = false;
								break;
								//printf("%d -> %d NG\n", jj, ii);
							}
							if (fabs(b(ii, jj)) > 0.001 || fabs(b(jj, ii)) < 0.001)
							{
								cond = false;
								break;
								//printf("%d -> %d NG\n", jj, ii);
							}
						}
						if (!cond)
						{
							if (prior_knowledge_rate < knowledge_rate(engine))
							{
								cond = true;
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

				if (use_intercept)
				{
					//切片の推定
					for (int i = 0; i < xs.n; i++)
					{
						intercept(i, 0) = Median(μ.Col(i));
						//intercept(i, 0) = μ.Col(i).mean();
					}
				}

				bool nonlinear = false;
#ifdef USE_LIBTORCH
				nonlinear = true;
				// nonlinear
				if (nonlinear)
				{
					std::vector<float_t> mean_x;
					std::vector<float_t> sigma_x;
					std::vector<float_t> mean_y;
					std::vector<float_t> sigma_y;

					for (int i = 0; i < xs.m; i++)
					{
						residual_error(i, 0) = 0;
					}

					//b.print("b");
					Matrix<dnn_double> x;
					Matrix<dnn_double> y;
					for (int i = 0; i < B.n; i++)
					{
						// y = B*y + μ + e
						x = Matrix<dnn_double>();
						y = X.Col(this->replacement[i]);

						for (int k = 0; k < B.n; k++)
						{
							if (k == i)continue;
							if (fabs(B(i, k)) > 0.01)
							{
								if (x.m == 0) x = X.Col(this->replacement[k]);
								else x = x.appendCol(X.Col(this->replacement[k]));
							}
						}

						for (int i = 0; i < B.n; i++)
						{
							if (x.m == 0) x = μ.Col(this->replacement[i]) * condition(engine);
							else x = x.appendCol(μ.Col(this->replacement[i])) * condition(engine);
						}

						if (x.n == 0)
						{
							for (int j = 0; j < xs.m; j++)
							{
								residual_error(j, i) = 0;
							}
							continue;
						}
						//printf("\n");
						//y.print("y");
						//x.print("x");

						tiny_dnn::tensor_t tx;
						tiny_dnn::tensor_t ty;
						MatrixToTensor(x, tx);
						MatrixToTensor(y, ty);

						normalizeZ(tx, mean_x, sigma_x);
						normalizeZ(ty, mean_y, sigma_y);

						int n_train_epochs_ = 60;
						int n_minibatch_ = x.m/ 5;
						if (n_minibatch_ > 2048)  n_minibatch_ = 2048;
						if (n_minibatch_ < 1) n_minibatch_ = 1;
						int input_size_ = 16;

						int n_layers_ = 3;
						int dropout_ = 0.0;
						int n_hidden_size_ = 32;
						int fc_hidden_size_ = 32;
						float learning_rate_ = 0.1;

						float clip_gradients_ = 0;
						int use_cnn_ = 0;
						int use_add_bn_ = 0;
						int use_cnn_add_bn_ = 0;
						int residual_ = 0;
						int padding_prm_ = 0;

						int classification_ = 0;
						char* weight_init_type_ = "xavier";
						char* activation_fnc_ = "tanh";
						int early_stopping_ = 10;
						char* opt_type_ = "adam_";
						bool batch_shuffle_ = true;
						int test_mode_ = 0;

						torch_params(
							n_train_epochs_,
							n_minibatch_,
							input_size_,

							n_layers_,
							dropout_,
							n_hidden_size_,
							fc_hidden_size_,
							learning_rate_,

							clip_gradients_,
							use_cnn_,
							use_add_bn_,
							use_cnn_add_bn_,
							residual_,
							padding_prm_,

							classification_,
							weight_init_type_,
							activation_fnc_,
							early_stopping_,
							opt_type_,
							batch_shuffle_,
							test_mode_
						);

#if 10
						auto on_enumerate_epoch = [&]() {};
						auto on_enumerate_minibatch = [&]() {};
#else
						int epoch = 1;
						// create callback
						auto on_enumerate_epoch = [&]() {
							std::cout << "\nEpoch " << epoch << "/" << n_train_epochs_ << " finished. " << std::endl;
							float loss = torch_get_Loss(n_minibatch_);
							std::cout << "loss :" << loss << std::endl;
							++epoch;
						};
						auto on_enumerate_minibatch = [&]() {};
#endif
						chrono::system_clock::time_point start, end;

						torch_train_init();

						start = chrono::system_clock::now();
						try {
							torch_train_fc(tx, ty, n_minibatch_, n_train_epochs_, "", on_enumerate_minibatch, on_enumerate_epoch);
						}
						catch (std::exception& err)
						{
							std::cout << err.what() << std::endl;
						}
						catch (...)
						{
							printf("exception!\n");
						}
						end = chrono::system_clock::now();
						printf("fitting %lf[ms]\n", static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0));
#pragma omp parallel for
						for (int j = 0; j < x.m; j++)
						{
							tiny_dnn::vec_t& predict_y = torch_predict(tx[j]);

							double prd_y = (double)predict_y[0] * sigma_y[0] + mean_y[0];
							double obs_y = (double)ty[j][0] * sigma_y[0] + mean_y[0];
							residual_error(j, i) = (obs_y - prd_y);
						}
						residual_error.print("residual_error");
						torch_delete_model();
						printf("Evaluation of independence\n");

						start = chrono::system_clock::now();
						
						//Evaluation of independence
						calc_mutual_information_r(x, residual_error.Col(i), residual_error_independ, bins, true);
						
						end = chrono::system_clock::now();
						printf("independence check %lf[ms]\n", static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0));

						if (error_dag < residual_error_independ.Max())
						{
							error_dag = residual_error_independ.Max();
						}
					}

					printf("error_dag:%f (%s)\n", error_dag, (use_hsic) ? "HSIC" : "corr(x,tanh(y)");
					double error_dag_treshold = 1.0;
					if (use_hsic) error_dag_treshold = 0.1;
					if (error_dag > error_dag_treshold) {
						std::cout << "残差と説明変数が独立ではないため間違ったDAGのためキャンセル" << std::endl;
						continue;
					}

					printf("Evaluation of independence\n");
					//Evaluation of independence
					//calc_mutual_information(residual_error, residual_error_independ, bins);
					calc_mutual_information(residual_error, residual_error_independ, bins/*, true*/);
				}
#endif
				if (!nonlinear)
				{
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
							if (use_intercept)
							{
								//切片の分離
								μT(i, 0) = μ(j, i) - intercept(i, 0);
							}
							// y = B*y + (μ- intercept)+ e
						}
						// y = B*y + (μ- intercept)+intercept + e
						// y = B*y + μ + e
						Matrix<dnn_double>& rr = y - b * y - μT - intercept;
						for (int i = 0; i < xs.n; i++)
						{
							//r += rr(0, i)*rr(0, i);
							residual_error(j, i) = rr(0, i);
						}
					}
					//r /= (xs.m*xs.n);

					//Evaluation of independence
					calc_mutual_information(residual_error, residual_error_independ, bins);
				}
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

			double  abs_independ_max_cur = max(error_dag, residual_error_independ.Max());
			
			if (abs_independ_max < 0)
			{
				abs_independ_max = abs_independ_max_cur;
				if (!use_hsic) abs_independ_max = 1.0;
			}
			if (abs_independ_max < 1.0e-10)
			{
				abs_independ_max = 1.0e-10;
			}			
			//if (abs_residual_errormax < abs_residual_errormax_cur)
			//{
			//	abs_residual_errormax = abs_residual_errormax_cur;
			//}
			//residual_error.print("residual_error");
			//printf("residual_error max:%f\n", Abs(residual_error).Max());

			double independ = max(error_dag, residual_error_independ.Max())/ abs_independ_max;
			double residual = abs_residual_errormax_cur / abs_residual_errormax;

			printf("abs_residual_errormax:%f residual:%f independ:%f\n", abs_residual_errormax, residual, independ);

			//{
			//	double w = weight1 + weight2;
			//	weight1 = weight1 / w;
			//	weight2 = weight2 / w;
			//}

			double w_tmp = sqrt(best_independ + best_residual);
			weight1 = best_residual / w_tmp;
			weight2 = best_independ / w_tmp;
			if (kk == 0)
			{
				double w_tmp = sqrt(independ + residual);
				weight1 = residual / w_tmp;
				weight2 = independ / w_tmp;
			}

			double value;
			if (loss_function == 0)
			{
				value = max(weight1 * residual, weight2 * independ) + 0.0001 * (weight1 * residual + weight2 * independ);
			}
			else
			{
				value = log(1 + fabs(residual - independ) + weight1 * residual + weight2 * independ);

			}

			if (start_value < 0)
			{
				start_independ = independ;
				start_residual = residual;
				start_value = value;
				loss_value = value;
				start_delta = fabs(residual - independ);
			}
			//printf("value:%f (%f) %f\n", value, log(1 + fabs(residual - independ)), weight1 *residual + weight2 * independ);
			bool accept_ = false;

			if (best_residual > residual && best_independ > independ)
			{
				accept_ = true;
			}
			else
				if (best_min_value > value)
				{
					accept_ = true;
				}

			if (kk == 0) accept_ = true;


			double rt = (double)kk / (double)confounding_factors_sampling;
			temperature = pow(temperature_alp, rt);

			if (!accept_ && cond)
			{
				double alp = acceptance_rate(engine);

				double th = -1.0;
				if (best_min_value < value)
				{
					th = exp((best_min_value - value) / temperature);
				}
				{
					char buf[256];
					sprintf(buf, "[%d][%f] %f:%f temperature:%f  alp:%f < th:%f %s", kk, fabs((best_min_value - value)), best_min_value, value, temperature, alp, th, (alp < th) ? "true" : "false");
					if (1)
					{
						DWORD  bytesWritten = 0;
						if (alp < th)
						{
							SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
						}
						else
						{
							SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
						}
						WriteFile(hStdout, buf, strlen(buf), &bytesWritten, NULL);
						SetConsoleTextAttribute(hStdout, csbi.wAttributes);
						WriteFile(hStdout, "\n", 1, &bytesWritten, NULL);
					}
					else
					{
						printf(buf); printf("\n");
					}
				}
				if (alp < th)
				{
					accept_ = true;
				}
			}

			if (!cond) accept_ = false;
			if (independ < 1.0e-4)  accept_ = false;

			bool best_update = false;
			if (accept_)
			{
				if (use_bootstrap)
				{
					auto b = before_sorting_(B);
					for (int j = 0; j < xs.n; j++)
					{
						for (int i = 0; i < xs.n; i++)
						{
							if (fabs(b(i, j)) > 0.01)
							{
								b_probability(i, j) += 1;
							}
						}
					}
					bootstrapN++;
				}

				//printf("+\n");
				if (best_min_value >= value || (best_residual > residual || best_independ > independ))
				{
					double d1 = (residual - best_residual);
					double d2 = (independ - best_independ);
					if (best_residual > residual && best_independ < independ || best_residual < residual && best_independ > independ)
					{
						if (fabs(d1) + fabs(d2) < 0.1 * (1.0 - rt))
						{
							best_update = true;
						}
					}
					else
					{
						if (best_residual > residual && best_independ > independ)
						{
							best_update = true;
						}
					}
				}

				if (loss_function == 0)
				{
					if (best_min_value < value)
					{
						best_update = false;
					}
				}
				if (best_update || kk == 0)
				{
					best_min_value = value;
					best_residual = residual;
					best_independ = independ;
					loss_value = value;
				}
				//{
				//	auto& b = before_sorting_(B);
				//	b.print("accept-b");
				//}

				μ_sv = μ;
				dist_t_param_sv = dist_t_param;
				neighborhood_search = xs.n;

				accept++;
				reject = 0;
				if (best_update)
				{
					no_accept_count = 0;
					update_count++;
					char buf[256];
					sprintf(buf, "@[%d/%d] %f (ind:%f,err:%f)accept:%d", kk, confounding_factors_sampling - 1, best_min_value, independ, residual, accept);

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

					//intercept_best.print("intercept_best");
					//μ.print("μ");
					Mu = μ;
					B_best_sv = B;
					B_pre_sort = B;
					modification_input = xs;
					replacement_best = replacement;
					residual_error_best = residual_error;
					residual_error_independ_best = residual_error_independ;

					intercept_best = intercept;
					dist_t_param_best = dist_t_param;

					calc_mutual_information(X, mutual_information, bins);

					if (use_bootstrap)
					{
						auto tmp = b_probability;
						b_probability /= bootstrapN;
						b_probability *= 0.99;

						save(std::string("lingam.model"));

						b_probability = tmp;
					}
					else
					{
						save(std::string("lingam.model"));
					}
					if (independ + residual < 0.000001)
					{
						printf("convergence!\n");
						break;
					}
				}

				//for (int i = 0; i < xs.n; i++)
				//{
				//	double residual_error_mean = residual_error.Col(i).mean();
				//	intercept(i, 0) += residual_error_mean/xs.n;
				//}
				fflush(stdout);
			}
			else
			{
				accept = 0;
				reject++;
				//neighborhood_search--;
				//if (neighborhood_search <= 0)
				//{
				//	neighborhood_search = 0;
				//}
				//else
				//{
				//	μ = Mu;
				//	dist_t_param = dist_t_param_best;
				//	intercept = intercept_best;
				//	accept = 1;
				//	reject = 0;
				//	printf("------------------\n");
				//}
				//if (acceptance_rate(engine) > 0.95)
				//{
				//	//μ = Mu;
				//	//dist_t_param = dist_t_param_best;
				//	//intercept = intercept_best;
				//	accept = 1;
				//	reject = 0;
				//	printf("------------------\n");
				//}
			}
			fput_loss(loss, best_residual, best_independ, best_min_value);

			//if ( confounding_factors_sampling >= 4000 && reject > confounding_factors_sampling / 4)
			//{
			//	break;
			//}

			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//printf("[%d/%d]%8.3f[msec]\n", kk, confounding_factors_sampling - 1, elapsed);

			elapsed_ave += elapsed;
			double t1 = (confounding_factors_sampling - 1 - kk) * (elapsed_ave * 0.001 / (kk + 1));
			if (kk != 0 && elapsed_ave * 0.001 / (kk + 1) < 1 && kk % 20)
			{
				continue;
			}

			if (t1 < 60)
			{
				printf("[%d/%d]Total elapsed time:%.3f[sec] Estimated end time:%.3f[sec] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001, t1, accept);
				fflush(stdout);
				continue;
			}
			t1 /= 60.0;
			if (t1 < 60)
			{
				printf("[%d/%d]Total elapsed time:%.3f[min] Estimated end time:%.3f[min] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001 / 60.0, t1, reject);
				fflush(stdout);
				continue;
			}
			t1 /= 60.0;
			if (t1 < 24)
			{
				printf("[%d/%d]Total elapsed time:%.3f[h] Estimated end time:%.3f[h] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001 / 60.0 / 60.0, t1, reject);
				fflush(stdout);
				continue;
			}
			t1 /= 365;
			if (t1 < 365)
			{
				printf("[%d/%d]Total elapsed time:%.3f[days] Estimated end time:%.3f[days] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001 / 60 / 60 / 24.0 / 365.0, t1, reject);
				fflush(stdout);
				continue;
			}
			printf("[%d/%d]Total elapsed time:%.3f[years] Estimated end time:%.3f[years] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001 / 60 / 60 / 24.0 / 365.0, t1, reject);
			fflush(stdout);
		}

		intercept = intercept_best;
		residual_error = residual_error_best;
		residual_error_independ = residual_error_independ_best;
		replacement = replacement_best;
		B = B_best_sv;

		if (use_bootstrap)
		{
			b_probability /= bootstrapN;
			b_probability *= 0.99;
		}

		calc_mutual_information(X, mutual_information, bins);

		double c_factors = residual_error_independ.Max();
		residual_error_independ.print_csv("confounding_factors_info.csv");
		FILE* fp = fopen("confounding_factors.txt", "w");
		if (fp)
		{
			fprintf(fp, "%f\n", c_factors);
			fclose(fp);
		}
		B_pre_sort = this->before_sorting_(B);
		if (update_count < 2)
		{
			error = 1;
			printf("WARNING:No valid path was found.\n");
		}
		return error;
	}

	void Causal_effect(std::vector<std::string>& header_names, double scale = 1)
	{
		std::vector<std::string> header_names2;
		for (int i = 0; i < header_names.size(); i++)
		{
			char buf[128];
			char buf2[128];
			strcpy(buf, header_names[i].c_str());
			if (buf[0] == '\"')
			{
				strcpy(buf2, buf + 1);
				buf2[strlen(buf2) - 1] = '\0';
				strcpy(buf, buf2);
			}
			header_names2.push_back(buf);
		}

		printf("Causal_effect\n");
		FILE* fp = fopen("Causal_effect.r", "w");
		if (fp == NULL)
		{
			return;
		}

		int plot = 1;
		fprintf(fp, "library(ggplot2)\n");
		fprintf(fp, "library(gridExtra)\n");
		fprintf(fp, "library(RColorBrewer)\n");
		for (int i = 0; i < variableNum; i++)
		{

			int count = 0;
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				count++;
			}
			if (count <= 0) continue;

			fprintf(fp, "x_ <- data.frame(\n");
			int s = 0;

			fprintf(fp, "%s=c(\n", header_names2[i].c_str());

			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				fprintf(fp, "\"%s\"", header_names2[j].c_str());
				s++;
			}
			fprintf(fp, "),\n");

			s = 0;
			fprintf(fp, "effect=c(\n");
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				fprintf(fp, "%.2f", B(i, j));
				s++;
			}
			fprintf(fp, ")\n");
			fprintf(fp, ")\n");

			fprintf(fp, "g%d <- ggplot(x_, aes(x = %s, y = effect, fill = effect))\n", plot, header_names2[i].c_str());
			fprintf(fp, "g%d <- g%d + geom_bar(stat = \"identity\")\n", plot, plot);
			fprintf(fp, "g%d <- g%d + labs(title = \"Causal_effect\")\n", plot, plot);
			//fprintf(fp, "g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'YlOrRd\')))\n", plot, plot);
			fprintf(fp, "g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'Spectral\')))\n", plot, plot);
			fprintf(fp, "g%d <- g%d + theme(text = element_text(size = 12))\n", plot, plot);
			fprintf(fp, "#plot(g%d)\n", plot);
			plot++;
		}

		fprintf(fp, "g <- grid.arrange(");
		for (int i = 1; i < plot; i++)
		{
			fprintf(fp, "g%d", i);
			if (i < plot - 1) fprintf(fp, ",");
		}
		fprintf(fp, ")\n");
		fprintf(fp, "ggplot2::ggsave(\"Causal_effect.png\",g,width = 15*%.2f, height = 8*%.2f, units = \"cm\", dpi = 400)\n", scale, scale);

		fclose(fp);
		printf("Causal_effect end\n");
	}

	void b_probability_barplot(std::vector<std::string>& header_names, double scale = 1)
	{
		if (b_probability.n != this->variableNum)
		{
			return;
		}
		std::vector<std::string> header_names2;
		for (int i = 0; i < header_names.size(); i++)
		{
			char buf[128];
			char buf2[128];
			strcpy(buf, header_names[i].c_str());
			if (buf[0] == '\"')
			{
				strcpy(buf2, buf + 1);
				buf2[strlen(buf2) - 1] = '\0';
				strcpy(buf, buf2);
			}
			header_names2.push_back(buf);
		}

		printf("b_probability_barplot\n");
		FILE* fp = fopen("b_probability_barplot.r", "w");
		if (fp == NULL)
		{
			return;
		}

		int plot = 1;
		fprintf(fp, "library(ggplot2)\n");
		fprintf(fp, "library(gridExtra)\n");
		fprintf(fp, "library(RColorBrewer)\n");
		for (int i = 0; i < variableNum; i++)
		{

			int count = 0;
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.05) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				count++;
			}
			if (count <= 1) continue;

			fprintf(fp, "x_ <- data.frame(\n");
			int s = 0;

			fprintf(fp, "%s=c(\n", header_names2[i].c_str());

			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.05) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				fprintf(fp, "\"%s\"", header_names2[j].c_str());
				s++;
			}
			fprintf(fp, "),\n");

			s = 0;
			fprintf(fp, "probability=c(\n");
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.05) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				fprintf(fp, "%.2f", b_probability(i, j) * 100.0);
				s++;
			}
			fprintf(fp, ")\n");
			fprintf(fp, ")\n");

			fprintf(fp, "g%d <- ggplot(x_, aes(x = %s, y = probability, fill = probability))\n", plot, header_names2[i].c_str());
			fprintf(fp, "g%d <- g%d + geom_bar(stat = \"identity\")\n", plot, plot);
			fprintf(fp, "g%d <- g%d + labs(title = \"probability\")\n", plot, plot);
			fprintf(fp, "g%d <- g%d + theme(text = element_text(size = 12))\n", plot, plot);
			fprintf(fp, "#g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'Blues\')))\n", plot, plot);
			fprintf(fp, "#plot(g%d)\n", plot);
			plot++;
		}

		fprintf(fp, "g <- grid.arrange(");
		for (int i = 1; i < plot; i++)
		{
			fprintf(fp, "g%d", i);
			if (i < plot - 1) fprintf(fp, ",");
		}
		fprintf(fp, ")\n");
		fprintf(fp, "ggplot2::ggsave(\"b_probability.png\",g,width = 15*%.2f, height = 8*%.2f, units = \"cm\", dpi = 400)\n", scale, scale);

		fclose(fp);
		printf("b_probability_barplot end\n");
	}
};
#endif
