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

#include "../../include/util/xgboost_util.h"
//#define USE_EIGEN

#ifdef USE_EIGEN
#include "../../include/statistical/RegularizationRegression_eigen_version.h"
#endif
#include <regex>

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
extern "C" _LIBRARY_EXPORTS int torch_train_init_seed(int seed);
extern "C" _LIBRARY_EXPORTS void torch_setDevice(const char* device_name);
extern "C" _LIBRARY_EXPORTS void torch_delete_model();
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_predict(tiny_dnn::vec_t x);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_post_predict(tiny_dnn::vec_t x);
extern  _LIBRARY_EXPORTS tiny_dnn::vec_t torch_invpost_predict(tiny_dnn::vec_t x);
extern "C" _LIBRARY_EXPORTS float torch_get_Loss(int batch);
extern "C" _LIBRARY_EXPORTS float torch_get_train_loss();

extern "C" _LIBRARY_EXPORTS void torch_params(
	int n_train_epochs_,
	int n_minibatch_,
	int input_size_,

	int n_layers_,
	float dropout_,
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
	int shuffle_seed_,
	bool L1_loss_,
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

extern "C" _LIBRARY_EXPORTS void torch_train_post_fc(
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
	double confounding_factors_upper2 = 0.05;
	bool random_pattern = false;
	bool L1_loss = false;

	vector<int> replacement;
	Matrix<dnn_double> B;
	Matrix<dnn_double> B_pre_sort;
	Matrix<dnn_double> input;
	Matrix<dnn_double> input_sample;
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

	bool nonlinear = false;
	bool use_hsic = false;
	bool use_gpu = false;
	int n_epoch = 40;
	int n_unit = 20;
	int n_layer = 5;
	float learning_rate = 0.001;
	std::vector<std::string> exper;
	std::vector<std::string> colnames;
	std::string activation_fnc = "relu";
	std::vector<std::vector<double>> observed;
	std::vector<std::vector<double>> predict;
	std::vector< std::vector<int>> colnames_id;
	std::vector< std::vector<int>> hidden_colnames_id;
	Matrix<dnn_double> importance_B;

	std::string R_cmd_path = "";
	std::string optimizer = "rmsprop";
	int minbatch = -1;
	double u1_param = 0.001;
	double dropout_rate = 0.0;
	bool use_pnl = 0;
	std::string layout = "dot";

	bool _Causal_Search_Experiment = false;
	bool eval_mode = false;

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
		if (this->exper.size() > 0)
		{
			fp = fopen((filename + ".Non-linear_regression_equation").c_str(), "w");
			if (fp)
			{
				for (int i = 0; i < exper.size(); i++)
				{
					fprintf(fp, "%s\n", exper[i].c_str());
				}
				fclose(fp);
			}
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
			input_sample.print_csv((char*)(filename + ".input_sample.csv").c_str(), this->colnames);

			residual_error_independ.print_csv((char*)(filename + ".residual_error_independ.csv").c_str());
			residual_error.print_csv((char*)(filename + ".residual_error.csv").c_str());

			fp = fopen((filename + ".colnames_id").c_str(), "w");
			if (fp)
			{
				fprintf(fp, "%d\n", colnames_id.size());
				for (int ii = 0; ii < colnames_id.size(); ii++)
				{
					fprintf(fp, "%d\n", colnames_id[ii].size());
					for (int jj = 0; jj < colnames_id[ii].size(); jj++)
					{
						fprintf(fp, "%d\n", colnames_id[ii][jj]);
					}
				}
				fclose(fp);
			}			
			fp = fopen((filename + ".hidden_colnames_id").c_str(), "w");
			if (fp)
			{
				fprintf(fp, "%d\n", hidden_colnames_id.size());
				for (int ii = 0; ii < hidden_colnames_id.size(); ii++)
				{
					fprintf(fp, "%d\n", hidden_colnames_id[ii].size());
					for (int jj = 0; jj < hidden_colnames_id[ii].size(); jj++)
					{
						fprintf(fp, "%d\n", hidden_colnames_id[ii][jj]);
					}
				}
				fclose(fp);
			}
			if (nonlinear)
			{
				this->fit_state(colnames_id, predict, observed);
				this->scatter(colnames_id, predict, observed);
				this->scatter2(colnames_id, predict, observed);
				this->error_hist(colnames_id);
			}
			if ( 1 )
			{
				auto B_sv = B;
				auto replacement_sv = replacement;
				before_sorting();

				if (use_bootstrap)
				{
					b_probability.print_csv((char*)(filename + ".b_probability.csv").c_str());
					b_probability_barplot(this->colnames, 1.0);
				}
				Causal_effect(this->colnames, 1.0);

				if (nonlinear && R_cmd_path != "")
				{
					importance_B = importance();
					importance_B.print_csv((char*)(filename + ".importance_B.csv").c_str());
				}
				B = B_sv;
				replacement = replacement_sv;
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

			CSVReader csv11((filename + ".input_sample.csv"), ',', false);
			input_sample = csv11.toMat();
			printf("input_sample\n"); fflush(stdout);

			if (use_bootstrap)
			{
				CSVReader csv10((filename + ".b_probability.csv"), ',', false);
				b_probability = csv10.toMat();
				printf("b_probability\n"); fflush(stdout);
			}

			fp = fopen((filename + ".colnames_id").c_str(), "r");
			if (fp)
			{
				colnames_id.clear();

				fgets(buf, 256, fp);
				int num = atoi(buf);

				for (int ii = 0; ii < num; ii++)
				{
					fgets(buf, 256, fp);
					int n = atoi(buf);
					std::vector<int> ids;
					for (int jj = 0; jj < n; jj++)
					{
						fgets(buf, 256, fp);
						int id = atof(buf);
						ids.push_back(id);
					}
					colnames_id.push_back(ids);
				}
				fclose(fp);
			}
			fp = fopen((filename + ".hidden_colnames_id").c_str(), "r");
			if (fp)
			{
				hidden_colnames_id.clear();

				fgets(buf, 256, fp);
				int num = atoi(buf);

				for (int ii = 0; ii < num; ii++)
				{
					fgets(buf, 256, fp);
					int n = atoi(buf);
					std::vector<int> ids;
					for (int jj = 0; jj < n; jj++)
					{
						fgets(buf, 256, fp);
						int id = atof(buf);
						ids.push_back(id);
					}
					hidden_colnames_id.push_back(ids);
				}
				fclose(fp);
			}			
			if (nonlinear && R_cmd_path != "")
			{
				FILE* fp = fopen((filename + ".importance_B.csv").c_str(), "r");
				if (fp)
				{
					fclose(fp);
					try
					{
						CSVReader csv10((filename + ".importance_B.csv"), ',', false);
						if (!csv10.parser_error)
						{
							importance_B = csv10.toMat();
						}
						else
						{
							importance_B = Matrix<dnn_double>();
						}

						//importance_B.print("importance_B");
					}
					catch (std::exception& e)
					{
						printf("importance_B load exception:%s\n", e.what());
					}
					printf("load importance_B\n"); fflush(stdout);
				}
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

	void diagram(const std::vector<std::string>& column_names, std::vector<std::string> y_var, std::vector<int>& residual_flag, const char* filename, bool sideways = false, int size=30, char* outformat="png", bool background_Transparent=false, double mutual_information_cut = 0, bool view_confounding_factors = false, std::string title="")
	{
		printf("view_confounding_factors:%d\n", view_confounding_factors?1:0);
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
		mutual_information.print("#mutual_information");

		double mi_max = mutual_information.Max();
		if (mi_max == 0.0) mi_max = 1.0;
		if (mi_max < 1.0e-10) mi_max = 1.0e-10;
		auto& mutual_information_tmp = mutual_information / mi_max;

		utf8str utf8;
		FILE* fp = fopen(filename, "w");
		utf8.fprintf(fp, "digraph {\n");
		if (title != "")
		{
			utf8.fprintf(fp, "labelloc=\"t\";\n");
			utf8.fprintf(fp, "label=\"%s\";\n", title.c_str());
		}
		if (background_Transparent)
		{
			utf8.fprintf(fp, "graph[bgcolor=\"#00000000\"];\n");
		}
		utf8.fprintf(fp, "size=\"%d!\"\n", size);
		utf8.fprintf(fp, "layout=%s\n", layout.c_str());
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
				//else
				{
					if (i < j && view_confounding_factors/* && B_tmp(j, i) == 0.0*/)
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
		input_sample = X;
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

		printf("replacement\n");
		for (int x = 0; x < replacement.size(); x++)
			std::cout << x << "," << replacement[x] << "\t";
		printf("\n");
		b_est.print_e("b_est");
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

		colnames_id.clear();
		for (int i = 0; i < B.n; i++)
		{
			std::vector<int> name_id;

			name_id.push_back(this->replacement[i]);
			for (int j = 0; j < B.n; j++)
			{
				if (fabs(B(i, j)) > 0.01)
				{
					name_id.push_back(this->replacement[j]);
				}
			}
			colnames_id.push_back(name_id);
		}

		B_pre_sort = this->before_sorting_(B);

		if (error)
		{
			printf("WARNING:No valid path was found.\n");
		}

		return error;
	}

	void calc_mutual_information_r( Matrix<dnn_double>& X, Matrix<dnn_double>& rerr, Matrix<dnn_double>& info, int bins = 30, bool nonlinner_cor = false)
	{
		info = info.zeros(X.n, rerr.n);
		printf("info %d x %d\n", info.m, info.n);
		if (X.n <= 1) return;

#pragma omp parallel for
		for (int j = 0; j < X.n; j++)
		{
			for (int i = 0; i < rerr.n; i++)
			{
				info(i, j) = 0;

				Matrix<dnn_double>& x = X.Col(j);
				Matrix<dnn_double>& y = rerr.Col(i);

				//if (x.m == y.m)
				//{
				//	printf("x.m == y.m:%d\n", x.m - y.m);
				//}
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
	
	double plot_max_loss = 2.0;

	inline void fput_loss(const char* loss, double best_residual, double best_independ, double best_min_value)
	{
		try
		{
			std::ofstream ofs(loss, std::ios::app);
			if (
				best_residual >= plot_max_loss ||
				best_independ >= plot_max_loss ||
				best_min_value >= plot_max_loss)
			{
				ofs << plot_max_loss << "," << plot_max_loss << "," << plot_max_loss << std::endl;
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
		input_sample = X_;
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
					
					colnames_id.clear();
					for (int i = 0; i < B.n; i++)
					{
						std::vector<int> name_id;

						name_id.push_back(this->replacement[i]);
						for (int j = 0; j < B.n; j++)
						{
							if (fabs(B(i, j)) > 0.01)
							{
								name_id.push_back(this->replacement[j]);
							}
						}
						colnames_id.push_back(name_id);
					}
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

		vector<vector<int>> pattern_list;
		vector<Matrix<dnn_double>> pattern_list_bmat;

		vector<int> pattern;
		for (int i = 0; i < X_.n; i++) pattern.push_back(i);

		if (X_.n < 8)
		{
			do {
				pattern_list.push_back(pattern);
			} while (next_permutation(pattern.begin(), pattern.end()));
			pattern_list_bmat.resize(pattern_list.size());
		}
		else
		{
			random_pattern = false;
		}
		int accept_pattern_idx = -1;
		int current_pattern_idx = -1;

		printf("pattern:%d\n", pattern_list.size());
		std::vector<int> pattern_count(pattern_list.size(), 0);

		CONSOLE_SCREEN_BUFFER_INFO csbi;
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hStdout, &csbi);

		std::chrono::system_clock::time_point  start, end;
		double elapsed;
		double elapsed_ave = 0.0;

		logmsg = false;
		error = 0;

		Matrix<double> X = X_;

		std::default_random_engine eng(123456789);
		int resample = 5000;
		if (X_.m > resample)
		{
			vector<size_t> permutation(X_.m, 0);
			for (size_t i = 0; i < X_.m; i++)
				permutation[i] = i;

			std::shuffle(permutation.begin(), permutation.end(), eng);

			Matrix<dnn_double> x_tmp(resample, X_.n);
			for (int i = 0; i < X_.n; i++)
			{
				for (int j = 0; j < resample; j++)
				{
					x_tmp(j, i) = X_(permutation[j], i);
				}
			}
			X = x_tmp;
		}


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
		modification_input = X_;
		input_sample = X;

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
		std::uniform_real_distribution<> condition2(0.0, 1.0);


		int no_accept_count = 0;
		int update_count = 0;
		int reject = 0;
		int accept = 0;
		double temperature = 0.0;

		std::vector<std::string> best_exper;
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

		int t_p = 12 * rho;
		if (t_p <= 6) t_p = 7;
		std::uniform_real_distribution<> student_t_dist(6, t_p);
		std::vector<double> dist_t_param(X.n);
		std::vector<double> dist_t_param_best;
		std::vector<double> dist_t_param_sv;
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

		std::uniform_real_distribution<> gg_rho(1, 50);
		std::uniform_real_distribution<> gg_beta(1, 50);
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

		double fit_loss = 0.005;

		std::vector<int> pattern_pikup;
		int train_size = X.m*0.7;
		if (train_size < 100) train_size = X.m;

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

			if (X_.m > resample)
			{
				vector<size_t> permutation(X_.m, 0);
				for (size_t i = 0; i < X_.m; i++)
					permutation[i] = i;

				std::shuffle(permutation.begin(), permutation.end(), eng);

				Matrix<dnn_double> x_tmp(resample, X_.n);
				for (int i = 0; i < X_.n; i++)
				{
					for (int j = 0; j < resample; j++)
					{
						x_tmp(j, i) = X_(permutation[j], i);
					}
				}
				X = x_tmp;
			}
			Matrix<dnn_double> xs = X;
			input_sample = X;


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

			Matrix<dnn_double> b;
			bool cond = true;

			if (pattern_pikup.size() == 0 ||!random_pattern)
			{
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

					//printf("replacement\n");
					//for (int x = 0; x < replacement.size(); x++)
					//	std::cout << x << "," << replacement[x] << "\t";
					//printf("\n");
					//b_est.print_e("b_est");
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

				b = before_sorting_(B);

				cond = true;
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
			}

			struct pttern_s {
				int idx;
				int count;
			};
			std::vector<VectorIndex<int>> pttern_s;

			// 300ケースの結果から得た因果グラフのパターンを抽出
			// 上位10パターンを選んで以降はその10パターンに対して最適なものを算出する
			if (random_pattern)
			{
				if (kk >= 300 && pattern_pikup.size() == 0)
				{
					printf("make pattern_pikup\n");

					pattern_pikup.clear();

					for (int jj = 0; jj < pattern_list.size(); jj++)
					{
						VectorIndex<int> a;
						a.id = jj;
						a.abs_dat = pattern_count[jj];
						pttern_s.push_back(a);
					}
					std::sort(pttern_s.begin(), pttern_s.end());

					for (int i = pttern_s.size() - 1; i >= 0; i--)
					{
						printf(")");
						for (int jjj = 0; jjj < pattern_list[pttern_s[i].id].size(); jjj++)
						{
							printf("%d ", pattern_list[pttern_s[i].id][jjj]);
						}
						printf(")  %d\n", pttern_s[i].abs_dat);
						if (pttern_s[i].abs_dat == 0) break;

						pattern_pikup.push_back(pttern_s[i].id);
						if (pattern_pikup.size() == 10) break;
					}
					if (pttern_s.size() > 0)
					{
						printf("pttern_s[pttern_s.size() - 1].abs_dat:%d\n", pttern_s[pttern_s.size() - 1].abs_dat);
						if (pttern_s[pttern_s.size() - 1].abs_dat == 0)
						{
							pattern_pikup.clear();
						}
					}
					if (pattern_list.size() < 10)
					{
						for (int ii = 0; ii < pattern_list.size(); ii++)
						{
							pattern_pikup.push_back(ii);
						}
					}
					printf("pattern_pikup:%d\n", pattern_pikup.size());
				}
			}

			if (pattern_pikup.size() == 0 || !random_pattern)
			{
				printf("pattern_pikup clear\n");
				pattern_pikup.push_back(-1);
			}

	

			bool edge_dag_error_flg = false;
			printf("pattern_pikup:%d\n", pattern_pikup.size());
			for ( int pt = 0; pt < 1; pt++)
			{
				double error_dag = 0.0;
				const double error_dag_max = 1.0;
				std::vector<std::vector<double>> observed_tmp;
				std::vector<std::vector<double>> predict_tmp;
				std::vector< std::vector<int>> name_id;
				std::vector< std::vector<int>> hidden_name_id;
				std::vector<std::string> exper_tmp;

				{

					if (pattern_pikup[0] >= 0)
					{
						int pt_idx = kk % pattern_pikup.size();
						printf("\nLeading candidate:%d\n", pt_idx);
						printf("(");
						for (int jjj = 0; jjj < pattern_list[pattern_pikup[pt_idx]].size(); jjj++)
						{
							printf("%d ", pattern_list[pattern_pikup[pt_idx]][jjj]);
						}
						printf(")\n\n");

						this->replacement = pattern_list[pattern_pikup[pt_idx]];
						B = pattern_list_bmat[pattern_pikup[pt_idx]];
					}
					else
					{
						pattern_pikup.clear();
					}
#ifdef USE_LIBTORCH
					// nonlinear
					if (nonlinear)
					{
						double max_count = 0;
						int lookup = -1;
						for (int jj = 0; jj < pattern_list.size(); jj++)
						{
							int c = 0;
							for (int i = 0; i < this->replacement.size(); i++)
							{
								if (pattern_list[jj][i] == this->replacement[i]) c++;
							}
							if (c == this->replacement.size())
							{
								pattern_count[jj] += 1;
								if (pattern_list_bmat[jj].m == 0)
								{
									pattern_list_bmat[jj] = B;
								}
								else
								{
									pattern_list_bmat[jj] = (pattern_list_bmat[jj] + B) * 0.5;
								}
								lookup = jj;
								current_pattern_idx = jj;
							}
							if (max_count < pattern_count[jj])max_count = pattern_count[jj];
						}
						//if (random_pattern)
						//{
						//	if (max_count > 1)
						//	{
						//		if (!accept || accept_pattern_idx >= 0)
						//		{
						//			int idx = -1;
						//			if (!accept)
						//			{
						//				//if (condition2(engine) > 0.20)
						//				{
						//					do {
						//						idx = (int)((condition2(engine) + 0.5) * (pattern_list.size() - 1));
						//						if (lookup == -1) break;
						//					} while (idx == lookup);
						//				}
						//			}
						//			else
						//			{
						//				if (condition2(engine) > 0.50)
						//				{
						//					idx = accept_pattern_idx;
						//				}
						//				else
						//				{
						//					do {
						//						idx = (int)((condition2(engine) + 0.5) * (pattern_list.size() - 1));
						//						if (lookup == -1) break;
						//					} while (idx == lookup);
						//				}
						//			}
						//			if (idx >= 0 && idx < pattern_list.size())
						//			{
						//				printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@ %d\n", idx);
						//				if (lookup >= 0)pattern_count[lookup] -= 1;
						//				this->replacement = pattern_list[idx];
						//				pattern_count[idx] += 1;
						//				B = B.zeros(X.n, X.n);
						//				for (int i = 0; i < X.n; i++)
						//				{
						//					for (int j = 0; j < X.n; j++)
						//					{
						//						if (j >= i) break;
						//						B(i, j) = 1;
						//					}
						//				}
						//			}
						//		}
						//	}
						//}

						if (_Causal_Search_Experiment && kk % 20 == 0)
						{
							char fname[256];
							{
								sprintf(fname, "..\\Causal_Search_Experiment\\pattern_count.txt");
							}
							FILE* fp = fopen(fname, "w");
							if (fp)
							{
								for (int jj = 0; jj < pattern_list.size(); jj++)
								{
									for (int jjj = 0; jjj < pattern_list[jj].size(); jjj++)
									{
										fprintf(fp, "%d ", pattern_list[jj][jjj]);
									}
									fprintf(fp, " %d\n", pattern_count[jj]);
								}
								fclose(fp);
							}
						}


						std::vector<float_t> mean_x;
						std::vector<float_t> sigma_x;
						std::vector<float_t> mean_y;
						std::vector<float_t> sigma_y;
						calc_mutual_information(X, mutual_information, bins);

						residual_error = Matrix<dnn_double>().zeros(X.m, X.n);

						//b.print("b");
						//B.print("B");

						if (0)
						{
							char buf[256] = { '\0' };
							for (int i = 0; i < B.n; i++)
							{
								char tmp_buf[256];

								sprintf(tmp_buf, "%d [%s] ", this->replacement[i], colnames[this->replacement[i]].c_str());
								strcat(buf, tmp_buf);
							}
							DWORD  bytesWritten = 0;
							SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY | BACKGROUND_RED | BACKGROUND_BLUE | BACKGROUND_INTENSITY);
							WriteFile(hStdout, buf, strlen(buf), &bytesWritten, NULL);
							SetConsoleTextAttribute(hStdout, csbi.wAttributes);
							WriteFile(hStdout, "\n", 1, &bytesWritten, NULL);
						}
						else
						{
							//printf("\n-----------\n");
							//for (int i = 0; i < B.n; i++)
							//{
							//	printf("%d [%s] ", this->replacement[i], colnames[this->replacement[i]].c_str());
							//}
							//printf("\n-----------\n");
						}
						Matrix<dnn_double> x;
						Matrix<dnn_double> y;
						Matrix<dnn_double> x_input_only;
						for (int i = 0; i < B.n; i++)
						{
							std::vector<int> xv;
							std::vector<int> hidden_xv;

							std::vector<int> xv_input_id;
							//if (accept)
							//{
							//	if (i >= colnames_id.size())
							//	{
							//		continue;
							//	}
							//	for (int k = 0; k < colnames_id[i].size(); k++)
							//	{
							//		for (int j = 0; j < replacement_best.size(); j++)
							//		{
							//			if (colnames_id[i][k] == replacement_best[j])
							//			{
							//				if (k == 0)
							//				{
							//					y = X.Col(j);
							//					xv.push_back(replacement_best[j]);
							//					break;
							//				}
							//				if (x.n == 0) x = X.Col(j);
							//				else x = x.appendCol(X.Col(j));
							//				xv.push_back(replacement_best[j]);
							//				break;
							//			}
							//		}
							//	}
							//	if (xv.size() <= 1)
							//	{
							//		for (int j = 0; j < xs.m; j++)
							//		{
							//			residual_error(j, i) = (X.Col(i))(j, 0);
							//		}
							//		continue;
							//	}

							//	for (int k = 0; k < hidden_colnames_id[i].size(); k++)
							//	{
							//		for (int j = 0; j < replacement_best.size(); j++)
							//		{
							//			if (hidden_colnames_id[i][k] == replacement_best[j])
							//			{
							//				if (x.n == 0) x = X.Col(j);
							//				else x = x.appendCol(X.Col(j));
							//				hidden_xv.push_back(replacement_best[j]);
							//			}
							//		}
							//	}
							//}

							//if (!accept)
							{
								x = Matrix<dnn_double>();
								x_input_only = x;
								y = X.Col(i);
								xv.push_back(this->replacement[i]);
								xv_input_id.push_back(i);

								std::vector<int>xv_id;
								double max_b = 0;
								for (int k = 0; k < B.n; k++)
								{
									if (k >= i) break;
									if (fabs(B(i, k)) > max_b) max_b = fabs(B(i, k));
									if (fabs(B(i, k)) > 0.01)
									{
										xv_id.push_back(k);
									}
								}
								//std::shuffle(xv_id.begin(), xv_id.end(), engine);

								for (int k = 0; k < xv_id.size(); k++)
								{
									//if (xv_id.size() > 1 )
									//{
									//	double corr = y.Cor(X.Col(xv_id[k]));
									//	if (corr > 0.8 && condition2(engine) < 0.5)
									//	{
									//		if (xv.size() == 1 && k == xv_id.size() - 1)
									//		{
									//			/* */
									//		}
									//		else
									//		{
									//			continue;
									//		}
									//	}
									//}
									if (x.n == 0) x = X.Col(xv_id[k]);
									else x = x.appendCol(X.Col(xv_id[k]));
									xv.push_back(this->replacement[xv_id[k]]);
									x_input_only = x;
									xv_input_id.push_back(xv_id[k]);
								}
								if (xv.size() == 1)
								{
									x = X.Col(i);
									hidden_xv.push_back(this->replacement[i]);
								}

								if (xv.size() > 1)
								{
									bool add_hidden = false;
									mutual_information.print("mutual_information");
									//for (int ki = 0; ki < xv.size(); ki++) {
									//	for (int kj = ki+1; kj < xv.size(); kj++) {
									//		if (mutual_information(xv[kj], xv[ki]) > 0.5)
									//		{
									//			add_hidden = true;
									//		}
									//	}
									//}

#if 10
									double max_cofound = 0;
									double max_cofound_id = -1;
									for (int k = 0; k < xv.size(); k++)
									{
										if (max_cofound < mutual_information(i, xv[k]))
										{
											max_cofound = mutual_information(i, xv[k]);
											max_cofound_id = k;
										}
									}
									printf("confounding_factors_upper2:%f max_cofound:%f\n", confounding_factors_upper2, max_cofound);
									for (int k = 0; k < xv.size(); k++)
									{
										printf("mutual_information:%f \n", mutual_information(i, xv[k]));
										if (fabs(mutual_information(i, xv[k])) > confounding_factors_upper2)
										{
											//if (k == max_cofound_id )
											{
												if (x.n == 0) x = μ.Col(xv[k]) * u1_param;
												else x = x.appendCol(μ.Col(xv[k])) * u1_param;
												hidden_xv.push_back(this->replacement[xv[k]]);
											}
										}
									}
#else
									if (x.n == 0) x = μ.Col(i);
									else x = x.appendCol(μ.Col(i));
									hidden_xv.push_back(this->replacement[i]);
#endif
								}
							}


							//printf("\n");
							//y.print("y");
							//x.print("x");
#if 0
							xgboost_util< dnn_double> xgb;
							xgb.set_train(x, y);
							xgb.train(500);

							auto& predict_y = xgb.predict(x);

							std::vector<double> observed_;
							std::vector<double> predict_;
							int nn = x.m / 100;
							if (nn <= 1) nn = x.m;
#pragma omp parallel for
							for (int j = 0; j < x.m; j++)
							{

								double prd_y = (double)predict_y[j];
								double obs_y = (double)y(j, 0);
								residual_error(j, i) = (obs_y - prd_y);

								if (j % nn == 0)
								{
#pragma omp critical
									{
										observed_.push_back(obs_y);
										predict_.push_back(prd_y);
									}
								}
							}
							observed_tmp.push_back(observed_);
							predict_tmp.push_back(predict_);
#else
							tiny_dnn::tensor_t tx;
							tiny_dnn::tensor_t ty;
							MatrixToTensor(x, tx);
							MatrixToTensor(y, ty);

							normalizeZ(tx, mean_x, sigma_x);
							normalizeZ(ty, mean_y, sigma_y);

							int n_train_epochs_ = this->n_epoch;
							if (n_train_epochs_ < 2) n_train_epochs_ = 2;

							int n_minibatch_ = /*0.7**/tx.size() / 5;
							if (n_minibatch_ > 2000)  n_minibatch_ = 2000;
							if (n_minibatch_ < 1) n_minibatch_ = 1;
							if (this->minbatch >= 0)
							{
								if (this->minbatch == 0) n_minibatch_ = train_size;///*0.7**/tx.size();
								else n_minibatch_ = this->minbatch;
							}
							int input_size_ = this->n_unit;

							int n_layers_ = this->n_layer;
							float dropout_ = dropout_rate;
							int n_hidden_size_ = 32;
							int fc_hidden_size_ = 32;
							float learning_rate_ = this->learning_rate;

							float clip_gradients_ = 0;
							int use_cnn_ = 0;
							int use_add_bn_ = 0;
							int use_cnn_add_bn_ = 0;
							int residual_ = 0;
							int padding_prm_ = 0;

							int classification_ = 0;

							char weight_init_type_[16];
							strcpy(weight_init_type_, "xavier");
							char activation_fnc_[16];
							strcpy(activation_fnc_, activation_fnc.c_str());

							int early_stopping_ = 0;
							char opt_type_[16];
							strcpy(opt_type_, optimizer.c_str());
							bool batch_shuffle_ = true;
							int shuffle_seed_ = kk;
							bool L1_loss_ = L1_loss;
							int test_mode_ = 0;

							torch_train_init_seed(shuffle_seed_);

							const int max_epohc_srch = 1;
							for (int jj = 0; jj < max_epohc_srch; jj++)
							{
								printf("dropout:%f\n", dropout_);
								printf("n_minibatch:%d\n", n_minibatch_);
								printf("activation_fnc:%s\n", activation_fnc_);
								printf("learning_rate:%f\n", learning_rate_);
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
									shuffle_seed_,
									L1_loss_,
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
								auto on_enumerate_minibatch = [&]() {
									float loss = torch_get_Loss(n_minibatch_);
									std::cout << "loss :" << loss << std::endl;
								};
#endif

								chrono::system_clock::time_point start, end;

								torch_train_init_seed(shuffle_seed_);

								start = chrono::system_clock::now();
								bool train_error = false;
								try {
									auto tx_ = tx;
									auto ty_ = ty;
									tx_.resize(train_size);
									ty_.resize(train_size);
									printf("tx_:%d\n", tx_.size());
									if (use_pnl)
									{
										torch_train_post_fc(tx_, ty_, n_minibatch_, n_train_epochs_, "", on_enumerate_minibatch, on_enumerate_epoch);
									}
									else
									{
										torch_train_fc(tx_, ty_, n_minibatch_, n_train_epochs_, "", on_enumerate_minibatch, on_enumerate_epoch);
									}
								}
								catch (std::exception& err)
								{
									std::cout << err.what() << std::endl;
									train_error = true;
								}
								catch (...)
								{
									printf("exception!\n");
									train_error = true;
								}
								end = chrono::system_clock::now();
								printf("fitting %lf[ms]\n", static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0));

								std::vector<double> observed_;
								std::vector<double> predict_;

								observed_.resize(x.m);
								predict_.resize(x.m);

#pragma omp parallel for
								for (int j = 0; j < x.m; j++)
								{
									if (use_pnl)
									{
										residual_error(j, i) = 0;
										observed_[j] = 0;
										predict_[j] = 0;
										const int nsample = 1;
										for (int jj = 0; jj < nsample; jj++)
										{
											tiny_dnn::vec_t& fx = torch_predict(tx[j]);		// f(x)
											tiny_dnn::vec_t& gy = torch_post_predict(ty[j]);// g(y)
											tiny_dnn::vec_t& igy = torch_invpost_predict(fx);// g^-1(f(x))

											double prd_y = (double)fx[0] * (double)sigma_y[0] + (double)mean_y[0];		//f(x)
											double obs_y = (double)ty[j][0] * (double)sigma_y[0] + (double)mean_y[0];	//y
											double pst_y = (double)gy[0] * (double)sigma_y[0] + (double)mean_y[0];		//g(y)
											double ipst_y = (double)igy[0] * (double)sigma_y[0] + (double)mean_y[0];	//g^-1(f(x))

											//prd_y = (int)(prd_y * 1000 + 0.5) / 1000.0;
											//obs_y = (int)(obs_y * 1000 + 0.5) / 1000.0;
											//pst_y = (int)(pst_y * 1000 + 0.5) / 1000.0;
											//ipst_y = (int)(ipst_y * 1000 + 0.5) / 1000.0;

											residual_error(j, i) += (pst_y - prd_y);	//g(y) - f(x)
											observed_[j] += obs_y;	// y
											predict_[j] += ipst_y;	//g^-1(f(x)) == y
										}
										residual_error(j, i) /= nsample;
										observed_[j] /= nsample;
										predict_[j] /= nsample;
										if (train_error)
										{
											residual_error(j, i) = 9999;	//g(y) - f(x)
											observed_[j] = 0;	// y
											predict_[j] = 0;	//g^-1(f(x)) == y
										}
										//printf("residual_error:%f\n", residual_error(j, i));
										//printf("observed_:%f\n", observed_[j]);
										//printf("predict_:%f\n", predict_[j]);
									}
									else
									{
										tiny_dnn::vec_t& predict_y = torch_predict(tx[j]);

										double prd_y = (double)predict_y[0] * (double)sigma_y[0] + (double)mean_y[0];
										double obs_y = (double)ty[j][0] * (double)sigma_y[0] + (double)mean_y[0];

										//prd_y = (int)(prd_y * 1000 + 0.5) / 1000.0;
										//obs_y = (int)(obs_y * 1000 + 0.5) / 1000.0;

										residual_error(j, i) = (obs_y - prd_y);
										observed_[j] = obs_y;
										predict_[j] = prd_y;
										if (train_error)
										{
											residual_error(j, i) = 9999;	//g(y) - f(x)
											observed_[j] = 0;	// y
											predict_[j] = 0;	//g^-1(f(x)) == y
										}
									}
								}
								//auto loss = torch_get_Loss(n_minibatch_);
								auto loss = torch_get_train_loss();
								printf("loss:%f\n", loss);
								fflush(stdout);
								if (loss < fit_loss || jj == max_epohc_srch - 1)
								{
									if (loss < fit_loss)
									{
										printf("fit ok!\n");
									}
									else
									{
										printf("fit ng!\n");
									}
									observed_tmp.push_back(observed_);
									predict_tmp.push_back(predict_);
									break;
								}
								n_train_epochs_ *= 2;
							}
#endif
							//if (xv.size() == 1)
							//{
							//	for (int j = 0; j < xs.m; j++)
							//	{
							//		residual_error(j, i) = y.Col(0)(j, 0);
							//	}
							//}

							name_id.push_back(xv);
							hidden_name_id.push_back(hidden_xv);

							residual_error.print("residual_error");

							torch_delete_model();

							printf("xv:%d\n", xv.size());
							fflush(stdout);

							if (xv.size() >= 1)
							{
								B.print("B");
								char tmp_buf[256];
								std::string tmp_buf2;

								if (xv.size() >= 2)
								{
									sprintf(tmp_buf, "x(%d:[%s]) = f_{%d}(x(%d:[%s])", xv[0] + 1, colnames[xv[0]].c_str(), xv[0] + 1, xv[1] + 1, colnames[xv[1]].c_str());
									tmp_buf2 = std::string(tmp_buf);
									if ( xv.size() >= 3 )
									{
										for (int kk = 2; kk < xv.size(); kk++)
										{
											sprintf(tmp_buf, ",x(%d:[%s])", xv[kk] + 1, colnames[xv[kk]].c_str());
											tmp_buf2 += std::string(tmp_buf);
										}
									}
									sprintf(tmp_buf, ")");
									tmp_buf2 += std::string(tmp_buf);
								}
								else
								{
									sprintf(tmp_buf, "x(%d:[%s]) = eps_{%d}", xv[0] + 1, colnames[xv[0]].c_str(), xv[0] + 1);
									tmp_buf2 = std::string(tmp_buf);
								}
								auto eps = Median(residual_error.Col(i));
								sprintf(tmp_buf, " %c %.4f{%.4f ~ %.4f}", (eps < 0) ? '-' : '+', fabs(eps), residual_error.Col(i).Min(), residual_error.Col(i).Max());
								tmp_buf2 += std::string(tmp_buf);
								exper_tmp.push_back(tmp_buf2);
							}

							printf("Evaluation of independence\n");

							start = chrono::system_clock::now();

							//Evaluation of independence
							//x.print("x");
							//x_input_only.print("x_input_only");
							//residual_error.print("residual_error");
							calc_mutual_information_r( x_input_only, residual_error.Col(i), residual_error_independ, bins, true);

							residual_error_independ.print("residual_error_independ");

							end = chrono::system_clock::now();
							printf("independence check %lf[ms]\n", static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0));

							if (error_dag < residual_error_independ.Max())
							{
								error_dag = residual_error_independ.Max();
							}
						}

						printf("error_dag:%f (%s)\n", error_dag, (use_hsic) ? "HSIC" : "corr(x,tanh(y)");
						fflush(stdout);
						double error_dag_treshold = 0.9;
						if (use_hsic) error_dag_treshold = 0.1;
						if (error_dag > error_dag_treshold) {
							std::cout << "残差と説明変数が独立ではないため間違ったDAGのためキャンセル" << std::endl;
							edge_dag_error_flg = true;
						}

						//printf("Evaluation of independence\n");
						//Evaluation of independence
						//calc_mutual_information(residual_error, residual_error_independ, bins);
						calc_mutual_information(residual_error, residual_error_independ, bins/*, true*/);

						for (int ii = 0; ii < exper_tmp.size(); ii++)
						{
							printf("%s\n", exper_tmp[ii].c_str());
						}
					}

					//if (edge_dag_error_flg) continue;
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

				if (!nonlinear)
				{
					if (abs_residual_errormax < 0)
					{
						abs_residual_errormax = abs_residual_errormax_cur;
					}
					if (abs_residual_errormax < 1.0e-10)
					{
						abs_residual_errormax = 1.0e-10;
					}
				}
				if (nonlinear)
				{
					if (0)
					{
						Matrix<double> mse_median(1, X.n);

						for (int i = 0; i < X.n; i++)
						{
							double max = Abs(residual_error.Col(i)).Max();
							double median = Median(Abs(residual_error).Col(i));
							double s = (max - median) / (0.5 * X.m);

							double sum = 0;
							int n = 0;
							for (int j = 0; j < X.m; j++)
							{
								if (fabs(residual_error.Col(i)(j, 0)) < max - s)
								{
									sum += pow(residual_error.Col(i)(j, 0), 2.0);
									n++;
								}
							}
							mse_median(0, i) = sum / (double)n;
						}
						mse_median(0, 0) = 0.0;
						mse_median.print("mse_median");
						printf("mse_median max:%f\n", mse_median.Max());

						abs_residual_errormax_cur = mse_median.Max();
					}
					if (0)
					{
						Matrix<double> mae(1, X.n);
						for (int i = 0; i < X.n; i++)
						{
							mae(0, i) = Abs(residual_error.Col(i)).Sum() / (double)X.m;
						}
						mae(0, 0) = 0.0;
						mae.print("mae");
						printf("mae max:%f\n", mae.Max());

						abs_residual_errormax_cur = mae.Max();
					}
					if (10)
					{
						Matrix<double> mse(1, X.n);
						for (int i = 0; i < X.n; i++)
						{
							//mse(0, i) = ((Pow(residual_error.Col(i), 2.0) / (double)X.m).Sum());

							double sum = 0.0;
							for (int j = train_size; j < X.m; j++)
							{
								sum += pow(residual_error(j, i), 2.0);
							}
							if (X.m == train_size)
							{
								mse(0, i) = sum / X.m;
							}
							else
							{
								mse(0, i) = sum / (X.m - train_size);
							}
						}
						mse(0, 0) = 0.0;
						mse.print("mse");
						printf("mse max:%f\n", mse.Max());

						//Matrix<double> mse_(1, X.n);
						//for (int i = 0; i < X.n; i++)
						//{
						//	double sum = 0.0;
						//	for (int j = 0; j < X.m; j++)
						//	{
						//		sum += pow(residual_error(j, i), 2.0);
						//	}
						//	mse_(0, i) = sum / (double)X.m;
						//}
						//mse_(0, 0) = 0.0;
						//mse_.print("mse_");
						//printf("mse_ max:%f\n", mse_.Max());

						abs_residual_errormax_cur = mse.Max();
					}

					if (0)
					{
						Matrix<double> r2(1, X.n);
						for (int i = 0; i < X.n; i++)
						{
							r2(0, i) = Pow(residual_error.Col(i), 2.0).Sum() /
								(Pow(X.Col(i) - X.Col(i).mean(), 2.0).Sum() + 1.0e-14);
						}

						r2(0, 0) = 0.0;
						r2.print("r2");
						printf("r2 max:%f\n", r2.Max());

						abs_residual_errormax_cur = r2.Max();
					}

					if (0)
					{
						Matrix<double> mer(1, X.n);
						Matrix<double> mer_(X.m, X.n);
						for (int i = 0; i < X.n; i++)
						{
							for (int j = 0; j < X.m; j++)
							{
								mer_(j, i) = fabs(residual_error(j, i)) / (X(j, i) + 1.0e-14);
							}
							mer(0, i) = Median(mer_.Col(i));
							if (i == 0)mer(0, i) = 0.0;
						}
						mer.print("mer");
						printf("mer max:%f\n", mer.Max());

						abs_residual_errormax_cur = mer.Max();
					}

					if (abs_residual_errormax < 0)
					{
						abs_residual_errormax = abs_residual_errormax_cur;
					}
					if (abs_residual_errormax < 1.0e-10)
					{
						abs_residual_errormax = 1.0e-10;
					}
				}

				double  abs_independ_max_cur = max(error_dag_max ,residual_error_independ.Max());
				//double  abs_independ_max_cur = residual_error_independ.Max();

				//if (nonlinear)
				//{
				//	abs_independ_max_cur = error_dag;
				//}

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

				double independ = max(error_dag, residual_error_independ.Max()) / abs_independ_max;
				//double independ = residual_error_independ.Max() / abs_independ_max;
				//if (nonlinear)
				//{
				//	independ = error_dag;
				//}
				double residual = abs_residual_errormax_cur / abs_residual_errormax;

				if (10)
				{
					char buf[256];
					sprintf(buf, "abs_residual_errormax:%f residual:%f (best:%f) independ:%f (best:%f)\n", abs_residual_errormax, residual, best_residual, independ, best_independ);

					DWORD  bytesWritten = 0;
					SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
					WriteFile(hStdout, buf, strlen(buf), &bytesWritten, NULL);
					SetConsoleTextAttribute(hStdout, csbi.wAttributes);
					WriteFile(hStdout, "\n", 1, &bytesWritten, NULL);
				}
				else
				{
					printf("abs_residual_errormax:%f residual:%f independ:%f\n", abs_residual_errormax, residual, independ);
				}
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

				if (nonlinear)
				{
					weight1 = 0.90;
					weight2 = 0.45;
					double w_tmp = weight1 + weight2;
					weight1 = weight1 / w_tmp;
					weight2 = weight2 / w_tmp;
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

				//if (nonlinear)
				//{
				//	if (best_residual > residual)
				//	{
				//		accept_ = true;
				//	}
				//}

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
							printf(buf);
							printf("\n");
						}
					}
					if (alp < th)
					{
						accept_ = true;
					}
				}

				if (!cond) accept_ = false;
				if (independ < 1.0e-4)  accept_ = false;

				//if (edge_dag_error_flg) accept_ = false;

				bool best_update = false;
				if (accept_)
				{
					accept_pattern_idx = current_pattern_idx;
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

					printf("best_min_value:%f value:%f\n", best_min_value, value);
					printf("best_residual:%f residual:%f\n", best_residual, residual);
					printf("best_independ:%f independ:%f\n", best_independ, independ);
					//printf("+\n");
					if (best_min_value >= value || (best_residual > residual || best_independ > independ))
					{
						double d1 = (residual - best_residual);
						double d2 = (independ - best_independ);
						if (best_residual > residual && best_independ < independ || best_residual < residual && best_independ > independ)
						{
							printf("d1:%f + d2:%f -> %f < %f\n", d1, d2, fabs(d1) + fabs(d2), 0.35 * (1.0 - rt));
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

					printf("loss_function:%d\n", loss_function);
					if (loss_function == 0)
					{
						if (best_min_value < value)
						{
							best_update = false;
						}
					}

					//if (nonlinear)
					//{
					//	if (best_residual < residual)
					//	{
					//		best_update = false;
					//	}
					//}

					if (best_update || kk == 0)
					{
						best_exper = exper_tmp;
						best_min_value = value;
						best_residual = residual;
						best_independ = independ;
						loss_value = value;
						for (int ii = 0; ii < best_exper.size(); ii++)
						{
							printf("%s\n", best_exper[ii].c_str());
						}
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

						observed = observed_tmp;
						predict = predict_tmp;
						colnames_id = name_id;
						hidden_colnames_id = hidden_name_id;

						intercept_best = intercept;
						dist_t_param_best = dist_t_param;

						exper = best_exper;
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
						{
							FILE* fp = fopen("lingam.model.update", "w");
							fprintf(fp, "%d/%d", kk, confounding_factors_sampling);
							fclose(fp);
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
					accept_pattern_idx = -1;
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
				if (pt == 0)
				{
					fput_loss(loss, best_residual, best_independ, best_min_value);
				}
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
					printf("[%d/%d]Total elapsed time:%.3f[sec] Estimated end time:%.3f[sec] reject:%d\n", kk, confounding_factors_sampling - 1, elapsed_ave * 0.001, t1, accept);
					fflush(stdout);
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

	Matrix<dnn_double> get_b_importance()
	{

		struct imp_ {
			int pa_id = -1;
			int id = -1;
			char name[32];
			double value = 0.0;
			char pa[32];

			bool operator<(const struct imp_& right) const {
				return fabs(value) >= fabs(right.value);
			}
		};
		FILE* fp = fopen("b_importance.txt", "r");

		printf("b_importance.txt:%p\n", (void*)fp);
		if (fp == NULL) return Matrix<dnn_double>();;

		std::vector< std::vector<struct imp_>> b_importance;

		char buf[1024];
		while (fgets(buf, 1024, fp) != NULL)
		{
			if (strstr(buf, "variable mean_dropout_loss") == NULL) continue;
			while (fgets(buf, 1024, fp) != NULL)
			{
				if (strstr(buf, "_full_model_") != NULL) continue;
				if (strstr(buf, "(Intercept)") != NULL) break;
				break;
			}

			std::vector<struct imp_> im_;
			while (fgets(buf, 1024, fp) != NULL)
			{
				int dmy;
				struct imp_ imp;
				if (strstr(buf, "_baseline_") != NULL) break;
				sscanf(buf, "%d %s %lf %s", &dmy, imp.name, &imp.value, imp.pa);

				//printf("@@@%d\n", imp.id);
				//printf("@@@[%s]\n", imp.name);
				//printf("@@@%f\n", imp.value);
				//printf("@@@[%s]\n", imp.pa);
				//fflush(stdout);

				char* p = imp.pa;
				p = strstr(p, "-[");
				if ( p != NULL ) *p = '\0';

				im_.push_back(imp);
			}

#if 0
			double max_v = 0.0;
			for (int kk = 0; kk < im_.size(); kk++)
			{
				if (max_v < im_[kk].value) max_v = im_[kk].value;
			}
			for (int kk = 0; kk < im_.size(); kk++)
			{
				im_[kk].value = im_[kk].value/ max_v;
			}
#endif
			std::sort(im_.begin(), im_.end());

			b_importance.push_back(im_);
		}
		fclose(fp);


		for (int i = 0; i < b_importance.size(); i++)
		{
			for (int ii = 0; ii < this->colnames.size(); ii++)
			{
				if (std::string(b_importance[i][0].pa) == this->colnames[ii])
				{
					b_importance[i][0].pa_id = ii;
					break;
				}
				if (std::string("\"")+std::string(b_importance[i][0].pa)+ std::string("\"") == this->colnames[ii])
				{
					b_importance[i][0].pa_id = ii;
					break;
				}
			}
			for (int j = 0; j < b_importance[i].size(); j++)
			{
				for (int ii = 0; ii < this->colnames.size(); ii++)
				{
					b_importance[i][j].pa_id = b_importance[i][0].pa_id;
					if (std::string(b_importance[i][j].name) == this->colnames[ii])
					{
						b_importance[i][j].id = ii;
						break;
					}
					if (std::string("\"") + std::string(b_importance[i][j].name) + std::string("\"") == this->colnames[ii])
					{
						b_importance[i][j].id = ii;
						break;
					}
				}
			}
		}
		printf("b_importance:%d\n", b_importance.size());
		for (int i = 0; i < b_importance.size(); i++)
		{
			printf("[%s(%d)]\n", b_importance[i][0].pa, b_importance[i][0].pa_id);
			for (int j = 0; j < b_importance[i].size(); j++)
			{
				printf("\t[%s(%d)]:%f\n", b_importance[i][j].name, b_importance[i][j].id, b_importance[i][j].value);
			}
		}

		////printf("\n------");
		//for (int k = 0; k < this->replacement.size(); k++)
		//{
		//	printf("%d ", this->replacement[k]);
		//}
		//printf("\n-----");

		Matrix<dnn_double> impB = Matrix<dnn_double>().values(B.n, B.n, 0.0);
		for (int i = 0; i < b_importance.size(); i++)
		{
			for (int j = 0; j < b_importance[i].size(); j++)
			{
				if (b_importance[i][0].pa_id >= B.n ) continue;
				if (b_importance[i][j].id == -1 ) continue;

				//if (fabs(B(ii, jj)) > 0.0)
				{
					impB(b_importance[i][0].pa_id, b_importance[i][j].id) = b_importance[i][j].value;
				}
				//printf("(%d,%d)%f(%f) ", ii, jj, impB(ii, jj), b_importance[i][j].value);
			}
			//printf("\n");
		}
		(B).print("B");
		//(impB).print("impB");
		(this->inv_before_sorting_(impB)).print("impB_");
		//(this->before_sorting_(impB)).print("impB__");

		for (int i = 0; i < B.n; i++)
		{
			double max = 0;
			for (int j = 0; j < B.n; j++)
			{
				if (fabs(impB(i, j)) > max) max = fabs(impB(i, j));
			}
			if (fabs(max) > 1.0e-10)
			{
				for (int j = 0; j < B.n; j++)
				{
					impB(i, j) = impB(i, j) / max;
				}
			}
		}

		impB = this->inv_before_sorting_(impB);
		return impB;
	}

	Matrix<dnn_double> importance()
	{
		printf("importance\n"); fflush(stdout);
		string cmd = "";
		cmd += "library(Ckmeans.1d.dp)\n";
		cmd += "options(width=10000)\n";
		cmd += "require(xgboost)\n";
		cmd += "require(Matrix)\n";
		cmd += "require(DALEX)\n";
		cmd += "require(DALEXtra)\n";
		cmd += "require(ingredients)\n";
		cmd += "require(mlr)\n";
		cmd += "library(SHAPforxgboost)\n";
		cmd += "library(gridExtra)\n";
		cmd += "\n";
		cmd += "\n";
		cmd += "previous_na_action <- options()$na.action\n";
		cmd += "options(na.action='na.pass')\n";
		cmd += "\n";
		cmd += "df <- read.csv( \"lingam.model.input_sample.csv\", header=T, stringsAsFactors = F, na.strings = c(\"\", \"NA\"))\n";
		cmd += "mu <- read.csv( \"lingam.model.mu.csv\", header=F, stringsAsFactors = F, na.strings = c(\"\", \"NA\"))\n";
		cmd += "colnames(mu)<- c(\""+ std::string("Unknown1") + "\"";

		for (int i = 1; i < this->B.n; i++)
		{
			char n[80];
			sprintf(n, "Unknown%d", i + 1);
			cmd += ",\"" + std::string(n) + "\"";
		}
		cmd += ")\n";
		cmd += "df <- cbind(df, mu)\n";
		cmd += "\n";
		cmd += "train <- df[1:min(3000,nrow(df)),]\n";
		cmd += "\n";

		cmd += "sink(\"b_importance.txt\")\n";
		int plot = 1;

		printf("colnames_id:%d\n", colnames_id.size());
		for (int i = 0; i < colnames_id.size(); i++)
		{
			char n[4096];
			if (this->colnames_id[i].size() < 2) continue;

			sprintf(n, "y_ <- train$'%s'\n", std::regex_replace(this->colnames[this->colnames_id[i][0]], regex("\""), "").c_str());
			cmd += std::string(n) + "\n";
			cmd += "train_mx<-sparse.model.matrix(y_ ~";

			for (int j = 1; j < this->colnames_id[i].size(); j++)
			{
				sprintf(n, "%s", std::regex_replace(this->colnames[this->colnames_id[i][j]], regex("\""), "").c_str());
				cmd += std::string(n);
				if (j < this->colnames_id[i].size() - 1)
				{
					cmd += "+";
				}
			}
			if (hidden_colnames_id[i].size() > 0) cmd += "+";
			for (int j = 0; j < hidden_colnames_id[i].size(); j++)
			{
				char n[80];
				
				sprintf(n, " Unknown%d", hidden_colnames_id[i][j]+1);
				if ( j < hidden_colnames_id[i].size()-1) sprintf(n, " Unknown%d +", hidden_colnames_id[i][j]+1);
				cmd += std::string(n);
			}

			cmd += ", data = train)\n";
			cmd += "train_dmat <- xgb.DMatrix(train_mx, label = y_)\n";
			cmd += "\n";
			cmd += "l_params= list(booster=\"gbtree\"\n";
			cmd += ",objective=\"reg:squarederror\"\n";
			cmd += ",eta=0.1\n";
			cmd += ",gamma=0.0\n";
			cmd += ",min_child_weight=1.0\n";
			cmd += ",subsample=1\n";
			cmd += ",max_depth=6\n";
			cmd += ",alpha=0.0\n";
			cmd += ",lambda=1.0\n";
			cmd += ",colsample_bytree=0.8\n";
			cmd += ",nthread=3\n";
			cmd += ",tree_method = 'hist'\n";
			cmd += ",predictor='cpu_predictor'\n";
			cmd += ")\n";
			cmd += "\n";
			cmd += "\n";
			cmd += "\n";
			cmd += "options(na.action=previous_na_action)\n";
			cmd += "\n";
			cmd += "set.seed(1) \n";
			cmd += "xgbmodel <- xgb.train(data = train_dmat,nrounds = 200,verbose = 2,\n";
			cmd += ",early_stopping_rounds = 100,params = l_params, watchlist = list(train = train_dmat, eval = train_dmat))\n";
			cmd += "\n";
			cmd += "explainer <-explain_xgboost(xgbmodel, data = train_mx, y_, label = \"Contribution of each variable\", type = \"regression\")\n";
			cmd += "imp_<-feature_importance(explainer, label=\""+ std::regex_replace(this->colnames[this->colnames_id[i][0]], regex("\""), "") + "-[contribution]\",loss_function = DALEX::loss_root_mean_square)\n";
			cmd += "#shap_values <- shap.values(xgb_model = xgbmodel, X_train = train_dmat)\n";
			cmd += "#plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score,top_n = 6, n_groups = 1)\n";
			cmd += "#shap.plot.force_plot(plot_data, zoom_in_location=1)\n";
			cmd += "#train_force_plot_plt <- shap.plot.force_plot_bygroup(plot_data)\n";
			cmd += "print(imp_)\n";
			cmd += "\n";

			sprintf(n, "plt%d<-plot(imp_)\n", plot);
			cmd += std::string(n);
			plot++;
		}
		cmd += "sink()\n";

		if (plot >= 2)
		{
			cmd += "g <- grid.arrange(plt1,";
			for (int i = 2; i < plot; i++)
			{
				char n[128];
				sprintf(n, "plt%d,", i);
				cmd += std::string(n);
			}
			char n[256];
			sprintf(n, " nrow =%d)\n", plot);
			cmd += std::string(n);

			sprintf(n, "ggplot2::ggsave(\"b_importance.png\", g, width = 4.8 * 4 * 1.00, height = 6.4 * %d, units = \"cm\", dpi = 100, limitsize = F)", plot);
			cmd += std::string(n);
		}

		FILE* fp = fopen("b_importance.r", "w");
		utf8str utf8;
		utf8.fprintf(fp, "%s\n",cmd.c_str());
		fclose(fp);
		printf("importance end\n"); fflush(stdout);

		if (this->eval_mode)
		{
			Matrix<dnn_double> m;
			printf("R_cmd_path:[%s]\n", R_cmd_path.c_str());
			if (R_cmd_path != "")
			{
				std::string cmd = R_cmd_path;
				cmd += " CMD BATCH --slave --vanilla  b_importance.r";

				printf("[%s]\n", cmd.c_str());
				system(cmd.c_str());

				try
				{
					m = get_b_importance();
				}
				catch (...)
				{
					printf("get_b_importance error.");
					return Matrix<dnn_double>();
				}
			}
			return m;
		}
		return Matrix<dnn_double>();
	}


	void fit_state(std::vector<std::vector<int>>& name_id, std::vector<std::vector<double>>& predict_y, std::vector<std::vector<double>>& observed_y)
	{
		FILE* fp = fopen("fit.r", "w");
		utf8str utf8;
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");
		utf8.fprintf(fp, "n <- %d\n", predict_y[0].size());

		int nn = predict_y[0].size() / 100;
		if (nn < 1) nn = predict_y[0].size();

		int num = 0;
		for (int j = 0; j < observed_y[0].size(); j++)
		{
			if (j % nn == 0) num++;
		}
		utf8.fprintf(fp, "n <- %d\n", num);

		for (int i = 0; i < predict_y.size(); i++)
		{
			utf8.fprintf(fp, "df%d<- data.frame(", name_id[i][0]);
			utf8.fprintf(fp, "x=c(1:n), %s_obs=c(%.3f", std::regex_replace(this->colnames [name_id[i][0]], regex("\""), "").c_str(), observed_y[i][0]);
			for (int j = 1; j < observed_y[i].size(); j++)
			{
				if ( j % nn == 0 ) utf8.fprintf(fp, ",%.3f", observed_y[i][j]);
			}
			utf8.fprintf(fp, "), %s_fit=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), predict_y[i][0]);
			for (int j = 1; j < predict_y[i].size(); j++)
			{
				if (j % nn == 0) utf8.fprintf(fp, ",%.3f", predict_y[i][j]);
			}
			utf8.fprintf(fp, "))\n");
			utf8.fprintf(fp, "g%d <- ggplot(df%d)\n", i, name_id[i][0]);
			utf8.fprintf(fp, "g%d <- g%d + geom_line(aes(x = x, y = %s_fit, colour =\"%s_fit\"))\n", i, i, std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
			utf8.fprintf(fp, "g%d <- g%d + geom_line(aes(x = x, y = %s_obs, colour =\"%s\"))\n", i, i, std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
		}
		utf8.fprintf(fp, "g <- grid.arrange(g%d", 0);
		for (int i = 1; i < predict_y.size(); i++)
		{
			utf8.fprintf(fp, ",g%d", i);
		}
		utf8.fprintf(fp, ", nrow = %d)\n", predict_y.size());
		utf8.fprintf(fp, "ggplot2::ggsave(\"fit.png\",g,width = 6.4*3.00, height = 4.8*%d, units = \"cm\", dpi = 100, limitsize=F)\n", predict_y.size());
		fclose(fp);
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

		utf8str utf8;
		int plot = 1;
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");
		utf8.fprintf(fp, "library(RColorBrewer)\n");
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

			utf8.fprintf(fp, "x_ <- data.frame(\n");
			int s = 0;

			utf8.fprintf(fp, "%s=c(\n", header_names2[i].c_str());

			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)utf8.fprintf(fp, ",");
				utf8.fprintf(fp, "\"%s\"", header_names2[j].c_str());
				s++;
			}
			utf8.fprintf(fp, "),\n");

			s = 0;
			utf8.fprintf(fp, "effect=c(\n");
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)utf8.fprintf(fp, ",");
				utf8.fprintf(fp, "%.2f", B(i, j));
				s++;
			}
			utf8.fprintf(fp, ")\n");
			utf8.fprintf(fp, ")\n");

			utf8.fprintf(fp, "g%d <- ggplot(x_, aes(x = %s, y = effect, fill = effect))\n", plot, header_names2[i].c_str());
			utf8.fprintf(fp, "g%d <- g%d + geom_bar(stat = \"identity\")\n", plot, plot);
			utf8.fprintf(fp, "g%d <- g%d + labs(title = \"Causal_effect\")\n", plot, plot);
			//utf8.fprintf(fp, "g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'YlOrRd\')))\n", plot, plot);
			utf8.fprintf(fp, "g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'Spectral\')))\n", plot, plot);
			utf8.fprintf(fp, "g%d <- g%d + theme(text = element_text(size = 12))\n", plot, plot);
			utf8.fprintf(fp, "#plot(g%d)\n", plot);
			plot++;
		}

		utf8.fprintf(fp, "g <- grid.arrange(");
		for (int i = 1; i < plot; i++)
		{
			utf8.fprintf(fp, "g%d", i);
			if (i < plot - 1) utf8.fprintf(fp, ",");
		}
		utf8.fprintf(fp, ", nrow = %d)\n", plot);
		utf8.fprintf(fp, "ggplot2::ggsave(\"Causal_effect.png\",g,width = 6.4*2*%.2f, height = 4.8*%.2f, units = \"cm\", dpi = 100, limitsize=F)\n", 1.0, (double)plot);

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

		utf8str utf8;
		printf("b_probability_barplot\n");
		FILE* fp = fopen("b_probability_barplot.r", "w");
		if (fp == NULL)
		{
			return;
		}

		int plot = 1;
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");
		utf8.fprintf(fp, "library(RColorBrewer)\n");
		for (int i = 0; i < variableNum; i++)
		{

			int count = 0;
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.001) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				count++;
			}
			if (count < 1) continue;

			utf8.fprintf(fp, "x_ <- data.frame(\n");
			int s = 0;

			utf8.fprintf(fp, "%s=c(\n", header_names2[i].c_str());

			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.001) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				utf8.fprintf(fp, "\"%s\"", header_names2[j].c_str());
				s++;
			}
			utf8.fprintf(fp, "),\n");

			s = 0;
			utf8.fprintf(fp, "probability=c(\n");
			for (int j = 0; j < variableNum; j++)
			{
				if (i == j) continue;
				if (b_probability(i, j) < 0.001) continue;
				if (fabs(B(i, j)) < 0.001) continue;

				if (s > 0)fprintf(fp, ",");
				utf8.fprintf(fp, "%.2f", b_probability(i, j) * 100.0);
				s++;
			}
			utf8.fprintf(fp, ")\n");
			utf8.fprintf(fp, ")\n");

			utf8.fprintf(fp, "g%d <- ggplot(x_, aes(x = %s, y = probability, fill = probability))\n", plot, header_names2[i].c_str());
			utf8.fprintf(fp, "g%d <- g%d + geom_bar(stat = \"identity\")\n", plot, plot);
			utf8.fprintf(fp, "g%d <- g%d + labs(title = \"probability\")\n", plot, plot);
			utf8.fprintf(fp, "g%d <- g%d + theme(text = element_text(size = 12))\n", plot, plot);
			utf8.fprintf(fp, "#g%d <- g%d + scale_fill_gradientn( colours = rev( brewer.pal( 7, \'Blues\')))\n", plot, plot);
			utf8.fprintf(fp, "#plot(g%d)\n", plot);
			plot++;
		}

		utf8.fprintf(fp, "g <- grid.arrange(");
		for (int i = 1; i < plot; i++)
		{
			utf8.fprintf(fp, "g%d", i);
			if (i < plot - 1) utf8.fprintf(fp, ",");
		}
		utf8.fprintf(fp, ", nrow = %d)\n", plot);
		utf8.fprintf(fp, "ggplot2::ggsave(\"b_probability.png\",g,width = 6.4*2*%.2f, height = 4.8*%.2f, units = \"cm\", dpi = 100, limitsize=F)\n", 1.0, (double)plot);

		fclose(fp);
		printf("b_probability_barplot end\n");
	}

	void scatter(std::vector<std::vector<int>>& name_id, std::vector<std::vector<double>>& predict_y, std::vector<std::vector<double>>& observed_y)
	{
		utf8str utf8;
		FILE* fp = fopen("scatter.r", "w");
		if (fp == NULL)
		{
			return;
		}
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");

		input_sample.print("input_sample");
		int count = 0;
		for (int i = 0; i < name_id.size(); i++)
		{
			for (int k = 1; k < name_id[i].size(); k++)
			{
				utf8.fprintf(fp, "df%d_%d<- data.frame(", name_id[i][0], name_id[i][k]);
				utf8.fprintf(fp, "%s=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), input_sample(0, name_id[i][0]));
				
				int n = input.m / 300;
				if (n <= 0) n = input_sample.m;
				for (int j = 1; j < input_sample.m; j++)
				{
					if ( j % n == 0 ) utf8.fprintf(fp, ",%.3f", input_sample(j, name_id[i][0]));
				}
				utf8.fprintf(fp, ")");
				utf8.fprintf(fp, ",%s=c(%.3f", std::regex_replace(this->colnames[name_id[i][k]], regex("\""), "").c_str(), input_sample(0, name_id[i][k]));
				for (int j = 1; j < input_sample.m; j++)
				{
					if (j % n == 0) utf8.fprintf(fp, ",%.3f", input_sample(j, name_id[i][k]));
				}
				utf8.fprintf(fp, ")");
				utf8.fprintf(fp, ", %s_fit=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), predict_y[i][0]);
				for (int j = 1; j < predict_y[i].size(); j++)
				{
					if (j % n == 0) utf8.fprintf(fp, ",%.3f", predict_y[i][j]);
				}
				utf8.fprintf(fp, "))\n");

				utf8.fprintf(fp, "g%d <- ggplot(df%d_%d,", count, name_id[i][0], name_id[i][k]);
				utf8.fprintf(fp, " aes(x=%s, y=%s))\n", std::regex_replace(this->colnames[name_id[i][k]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
				utf8.fprintf(fp, "g%d <- g%d + geom_point(alpha=0.4)\n", count, count);
				//if (!use_pnl)
				{
					utf8.fprintf(fp, "g%d <- g%d + geom_point(aes(x=%s, y=%s_fit), color =\"#FF4B00\", size=2, alpha=0.7)\n", count, count, std::regex_replace(this->colnames[name_id[i][k]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
				}
				utf8.fprintf(fp, "#g%d <- g%d + scale_color_brewer(palette = \"Set2\")\n", count, count);
				count++;
			}
		}
		if (count >= 1)
		{
			if (count == 1)
			{
				utf8.fprintf(fp, "g <- g0\n");
			}
			else
			{
				utf8.fprintf(fp, "g <- grid.arrange(g%d", 0);
				for (int i = 1; i < count; i++)
				{
					utf8.fprintf(fp, ",g%d", i);
				}
				utf8.fprintf(fp, ", nrow = %d)\n", count);
			}
			utf8.fprintf(fp, "ggplot2::ggsave(\"scatter.png\",g,width = 6.4, height = 4.8*%d, units = \"cm\", dpi = 100, limitsize=F)\n", count);
		}
		fclose(fp);
	}

	void scatter2(std::vector<std::vector<int>>& name_id, std::vector<std::vector<double>>& predict_y, std::vector<std::vector<double>>& observed_y)
	{
		utf8str utf8;
		FILE* fp = fopen("scatter2.r", "w");
		if (fp == NULL)
		{
			return;
		}
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");

		input_sample.print("input_sample");
		int count = 0;
		for (int i = 0; i < name_id.size(); i++)
		{
			if (name_id[i].size() == 1) continue;
			utf8.fprintf(fp, "df%d<- data.frame(", name_id[i][0]);

			int n = input_sample.m / 300;
			if (n <= 0) n = input_sample.m;
			utf8.fprintf(fp, "%s_fit=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), predict_y[i][0]);
			for (int j = 1; j < predict_y[i].size(); j++)
			{
				if (j % n == 0) utf8.fprintf(fp, ",%.3f", predict_y[i][j]);
			}
			utf8.fprintf(fp, "),\n");
			utf8.fprintf(fp, "%s_obs=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), observed_y[i][0]);
			for (int j = 1; j < observed_y[i].size(); j++)
			{
				if (j % n == 0) utf8.fprintf(fp, ",%.3f", observed_y[i][j]);
			}
			utf8.fprintf(fp, "))\n");

			utf8.fprintf(fp, "g%d <- ggplot(df%d)\n", count, name_id[i][0]);
			utf8.fprintf(fp, "g%d <- g%d + geom_point(aes(x=%s_obs, y=%s_fit), color =\"#FF4B00\", size=2, alpha=0.7)\n", count, count, std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
			//if (!use_pnl)
			{
				utf8.fprintf(fp, "g%d <- g%d + geom_line(aes(x=%s_obs, y=%s_obs), color =\"#005AFF\", size=2, alpha=0.7)\n", count, count, std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
			}
			count++;
		}
		if (count >= 1)
		{
			if (count == 1)
			{
				utf8.fprintf(fp, "g <- g0\n");
			}
			else
			{
				utf8.fprintf(fp, "g <- grid.arrange(g%d", 0);
				for (int i = 1; i < count; i++)
				{
					utf8.fprintf(fp, ",g%d", i);
				}
				utf8.fprintf(fp, ", nrow = %d)\n", count);
			}
			utf8.fprintf(fp, "ggplot2::ggsave(\"scatter2.png\",g,width = 6.4, height = 6.4*%d, units = \"cm\", dpi = 100, limitsize=F)\n", count);
		}
		fclose(fp);
	}

	void error_hist(std::vector<std::vector<int>>& name_id)
	{
		utf8str utf8;
		FILE* fp = fopen("error_hist.r", "w");
		if (fp == NULL)
		{
			return;
		}
		utf8.fprintf(fp, "library(ggplot2)\n");
		utf8.fprintf(fp, "library(gridExtra)\n");

		int count = 0;
		for (int i = 0; i < name_id.size(); i++)
		{
			utf8.fprintf(fp, "df%d<- data.frame(", name_id[i][0]);
			utf8.fprintf(fp, "%s=c(%.3f", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str(), residual_error(0, name_id[i][0]));

			int n = residual_error.m / 300;
			if (n <= 0) n = residual_error.m;
			for (int j = 1; j < residual_error.m; j++)
			{
				if (j % n == 0) utf8.fprintf(fp, ",%.3f", residual_error(j, name_id[i][0]));
			}
			utf8.fprintf(fp, "))\n");

			utf8.fprintf(fp, "g%d <- ggplot(df%d,", count, name_id[i][0]);
			utf8.fprintf(fp, " aes(x=%s))\n", std::regex_replace(this->colnames[name_id[i][0]], regex("\""), "").c_str());
			utf8.fprintf(fp, "g%d <- g%d + geom_histogram()\n", count, count);
			count++;
		}
		if (count >= 1)
		{
			if (count == 1)
			{
				utf8.fprintf(fp, "g <- g0\n");
			}
			else
			{
				utf8.fprintf(fp, "g <- grid.arrange(g%d", 0);
				for (int i = 1; i < count; i++)
				{
					utf8.fprintf(fp, ",g%d", i);
				}
				utf8.fprintf(fp, ", nrow = %d)\n", count);
			}
			utf8.fprintf(fp, "ggplot2::ggsave(\"err_histogram.png\",g,width = 6.4, height = 6.4*%d, units = \"cm\", dpi = 100, limitsize=F)\n", count);
		}
		fclose(fp);
	}

};
#endif
