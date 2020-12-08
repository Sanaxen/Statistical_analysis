#ifndef _NonLinearRegression_H

#define _NonLinearRegression_H

#include "../../include/util/mathutil.h"
#include "../../include/nonlinear/MatrixToTensor.h"
#include <signal.h>
#pragma warning( disable : 4305 ) 


#ifdef USE_LIBTORCH
#include "../pytorch_cpp/tiny_dnn2libtorch_dll.h"
#pragma comment(lib, "../../pytorch_cpp/lib/rnn6.lib")
#endif

#define EARLY_STOPPING_	10

/* シグナル受信/処理 */
bool ctr_c_stopping_nonlinear_regression = false;
inline void SigHandler_nonlinear_regression(int p_signame)
{
	static int sig_catch = 0;
	if (sig_catch)
	{
		printf("割り込みです。終了します\n");
		ctr_c_stopping_nonlinear_regression = true;
		fflush(stdout);
		exit(0);
	}
	sig_catch++;
	return;
}

inline void multiplot_gnuplot_script(int ydim, int step, std::vector<std::string>& names, std::vector<int>& idx, bool putImage, int confidence)
{
	if (!putImage)
	{
		for (int i = 0; 1; i++)
		{
			char fname[256];
			sprintf(fname, "multi_data%03d.plt", i);

			FILE* fp = fopen(fname, "r");
			if (fp == NULL) break;
			if (fp) fclose(fp);
			if (!putImage) remove(fname);

			sprintf(fname, "multi_data%03d_png.plt", i);
			fp = fopen(fname, "r");
			if (fp)
			{
				fclose(fp);
				if (putImage) remove(fname);
			}

			sprintf(fname, "multi_data%03d.png", i);
			fp = fopen(fname, "r");
			if (fp)
			{
				fclose(fp);
				if (putImage) remove(fname);
			}
		}
	}
	FILE* genImg = NULL;
	if (putImage)
	{
		genImg = fopen("gen_multi_data2png.bat", "w");
	}

	int count = 0;
	for (int i = 0; 1; i++)
	{
		char fname[256];
		if (!putImage)
		{
			sprintf(fname, "multi_data%03d.plt", i);
		}
		else
		{
			sprintf(fname, "multi_data%03d_png.plt", i);
		}

		FILE* fp = fopen(fname, "w");
		if (fp == NULL) return;

		fprintf(fp, "bind \"Close\" \"if (GPVAL_TERM eq \'wxt\') bind \'Close\' \'\'; exit gnuplot; else bind \'Close\' \'\'; exit\"\n");

		fprintf(fp, "set term windows size %d,%d font \"arial,10\"\n", (int)(640 * 2.8), (int)(380 * 1.5)*step / 2);
		if (putImage)
		{
			fprintf(fp, "set term pngcairo size %d,%d font \"arial,10\"\n", (int)(640 * 2.8), (int)(380 * 1.5)*step / 2);
			fprintf(fp, "set output \"multi_data%03d.png\"\n", i);
		}
		if (ydim <= 2) step = 2;
		if (ydim >= 2)
		{
			fprintf(fp, "set multiplot layout %d,%d\n", step, 1);
		}
		fprintf(fp, "set key left top box\n");
		if (confidence)
		{
			fprintf(fp, "set style fill transparent solid 0.4 noborder\n");
		}

		//char* timex =
		//	"x_timefromat=0\n"
		//	"if (x_timefromat != 0) set xdata time\n"
		//	"if (x_timefromat != 0) set timefmt \"%%Y/ %%m/ %%d[%%H:%%M:%%S]\"\n"
		//	"if (x_timefromat != 0) set xtics timedate\n"
		//	"if (x_timefromat != 0) set xtics format \"%%Y/%%m/%%d\"\n";
		//fprintf(fp, timex,  0);

		for (int k = 0; k < step; k++)
		{
			fprintf(fp, "set title %s\n", names[idx[count]].c_str());
			std::string  plot =
				"file = \"test.dat\"\n"
				"plot file using 1:%d   t \"observation\"  with lines linewidth 2 lc \"#0068b7\", \\\n"
				"file using 1:%d   t \"predict\"  with lines linewidth 2 lc \"#ff8000\"";

			if (confidence)
			{
				plot = plot + ",\\\n\"confidence.dat\" using 1:%d:%d  with filledcurves fc \"#5f9ea0\" t \"95%% confidence\"\n\n";
				fprintf(fp, plot.c_str(), count * 2 + 3, count * 2 + 2,
					count*ydim + 3, count*ydim + 4);
			}
			else
			{
				plot = plot + "\n\n";
				fprintf(fp, plot.c_str(), count * 2 + 3, count * 2 + 2);
			}
			count++;
			if (count >= ydim) break;
		}
		fprintf(fp, "unset multiplot\n");
		if (!putImage)
		{
			fprintf(fp, "pause -1\n");
		}
		fclose(fp);
		convf(fname);

		if (putImage)
		{
			gnuplot_path_::getGnuplotPath();
			if (gnuplot_path_::path_ != "")
			{
				std::string& gnuplot_exe_path = "\"" + gnuplot_path_::path_ + "\\gnuplot.exe\"";
				system((gnuplot_exe_path + " " + fname).c_str());
				//printf("%s\n", (gnuplot_exe_path + " " + fname).c_str());
				if (genImg) fprintf(genImg, "%s\n", (gnuplot_exe_path + " " + fname).c_str());
			}
		}

		if (count >= ydim) break;
	}
	if (genImg)
	{
		fclose(genImg);
		system("gen_multi_data2png.bat");
	}
}

class NonLinearRegression
{
	bool convergence = false;
	int error = 0;
	FILE* fp_accuracy = NULL;
	FILE* fp_error_loss = NULL;
	FILE* fp_error_vari_loss = NULL;
	bool visualize_state_flag = true;

	class writing
	{
	public:
		writing()
		{
			FILE* fp = fopen("Writing_NonLinearRegression_", "w");
			fclose(fp);
		}
		~writing()
		{
			unlink("Writing_NonLinearRegression_");
		}
	};

public:
	bool use_trained_scale = true;
	bool normalized = false;
	std::vector<std::string> header;
	std::vector<int> normalizeskipp;			// non normalize var
	double xx_var_scale = 1.0;			// non normalize var scaling
	std::vector<int> x_idx;
	std::vector<int> y_idx;
	bool fit_best_saved = false;
	bool batch_shuffle = true;

	std::string activation_fnc = "tanh";
	int n_sampling = 10;


private:
	void normalizeZ(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
	{
		if (!normalized)
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
					sigma[k] += (X[i][k] - mean[k])*(X[i][k] - mean[k]);
				}
			}
			for (int k = 0; k < X[0].size(); k++)
			{
				sigma[k] /= (X.size() - 1);
				sigma[k] = sqrt(sigma[k]);
			}
			if (normalizeskipp.size() > 0)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					//printf("[%d] %d\n", k, normalizeskipp[k]);
					if (normalizeskipp[k])
					{
						sigma[k] = xx_var_scale;
						mean[k] = 0.0;
					}
				}
			}
		}

		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - mean[k]) / (sigma[k] + 1.0e-10);
			}
		}
	}

	void normalizeMinMax(tiny_dnn::tensor_t& X, std::vector<float_t>& min_, std::vector<float_t>& maxmin_)
	{
		if (!normalized)
		{
			min_ = std::vector<float_t>(X[0].size(), 0.0);
			maxmin_ = std::vector<float_t>(X[0].size(), 1.0);

#if 0
			float max_value = -std::numeric_limits<float>::max();
			float min_value = std::numeric_limits<float>::max();
			for (int i = 0; i < X.size(); i++)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (max_value < X[i][k]) max_value = X[i][k];
					if (min_value > X[i][k]) min_value = X[i][k];
				}
			}
			for (int k = 0; k < X[0].size(); k++)
			{
				min_[k] = min_value;
				maxmin_[k] = (max_value - min_value);

				if (fabs(maxmin_[k]) < 1.0e-14)
				{
					min_[k] = 0.0;
					maxmin_[k] = max_value;
				}
			}

			for (int i = 0; i < X.size(); i++)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					X[i][k] = (X[i][k] - min_[k]) / maxmin_[k];
				}
			}
#else
			for (int k = 0; k < X[0].size(); k++)
			{
				float max_value = -std::numeric_limits<float>::max();
				float min_value = std::numeric_limits<float>::max();
				for (int i = 0; i < X.size(); i++)
				{
					if (max_value < X[i][k]) max_value = X[i][k];
					if (min_value > X[i][k]) min_value = X[i][k];
				}
				min_[k] = min_value;
				maxmin_[k] = (max_value - min_value);
				if (fabs(maxmin_[k]) < 1.0e-14)
				{
					min_[k] = 0.0;
					maxmin_[k] = max_value;
				}
			}
			if (normalizeskipp.size() > 0)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (normalizeskipp[k])
					{
						maxmin_[k] = 1.0 / xx_var_scale;
						min_[k] = 0.0;
					}
				}
			}
#endif
		}
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) / (maxmin_[k]+ 1.0e-10);
			}
		}
	}
	void normalize1_1(tiny_dnn::tensor_t& X, std::vector<float_t>& min_, std::vector<float_t>& maxmin_)
	{
		if (!normalized)
		{
			min_ = std::vector<float_t>(X[0].size(), 0.0);
			maxmin_ = std::vector<float_t>(X[0].size(), 1.0);

#if 0
			float max_value = -std::numeric_limits<float>::max();
			float min_value = std::numeric_limits<float>::max();
			for (int i = 0; i < X.size(); i++)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (max_value < X[i][k]) max_value = X[i][k];
					if (min_value > X[i][k]) min_value = X[i][k];
				}
			}
			for (int k = 0; k < X[0].size(); k++)
			{
				min_[k] = min_value;
				maxmin_[k] = (max_value - min_value);

				if (fabs(maxmin_[k]) < 1.0e-14)
				{
					min_[k] = 0.0;
					maxmin_[k] = max_value;
				}
			}

			for (int i = 0; i < X.size(); i++)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					X[i][k] = (X[i][k] - min_[k]) / maxmin_[k];
		}
	}
#else
			for (int k = 0; k < X[0].size(); k++)
			{
				float max_value = -std::numeric_limits<float>::max();
				float min_value = std::numeric_limits<float>::max();
				for (int i = 0; i < X.size(); i++)
				{
					if (max_value < X[i][k]) max_value = X[i][k];
					if (min_value > X[i][k]) min_value = X[i][k];
				}
				min_[k] = min_value;
				maxmin_[k] = (max_value - min_value);
				if (fabs(maxmin_[k]) < 1.0e-14)
				{
					min_[k] = 0.0;
					maxmin_[k] = max_value;
				}
			}
			if (normalizeskipp.size() > 0)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (normalizeskipp[k])
					{
						maxmin_[k] = 2.0*(1.0 / xx_var_scale);
						min_[k] = -1.0 * (1.0 / xx_var_scale);
					}
				}
			}

#endif
		}
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) * 2 / (maxmin_[k]+ 1.0e-10) - 1;
			}
		}
	}

	float cost_min = std::numeric_limits<float>::max();
	float cost_min0 = std::numeric_limits<float>::max();
	float cost_pre = 0;
	float accuracy_max = std::numeric_limits<float>::min();
	float accuracy_pre = 0;

	tiny_dnn::tensor_t st_mean;
	tiny_dnn::tensor_t st_sigma2;
	int n_samples_cnt = 0;

	size_t net_test_no_Improvement_count = 0;
	void net_test(bool sampling_only = false)
	{
		writing lock;

		tiny_dnn::network2<tiny_dnn::sequential> nn_test;

		void* torch_nn_test = NULL;
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			if (test_mode)
			{
				torch_nn_test = torch_load_new("fit_best.pt");
				fit_best_saved = true;
			}
			else
			{
				torch_nn_test = torch_load_new("tmp.pt");
			}
		}
		else
#endif
		{
			if (test_mode)
			{
				nn_test.load("fit_best.model");
			}
			else
			{
				nn_test.load("tmp.model");
			}
		}
		//printf("layers:%zd\n", nn_test.depth());

		set_test(nn_test, 1);

		FILE* fp_predict = NULL;
		if (!sampling_only)
		{
			if (test_mode)
			{
				fp_predict = fopen("predict_dnn.csv", "w");
				if (fp_predict)
				{
					for (int i = 0; i < nX[0].size(); i++)
					{
						fprintf(fp_predict, "%s,", header[x_idx[i]].c_str());
					}
					if (classification >= 2)
					{
						fprintf(fp_predict, "predict[%s],", header[y_idx[0]].c_str());
						fprintf(fp_predict, "probability\n");
					}
					else
					{
						for (int i = 0; i < nY[0].size() - 1; i++)
						{
							fprintf(fp_predict, "predict[%s], %s,dy,", header[y_idx[i]].c_str(), header[y_idx[i]].c_str());
						}
						fprintf(fp_predict, "predict[%s], %s, dy\n", header[y_idx[nY[0].size() - 1]].c_str(), header[y_idx[nY[0].size() - 1]].c_str());
					}
				}
			}
		}

		if (classification >= 2)
		{
			float loss_train = 0;
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				loss_train = torch_get_loss_nn(torch_nn_test, train_images, train_labels, n_eval_minibatch);
			}
			else
#endif
			{
				loss_train = get_loss(nn_test, train_images, train_labels);
			}
			if (loss_train < cost_min)
			{
				cost_min = loss_train;
			}
			if (cost_min < std::numeric_limits<float>::max())
			{
				if (cost_min0 == std::numeric_limits<float>::max())
				{
					cost_min0 = cost_min;
				}
				if (fp_error_loss)
				{
					fprintf(fp_error_loss, "%.10f %.10f %.4f\n", loss_train, cost_min, tolerance);
					fflush(fp_error_loss);
				}
			}
			float loss_test = 0;
			if (test_images.size() > 0)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					loss_test = torch_get_loss_nn(torch_nn_test, test_images, test_labels, n_eval_minibatch);
				}
				else
#endif
				{
					loss_test = get_loss(nn_test, test_images, test_labels);
				}
			}

			tiny_dnn::result train_result;
			tiny_dnn::result test_result;

#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				train_result = torch_get_accuracy_nn(torch_nn_test, train_images, train_labels, n_eval_minibatch);
				if (test_images.size() > 0)
				{
					test_result = torch_get_accuracy_nn(torch_nn_test, test_images, test_labels, n_eval_minibatch);
				}
			}
			else
#endif
			{
				train_result = get_accuracy(nn_test, train_images, train_labels);
				if (test_images.size() > 0)
				{
					test_result = get_accuracy(nn_test, test_images, test_labels);
				}

			}
			if (fp_accuracy)
			{
				fprintf(fp_accuracy, "%.4f %.4f\n", train_result.accuracy(), test_result.accuracy());
				fflush(fp_accuracy);
			}
			{
				FILE* fp = fopen("nonlinear_error_vari_loss.txt", "w");
				if (fp)
				{
					fprintf(fp, "best.model loss:%.2f\n", cost_min);
					fprintf(fp, "total loss:%.2f\n", loss_train);
					fprintf(fp, "validation loss:%.2f\n", loss_test);
					fprintf(fp, "accuracy:%.2f%%\n", train_result.accuracy());
					fprintf(fp, "validation accuracy:%.2f%%\n", test_result.accuracy());
					fclose(fp);
				}
			}

			if (accuracy_max < train_result.accuracy())
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_save("fit_best.pt");
					fit_best_saved = true;
				}
				else
#endif
				{
					nn_test.save("fit_best.model");
					nn_test.save("fit_best.model.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
				}
				accuracy_max = train_result.accuracy();
			}
			if (1.0 - accuracy_max*0.01 < tolerance)
			{
				printf("accuracy_max:%f%%\n", accuracy_max);
				convergence = true;
			}
			if (accuracy_pre <= train_result.accuracy() || fabs(accuracy_pre - train_result.accuracy())*0.01 < 1.0e-3)
			{
				net_test_no_Improvement_count++;
			}
			else
			{
				net_test_no_Improvement_count = 0;
			}
			if (early_stopping && net_test_no_Improvement_count == EARLY_STOPPING_)
			{
				early_stopp = true;
			}
			accuracy_pre = train_result.accuracy();

			std::vector< tiny_dnn::vec_t> y_predict_n;
#ifdef USE_LIBTORCH
			y_predict_n = torch_model_predict_batch(torch_nn_test, nX, n_eval_minibatch);
#endif
			if (fp_predict)
			{
				for (int i = 0; i < nX.size(); i++)
				{
					tiny_dnn::vec_t x = nX[i];
					tiny_dnn::vec_t y_predict;
#ifdef USE_LIBTORCH
					if (use_libtorch)
					{
						//y_predict = torch_model_predict(torch_nn_test, x);
						y_predict = y_predict_n[i];
					}
					else
#endif
					{
						y_predict = nn_test.predict(x);
					}
					if (fp_predict)
					{
						for (int k = 0; k < nX[0].size(); k++)
						{
							fprintf(fp_predict, "%.3f,", iX[i][k]);
						}

						int idx = -1;
						float_t y = -1;
						for (int k = 0; k < classification; k++)
						{
							if (y < y_predict[k])
							{
								y = y_predict[k];
								idx = k;
							}
						}
						fprintf(fp_predict, "%d,%.3f\n", idx, y);
					}
				}
				fclose(fp_predict);
			}
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_delete_load_model(torch_nn_test);
				torch_nn_test = NULL;
			}
#endif
			return;
		}

		//////////////////////////////////////////////////
		std::vector<tiny_dnn::vec_t> Y_predict(nX.size());

#ifdef USE_LIBTORCH
		Y_predict = torch_model_predict_batch(torch_nn_test, nX, n_eval_minibatch);
#endif

//#pragma omp parallel for
		for (int i = 0; i < nX.size(); i++)
		{
			tiny_dnn::vec_t& x = nX[i];
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				//Y_predict[i] = torch_model_predict(torch_nn_test, x);
			}
			else
#endif
			{
				Y_predict[i] = nn_test.predict(x);
			}

			for (int k = 0; k < Y_predict[i].size(); k++)
			{
				if (zscore_normalization)
				{
					Y_predict[i][k] = Y_predict[i][k] * Sigma_y[k] + Mean_y[k];
				}
				if (minmax_normalization)
				{
					Y_predict[i][k] = Y_predict[i][k] * MaxMin_y[k] + Min_y[k];
				}
				if (_11_normalization)
				{
					Y_predict[i][k] = 0.5*(Y_predict[i][k] + 1) * MaxMin_y[k] + Min_y[k];
				}
			}
		}

		//distribution_ts
		if (/*this->n_sampling > 0*/ sampling_only)
		{
			if (test_mode)
			{
				tiny_dnn::vec_t dmy1(Y_predict[0].size());
				tiny_dnn::vec_t dmy2(Y_predict[0].size());

				char buf[256];
				FILE* fp = fopen("distribution.txt", "r");
				if (fp)
				{
					fgets(buf, 256, fp);
					fgets(buf, 256, fp);
					for (int k = 0; k < Y_predict[0].size(); k++)
					{
						fgets(buf, 256, fp);
						double x, y;
						sscanf(buf, "%lf %lf", &x, &y);
						dmy1[k] = x;
						dmy2[k] = y;
					}
					fclose(fp);
				}

				fp = fopen("confidence.dat", "w");
				if (fp)
				{
					for (int j = 0; j < nX.size(); j++)
					{
						fprintf(fp, "%d", j);
						for (int k = 0; k < Y_predict[0].size(); k++)
						{
							fprintf(fp, " %f %f %f",
								Y_predict[j][k],
								Y_predict[j][k] - dmy2[k],
								Y_predict[j][k] + dmy2[k]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);
				}
			}
			if (!test_mode)
			{
				//printf("n_samples_cnt:%d\n", n_samples_cnt); fflush(stdout);
				if (n_samples_cnt == 0)
				{
					st_mean.resize(nX.size(), tiny_dnn::vec_t(Y_predict[0].size(), 0));
				}

				for (int i = 0; i < nX.size(); i++)
				{
					for (int k = 0; k < Y_predict[0].size(); k++)
					{
						st_mean[i][k] += Y_predict[i][k];
					}
				}

				if (n_samples_cnt > 1)
				{
					st_sigma2.resize(nX.size(), tiny_dnn::vec_t(Y_predict[0].size(), 0));
					for (int i = 0; i < nX.size(); i++)
					{
						for (int k = 0; k < Y_predict[0].size(); k++)
						{
							auto d = (train_labels[i][k] - st_mean[i][k] / n_samples_cnt);
							st_sigma2[i][k] += d*d / n_samples_cnt;
						}
					}

					FILE* fp = fopen("confidence.dat", "w");
					if (fp)
					{
						for (int j = 0; j < nX.size(); j++)
						{
							fprintf(fp, "%d", j);
							for (int k = 0; k < Y_predict[0].size(); k++)
							{
								fprintf(fp, " %f %f %f",
									Y_predict[j][k],
									Y_predict[j][k] - 1.96*sqrt(st_sigma2[j][k] / n_samples_cnt),
									Y_predict[j][k] + 1.96*sqrt(st_sigma2[j][k] / n_samples_cnt));
							}
							fprintf(fp, "\n");
						}
						fclose(fp);

						fp = fopen("distribution.txt", "w");
						if (fp)
						{
							fprintf(fp, "%d\n", n_samples_cnt);
							int sz = train_images.size() - 1;
							fprintf(fp, "%d\n", Y_predict[0].size());
							for (int k = 0; k < Y_predict[0].size(); k++)
							{
								fprintf(fp, "%f %f\n", st_mean[sz][k] / n_samples_cnt, 1.96*sqrt(st_sigma2[sz][k] / n_samples_cnt));
							}
							fclose(fp);
						}
					}
				}
				n_samples_cnt++;
			}
			if (sampling_only) return;
		}

		//////////////////////////////////////////////////

		char plotName[256];
		//sprintf(plotName, "test%04d.dat", plot_count);
		sprintf(plotName, "test.dat");
		FILE* fp_test = fopen(plotName, "w");


		Diff.clear();
		float cost = 0.0;
		float vari_cost = 0.0;
		float cost_tot = 0.0;
		if (fp_test)
		{
			for (int i = 0; i < nX.size(); i++)
			{
				std::vector<double> diff;
				tiny_dnn::vec_t x = nX[i];

				tiny_dnn::vec_t y_predict;
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					//y_predict = torch_model_predict(torch_nn_test, x);
					y_predict = Y_predict[i];
				}
				else
#endif
				{
					y_predict = nn_test.predict(x);
				}


				if (fp_predict)
				{
					for (int k = 0; k < nX[0].size(); k++)
					{
						fprintf(fp_predict, "%.3f,", iX[i][k]);
					}
				}

				tiny_dnn::vec_t& y = iY[i];
				fprintf(fp_test, "%d ", i);
				for (int k = 0; k < Y_predict[i].size()-1; k++)
				{
					float_t yy = Y_predict[i][k];
					fprintf(fp_test, "%f %f ", yy, y[k]);

					diff.push_back(y[k]);
					diff.push_back(yy);

					if (fp_predict)
					{
						fprintf(fp_predict, "%.3f,%.3f,%.3f", yy, y[k], yy-y[k]);
					}

					if (test_data_index[i] >= 0)
					{
						vari_cost += (yy- y[k])*(yy - y[k]);

					}
					else
					{
						cost += (yy - y[k])*(yy - y[k]);
					}
					cost_tot += (yy - y[k])*(yy - y[k]);
				}

				float_t yy = Y_predict[i][Y_predict[i].size() - 1];
				fprintf(fp_test, "%f %f\n", yy, y[y_predict.size() - 1]);
				diff.push_back(y[y_predict.size() - 1]);
				diff.push_back(yy);
				Diff.push_back(diff);

				if (fp_predict)
				{
					fprintf(fp_predict, "%.3f, %.3f, %.3f\n", yy, y[y_predict.size() - 1], yy- y[y_predict.size() - 1]);
				}

				if (test_data_index[i] >= 0)
				{
					vari_cost += (yy - y[y_predict.size() - 1])*(yy - y[y_predict.size() - 1]);
				}
				else
				{
					cost += (yy - y[y_predict.size() - 1])*(yy- y[y_predict.size() - 1]);
				}
				cost_tot += (yy - y[y_predict.size() - 1])*(yy - y[y_predict.size() - 1]);
			}
			fclose(fp_test);
		}
		if (fp_predict) fclose(fp_predict);

		sprintf(plotName, "predict.dat");
		fp_test = fopen(plotName, "w");
		/**/
		if (fp_test)
		{
			fclose(fp_test);
			fp_test = NULL;
		}
		/**/
		if (fp_test)
		{
			for (int i = train_images.size(); i < nY.size(); i++)
			{
				tiny_dnn::vec_t x = nX[i];
				tiny_dnn::vec_t y_predict = Y_predict[i];

				tiny_dnn::vec_t& y = iY[i];
				fprintf(fp_test, "%d ", i);
				for (int k = 0; k < y_predict.size() - 1; k++)
				{
					fprintf(fp_test, "%f %f ", y_predict[k], y[k]);
				}
				fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1], y[y_predict.size() - 1]);
			}
			fclose(fp_test);
		}
		cost /= train_images.size();
		vari_cost /= test_images.size();
		cost_tot /= iY.size();
		//printf("%f %f\n", cost_min, cost);
		if (cost_tot < cost_min)
		{
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_save_nn(torch_nn_test, "fit_best.pt");
				fit_best_saved = true;
			}
			else
#endif
			{
				nn_test.save("fit_best.model");
				nn_test.save("fit_best.model.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
				fit_best_saved = true;
			}
			cost_min = cost_tot;
		}
		if (cost_min < tolerance)
		{
			printf("cost_min:%f\n", cost_min);
			convergence = true;
		}
		if (cost_min < std::numeric_limits<float>::max())
		{
			if (cost_min0 == std::numeric_limits<float>::max())
			{
				cost_min0 = cost_min;
			}
			if (fp_error_loss)
			{
				fprintf(fp_error_loss, "%.10f %.10f %.4f\n", cost, cost_min, tolerance);
				fflush(fp_error_loss);
			}
			if (fp_error_vari_loss)
			{
				fprintf(fp_error_vari_loss, "%.10f\n", vari_cost);
				fflush(fp_error_vari_loss);
			}
		}
		if (cost_pre <= cost_tot || fabs(cost_pre - cost_tot) < 1.0e-3)
		{
			net_test_no_Improvement_count++;
		}
		else
		{
			net_test_no_Improvement_count = 0;
		}
		if (early_stopping && net_test_no_Improvement_count == EARLY_STOPPING_)
		{
			early_stopp = true;
		}
		cost_pre = cost_tot;

		{
			FILE* fp = fopen("nonlinear_error_vari_loss.txt", "w");
			if (fp)
			{
				fprintf(fp, "best.model loss:%f\n", cost_min);
				fprintf(fp, "total loss:%f\n", cost_tot);
				fprintf(fp, "validation loss:%f\n", vari_cost);
				fclose(fp);
			}
		}
		set_train(nn, 1, 0, default_backend_type);
		visualize_observed_predict();
#ifdef USE_LIBTORCH
		torch_delete_load_model(torch_nn_test);
#endif
	}

public:
	void gen_visualize_fit_state(bool sampling_only = false)
	{
		set_test(nn, 1);
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			torch_save("tmp.pt");
		}else
#endif
		{
			nn.save("tmp.model");
		}
		net_test(sampling_only);
		set_train(nn, 1, 0, default_backend_type);


#ifdef USE_GNUPLOT
		//if (capture)
		//{
		//	printf("capture\n");
		//	std::string plot = std::string(GNUPLOT_PATH);
		//	plot += " test_plot_capture1.plt";
		//	system(plot.c_str());

		//	char buf[256];
		//	sprintf(buf, "images\\test_%04d.png", plot_count);
		//	std::string cmd = "cmd.exe /c ";
		//	cmd += "copy images\\test.png " + std::string(buf);
		//	system(cmd.c_str());
		//	printf("%s\n", cmd.c_str());
		//}
		plot_count++;
#endif
	}
private:
	std::vector<int> test_data_index;

	//int load_count = 1;
	int epoch = 1;
	int plot_count = 0;
	int batch = 0;
	bool early_stopp = false;
public:
	std::string device_name = "cpu";
	int use_libtorch = 0;
	float class_minmax[2] = { 0,0 };
	float dropout = 0;
	int classification = -1;
	bool visualize_observed_predict_plot = false;
	std::string regression = "";
	std::vector < std::vector<double>> Diff;
	float_t fluctuation = 0.0;
	float_t dec_random = 0;	//Decimation of random points
	size_t freedom = 0;
	bool minmax_normalization = false;
	bool zscore_normalization = false;
	bool _11_normalization = false;
	bool early_stopping = true;
	bool test_mode = false;

	bool capture = false;
	bool progress = true;
	float tolerance = 1.0e-6;
	std::string weight_init_type = "xavier";
	bool layer_graph_only = 0;

	tiny_dnn::core::backend_t default_backend_type = tiny_dnn::core::backend_t::internal;
	//tiny_dnn::core::backend_t default_backend_type = tiny_dnn::core::backend_t::intel_mkl;

	tiny_dnn::tensor_t iX;
	tiny_dnn::tensor_t iY;
	tiny_dnn::tensor_t nX;
	tiny_dnn::tensor_t nY;
	std::vector<float_t> Mean_x, Mean_y;
	std::vector<float_t> Sigma_x, Sigma_y;
	std::vector<float_t> Min_x, Min_y;
	std::vector<float_t> MaxMin_x, MaxMin_y;
	tiny_dnn::network2<tiny_dnn::sequential> nn;
	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	std::string opt_type = "adam";
	size_t input_size = 32;
	size_t n_minibatch = 10;
	size_t n_eval_minibatch = 10;
	size_t n_train_epochs = 2000;
	float_t learning_rate = 1.0;
	int plot = 1;
#ifdef USE_LIBTORCH
	std::string model_file = "fit.pt";
#else
	std::string model_file = "fit.model";
#endif

	int getStatus() const
	{
		return error;
	}

	void normalize_info_save(std::string& normalize_type)
	{
		if (test_mode) return;

		FILE* fp = fopen("normalize_info.txt", "w");
		if (fp)
		{
			fprintf(fp, "%s\n", normalize_type.c_str());
			fprintf(fp, "%d %d\n", nX[0].size(), nY[0].size());
			for (int i = 0; i < nX[0].size(); i++)
			{
				fprintf(fp, "説明変数(%d)平均.分散:%.16f %.16f\n", i, Mean_x[i], Sigma_x[i]);
				fprintf(fp, "説明変数(%d)Min.Max:%.16f %.16f\n", i, Min_x[i], MaxMin_x[i]+ Min_x[i]);
			}
			for (int i = 0; i < nY[0].size(); i++)
			{
				fprintf(fp, "目的変数(%d)平均.分散:%.16f %.16f\n", i, Mean_y[i], Sigma_y[i]);
				fprintf(fp, "目的変数(%d)Min.Max:%.16f %.16f\n", i, Min_y[i], MaxMin_y[i]+ Min_y[i]);
			}
			fclose(fp);
		}
	}
	void normalize_info_load(std::string& normalize_type)
	{
		normalized = false;
		if (!use_trained_scale)
		{
			return;
		}
		if (!test_mode) return;

		FILE* fp = fopen("normalize_info.txt", "r");
		if (fp)
		{
			char buf[256];
			char dummy[128];
			double a = 0.0, b = 0.0;
			int d = 0, dd = 0;
			int tmp = 0;

			fgets(buf, 256, fp);
			sscanf(buf, "%s\n", dummy);
			char *p = strchr(dummy, '\n');
			if (p) *p = '\0';
			if (normalize_type != dummy)
			{
				printf("ERROR:normalize_type miss match!\n");
				exit(0);
			}

			fgets(buf, 256, fp);
			sscanf(buf, "%d %d\n", &d, &dd);
			if (d != nX[0].size() || dd != nY[0].size())
			{
				printf("ERROR:dimension miss match!\n");
				exit(0);
			}
			Mean_x = std::vector<float_t>(d, 0.0);
			Sigma_x = std::vector<float_t>(d, 0.0);
			Min_x = std::vector<float_t>(d, 0.0);
			MaxMin_x = std::vector<float_t>(d, 0.0);

			Mean_y = std::vector<float_t>(dd, 0.0);
			Sigma_y = std::vector<float_t>(dd, 0.0);
			Min_y = std::vector<float_t>(dd, 0.0);
			MaxMin_y = std::vector<float_t>(dd, 0.0);
			for (int i = 0; i < d; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "説明変数(%d)平均.分散:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Mean_x[i] = a;
				Sigma_x[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "説明変数(%d)Min.Max:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Min_x[i] = a;
				MaxMin_x[i] = b - a;
			}
			for (int i = 0; i < dd; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "目的変数(%d)平均.分散:%lf %lf\n", &tmp, &a, &b);
				Mean_y[i] = a;
				Sigma_y[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "目的変数(%d)Min.Max:%lf %lf\n", &tmp, &a, &b);
				Min_y[i] = a;
				MaxMin_y[i] = b - a;
			}
			printf("load scaling data\n");
			fclose(fp);
		}
		normalized = true;
	}

	NonLinearRegression(tiny_dnn::tensor_t& Xi,  tiny_dnn::tensor_t& Yi, std::vector<int>& normalizeskipp_, std::string& normalize_type= std::string("zscore"), double dec_random_=0.0, double fluctuation_=0.0, std::string regression_type = "", int classification_ = -1, bool test_mode_ = false, bool use_trained_scale_=true)
	{	
		normalizeskipp = normalizeskipp_;
		use_trained_scale = use_trained_scale_;
		test_mode = test_mode_;
		iX = Xi;
		iY = Yi;
		nX = Xi;
		nY = Yi;
		if (normalize_type == "zscore") zscore_normalization = true;
		if (normalize_type == "minmax") minmax_normalization = true;
		if (normalize_type == "[-1..1]") _11_normalization = true;

		classification = classification_;
		printf("classification:%d\n", classification);

		regression = regression_type;
		dec_random = dec_random_;
		fluctuation = fluctuation_;

		if (_11_normalization)
		{
			if (test_mode) normalize_info_load(normalize_type);

			normalize1_1(nX, Min_x, MaxMin_x);
			normalize1_1(nY, Min_y, MaxMin_y);
			printf("[-1,1] normalization\n");

			tiny_dnn::tensor_t dmyX = nX;
			tiny_dnn::tensor_t dmyY = nY;
			normalizeZ(dmyX, Mean_x, Sigma_x);
			normalizeZ(dmyY, Mean_y, Sigma_y);
			if (!test_mode) normalize_info_save(normalize_type);

			if (regression == "logistic" || classification >= 2)
			{
				printf("ERROR:no!! [-1, 1] normalization");
				exit(0);
			}

		}

		if (minmax_normalization)
		{
			if (test_mode) normalize_info_load(normalize_type);

			normalizeMinMax(nX, Min_x, MaxMin_x);
			normalizeMinMax(nY, Min_y, MaxMin_y);
			printf("minmax_normalization\n");

			tiny_dnn::tensor_t dmyX = nX;
			tiny_dnn::tensor_t dmyY = nY;
			normalizeZ(dmyX, Mean_x, Sigma_x);
			normalizeZ(dmyY, Mean_y, Sigma_y);

			if (!test_mode) normalize_info_save(normalize_type);

			if (regression == "logistic" || classification >= 2)
			{
				nY = Yi;
				for (int k = 0; k < nY[0].size(); k++)
				{
					Min_y[k] = 0.0;
					MaxMin_y[k] = 1.0;
				}
			}
		}
		if (zscore_normalization)
		{
			if (test_mode) normalize_info_load(normalize_type);

			normalizeZ(nX, Mean_x, Sigma_x);
			normalizeZ(nY, Mean_y, Sigma_y);
			printf("zscore_normalization\n");

			tiny_dnn::tensor_t dmyX = nX;
			tiny_dnn::tensor_t dmyY = nY;
			normalizeMinMax(dmyX, Min_x, MaxMin_x);
			normalizeMinMax(dmyY, Min_y, MaxMin_y);

			if (!test_mode) normalize_info_save(normalize_type);

			if (regression == "logistic" || classification >= 2 )
			{
				nY = Yi;
				for (int k = 0; k < nY[0].size(); k++)
				{
					Mean_y[k] = 0.0;
					Sigma_y[k] = 1.0;
				}
			}
			//FILE* fp = fopen("aaaaa.txt", "w");
			//fprintf(fp, "%s\n", regression.c_str());
			//fclose(fp);
			//Matrix<dnn_double> tmpx, tmpy;
			//TensorToMatrix(nX, tmpx);
			//TensorToMatrix(nY, tmpy);
			//tmpy = tmpy.appendCol(tmpx);
			//tmpy.print_csv("Tn.csv");
		}
		if (regression == "logistic" || classification >= 0)
		{
			int class_num = classification;
			if (regression == "logistic")
			{
				class_num = 2;
			}

			bool class_id_number = true;
			float class_min = 10000000;
			float class_max = -1;
			for (int i = 0; i < nY.size(); i++)
			{
				for (int k = 0; k < nY[0].size(); k++)
				{
					if (nY[i][k] > class_max)
					{
						class_max = nY[i][k];
					}
					if (nY[i][k] < class_min)
					{
						class_min = nY[i][k];
					}
					if (fabs(nY[i][k] - (int)nY[i][k]) > 1.0e-10)
					{
						class_id_number = false;
					}
				}
			}
			class_minmax[0] = class_min;
			class_minmax[1] = class_max;

			if (class_min > class_max)
			{
				error = -1;
				return;
			}

			if (class_max > class_num)
			{
				printf("WARNING:classification:%d < class_max:%d\n", class_num, (int)class_max);
				fflush(stdout);
				error = -1;
			}
			if (class_min < 0)
			{
				printf("WARNING:class_min:%f\n", class_min);
				fflush(stdout);
				error = -1;
			}
			if (class_num < 2)
			{
				error = -1;
			}
			if (!class_id_number)
			{
				error = -1;
			}
			if (fabs(class_min) < 0.0 || fabs(class_max) < 1.0)
			{
				error = -1;
			}

			if (class_num >= 2 && error == -1)
			{
				for (int i = 0; i < nY.size(); i++)
				{
					for (int k = 0; k < nY[0].size(); k++)
					{
						nY[i][k] = (int)((class_num - 1)*(nY[i][k] - class_min) / (class_max - class_min));
						nY[i][k] = std::min((float_t)(class_num-1), std::max(float_t(0.0), float_t(nY[i][k])));
						iY[i][k] = nY[i][k];
					}
				}
				std::ofstream stream("classification_warning.txt");
				if (!stream.bad())
				{
					stream << class_minmax[0] << "---" << class_minmax[1] << std::endl;
					double dt = (class_max - class_min) / class_num;
					for (int i = 0; i < class_num; i++)
					{
						stream << "class index:" << i << " (class number:" << i + 1 << ") " << (i*dt + class_min) << " " << (i + 1)*dt + class_min << std::endl;
					}
					for (int i = 0; i < class_num; i++)
					{
						stream <<  i+1 << " " << (i*dt+ class_min) << " " << (i+1)*dt+ class_min << std::endl;
					}
					for (int i = 0; i < nY.size(); i++)
					{
						for (int k = 0; k < nY[0].size(); k++)
						{
							stream << nY[i][k] << std::endl;
						}
					}
					stream.flush();
				}
				error = 0;
			}
		}
	}
	void visualize_loss(int n)
	{
		visualize_state_flag = n;
		if (n > 0)
		{
			fp_error_loss = fopen("error_loss.dat", "w");
			fp_error_vari_loss = fopen("error_var_loss.dat", "w");
			fp_accuracy = fopen("accuracy.dat", "w");
		}
		else
		{
			if (fp_error_loss)
			{
				fclose(fp_error_loss);
				fp_error_loss = NULL;
			}
			if (fp_error_vari_loss)
			{
				fclose(fp_error_vari_loss);
				fp_error_vari_loss = NULL;
			}
			if (fp_accuracy)
			{
				fclose(fp_accuracy);
				fp_accuracy = NULL;
			}
		}
	}

	int data_set(float test = 0.3f)
	{
		train_images.clear();
		train_labels.clear();
		test_images.clear();
		test_images.clear();

		size_t dataAll = iY.size();
		printf("dataset All:%d->", dataAll);
		size_t test_Num = dataAll*test;
		printf("test num(%f%%):%d->", test, test_Num);
		int datasetNum = dataAll - test_Num;

		if (datasetNum == 0 || datasetNum < this->n_minibatch)
		{
			printf("ERROR:Too many min_batch or Sequence length\n");
			error = -1;
			return error;
		}
		size_t train_num_max = datasetNum;
		printf("train:%d test:%d\n", datasetNum, test_Num);

		std::random_device rnd;     // 非決定的な乱数生成器を生成
		std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタ
		std::uniform_int_distribution<> rand1(0, dataAll - 1);

		std::vector<int> use_index(dataAll, -1);

		//Add random fluctuation
		if (fluctuation > 0)
		{
			printf("Add random fluctuation:%f\n", fluctuation);
			std::normal_distribution<> rand_fl(0.0, 1.0);
			for (int i = 0; i < dataAll; i++)
			{
				for (int k = 0; k < nY[0].size(); k++)
				{
					nY[i][k] = nY[i][k] * (1.0 + fluctuation*rand_fl(mt));
				}
			}
		}

		printf("dec_random:%f\n", dec_random);
		if (dec_random > 0)
		{
			std::uniform_real_distribution<> rand(0.0, 1.0);
			for (int i = 0; i < dataAll; i++)
			{
				if (rand(mt) < dec_random)
				{
					use_index[i] = i;
				}
			}
		}

		if (test_Num > 0)
		{
			do
			{
				int ii = rand1(mt);
				if (use_index[ii] != -1)
				{
					continue;
				}

				if (classification >= 2)
				{
					test_images.push_back(nX[ii]);
					test_labels.push_back(label2tensor(iY[ii][0], classification));
				}
				else
				{
					test_images.push_back(nX[ii]);
					test_labels.push_back(nY[ii]);
				}
				use_index[ii] = ii;

				if (test_images.size() == test_Num) break;
				for (int i = 1; i < test_Num*0.05; i++)
				{
					if (ii + i >= dataAll) break;

					if (classification >= 2)
					{
						test_images.push_back(nX[ii + i]);
						test_labels.push_back(label2tensor(iY[ii + i][0], classification));
					}
					else
					{
						test_images.push_back(nX[ii + i]);
						test_labels.push_back(nY[ii + i]);
					}
					use_index[ii + i] = ii + i;
					if (test_images.size() == test_Num) break;
				}
			} while (test_images.size() != test_Num);
		}
		test_data_index = use_index;

		std::vector<int> index(dataAll, -1);

		size_t l = 0;
		do
		{
			int ii = rand1(mt);
			if (test_data_index[ii] != -1) continue;
			if (index[ii] != -1) continue;
			if (classification >= 2)
			{
				train_images.push_back(nX[ii]);
				train_labels.push_back(label2tensor(iY[ii][0], classification));
			}
			else
			{
				train_images.push_back(nX[ii]);
				train_labels.push_back(nY[ii]);
			}
			index[ii] = ii;
			++l;
			if (l > datasetNum * 100 && train_images.size() != datasetNum)
			{
				for (int k = 0; k < index.size(); k++)
				{
					if (index[k] != -1)
					{
						if (classification >= 2)
						{
							train_images.push_back(nX[ii]);
							train_labels.push_back(label2tensor(iY[ii][0], classification));
						}
						else
						{
							train_images.push_back(nX[ii]);
							train_labels.push_back(nY[ii]);
						}
						index[ii] = ii;
					}
					if (train_images.size() == datasetNum) break;
				}
				break;
			}
		} while (train_images.size() != datasetNum);

		FILE* fp = fopen("test_point.dat", "w");

		for (int ii = 0; ii < iY.size(); ii++)
		{
			int i = test_data_index[ii];
			if (i == -1) continue;

			fprintf(fp, "%d ", i);
			for (int k = 0; k < iY[0].size()-1; k++)
			{
				fprintf(fp, "%f ", iY[i][k]);
		}
			fprintf(fp, "%f\n", iY[i][iY[0].size() - 1]);
		}
		fclose(fp);
		printf("train:%d test:%d\n", train_images.size(), test_images.size());
		return 0;
	}

	void construct_net(int n_layers = 5)
	{
		SigHandler_nonlinear_regression(0);
		signal(SIGINT, SigHandler_nonlinear_regression);
		signal(SIGTERM, SigHandler_nonlinear_regression);
		signal(SIGBREAK, SigHandler_nonlinear_regression);
		signal(SIGABRT, SigHandler_nonlinear_regression);

		using tanh = tiny_dnn::activation::tanh;
		using recurrent = tiny_dnn::recurrent_layer;

		int hidden_size = train_images[0].size() * 50;

		// clip gradients
		tiny_dnn::recurrent_layer_parameters params;
		params.clip = 0;
		params.bptt_max = 1e9;

		size_t in_w = train_images[0].size();
		size_t in_h = 1;
		size_t in_map = 1;

		LayerInfo layers(in_w, in_h, in_map);
		if (regression == "linear" || regression == "logistic")
		{
			/**/
		}
		else
		{
			nn << layers.add_fc(input_size);
			if ( this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();

			for (int i = 0; i < n_layers; i++) {
				if (dropout && i == n_layers-1) nn << layers.add_dropout(dropout);
				nn << layers.add_fc(input_size);
				if (this->activation_fnc == "tanh") nn << layers.tanh();
				if (this->activation_fnc == "relu") nn << layers.relu();
			}
		}
		if (classification >= 2)
		{
			if (dropout ) nn << layers.add_dropout(dropout);
			nn << layers.add_fc(std::min((int)input_size, classification*2));
			if (this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();
			nn << layers.add_fc(classification);
		}
		else
		{
			nn << layers.add_fc(train_labels[0].size());
		}

		if (regression == "logistic")
		{
			nn << layers.sigmoid();
		}
		if (classification >= 2)
		{
			nn << layers.softmax(classification);
		}

#if 10
#ifdef CNN_USE_AVX
		for (auto n : nn)
		{
			if (n->layer_type() == "fully-connected")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::avx);
			}
		}
#endif

#ifdef CNN_USE_INTEL_MKL
		for (auto n : nn)
		{
			if (n->layer_type() == "fully-connected")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::intel_mkl);
			}
		}
#endif
#endif

		if (weight_init_type == "xavier")
		{
			nn.weight_init(tiny_dnn::weight_init::xavier());
		}
		if (weight_init_type == "lecun")
		{
			nn.weight_init(tiny_dnn::weight_init::lecun());
		}
		if (weight_init_type == "gaussian")
		{
			nn.weight_init(tiny_dnn::weight_init::gaussian());
		}
		if (weight_init_type == "constant")
		{
			nn.weight_init(tiny_dnn::weight_init::constant());
		}
		if (weight_init_type == "he")
		{
			nn.weight_init(tiny_dnn::weight_init::he());
		}
		for (auto n : nn) n->set_parallelize(true);
		printf("layers:%zd\n", nn.depth());
		freedom = layers.get_parameter_num();
		printf("freedom:%zd\n", freedom);

#ifdef USE_GRAPHVIZ_DOT
		char cmd[512];
		// generate graph model in dot language
		std::ofstream ofs("graph_net.txt");
		tiny_dnn::graph_visualizer viz(nn, "graph");
		viz.generate(ofs);
		
		graphviz_path_::getGraphvizPath();
		std::string path = graphviz_path_::path_;
		if (path != "")
		{
			sprintf(cmd, "\"%s\\dot.exe\" -T%s %s -o Digraph.%s", path.c_str(), "png", "graph_net.txt", "png");
		}
		else
		{
			sprintf(cmd, "dot.exe -T%s %s -o Digraph.%s", "png", "graph_net.txt", "png");
		}
		system(cmd);
		printf("%s\n", cmd);
		//if (path != "")
		//	printf("\"%s\\dot.exe\" -Tgif graph_net.txt -o graph.gif\n", path.c_str());
		//else
		//	printf("dot -Tgif graph_net.txt -o graph.gif\n");
#endif
	}


	void fit(int n_layers = 5, int input_unit = 32)
	{
		if (n_layers < 0) n_layers = 5;
		if (input_unit < 0) input_unit = 32;
		
		input_size = input_unit;

		nn.set_input_size(train_images.size());
		using train_loss = tiny_dnn::mse;

		{
			int hidden_size = train_images[0].size() * 50;

			char *param = "train_params.txt";
			if (test_mode) param = "test_params.txt";
			FILE* fp = fopen(param, "w");
			if (fp)
			{
				fprintf(fp, "test_mode:%d\n", test_mode);
				fprintf(fp, "learning_rate:%f\n", learning_rate);
				fprintf(fp, "opt_type:%s\n", opt_type.c_str());
				fprintf(fp, "n_train_epochs:%d\n", n_train_epochs);
				fprintf(fp, "n_minibatch:%d\n", n_minibatch);
				fprintf(fp, "n_eval_minibatch:%d\n", n_eval_minibatch);

				fprintf(fp, "n_layers:%d\n", n_layers);
				fprintf(fp, "n_hidden_size:%d\n", hidden_size);
				fprintf(fp, "dropout:%f\n", dropout);
				fprintf(fp, "prophecy:%d\n", 0);
				fprintf(fp, "tolerance:%f\n", tolerance);
				fprintf(fp, "early_stopping:%d\n", this->early_stopping);
				fprintf(fp, "input_size:%d\n", input_size);
				fprintf(fp, "classification:%d\n", classification);
				fprintf(fp, "batch_shuffle:%d\n", batch_shuffle);
				fprintf(fp, "weight_init_type:%s\n", this->weight_init_type.c_str());
				fprintf(fp, "activation_fnc:%s\n", this->activation_fnc.c_str());
				fclose(fp);
			}
			else
			{
				return;
			}


			float maxvalue = -999999999.0;
			float minvalue = -maxvalue;
			char* images_file = "train_images_tr.csv";
			if (test_mode) images_file = "train_images_ts.csv";
			fp = fopen(images_file, "w");
			if (fp)
			{
				for (int i = 0; i < train_images.size(); i++)
				{
					for (int j = 0; j < train_images[i].size(); j++)
					{
						if (maxvalue < train_images[i][j])
						{
							maxvalue = train_images[i][j];
						}
						if (minvalue > train_images[i][j])
						{
							minvalue = train_images[i][j];
						}
						fprintf(fp, "%f", train_images[i][j]);
						if (j == train_images[i].size() - 1)
						{
							fprintf(fp, "\n");
						}
						else
						{
							fprintf(fp, ",");
						}
					}
				}
				fclose(fp);
			}
			else
			{
				return;
			}

			images_file = "test_images_tr.csv";
			if (test_mode) images_file = "test_images_ts.csv";
			fp = fopen(images_file, "w");
			if (fp)
			{
				for (int i = 0; i < test_images.size(); i++)
				{
					for (int j = 0; j < test_images[i].size(); j++)
					{
						fprintf(fp, "%f", test_images[i][j]);
						if (j == test_images[i].size() - 1)
						{
							fprintf(fp, "\n");
						}
						else
						{
							fprintf(fp, ",");
						}
					}
				}
				fclose(fp);
			}
			else
			{
				return;
			}

			char* labels_file = "train_labels_tr.csv";
			if (test_mode) labels_file = "train_labels_ts.csv";
			fp = fopen(labels_file, "w");
			if (fp)
			{
				for (int i = 0; i < train_labels.size(); i++)
				{
					for (int j = 0; j < train_labels[i].size(); j++)
					{
						if (maxvalue < train_labels[i][j])
						{
							maxvalue = train_labels[i][j];
						}
						if (minvalue > train_labels[i][j])
						{
							minvalue = train_labels[i][j];
						}
						fprintf(fp, "%f", train_labels[i][j]);
						if (j == train_labels[i].size() - 1)
						{
							fprintf(fp, "\n");
						}
						else
						{
							fprintf(fp, ",");
						}
					}
				}
				fclose(fp);
			}
			else
			{
				return;
			}

			labels_file = "test_labels_tr.csv";
			if (test_mode) labels_file = "test_labels_ts.csv";
			fp = fopen(labels_file, "w");
			if (fp)
			{
				for (int i = 0; i < test_labels.size(); i++)
				{
					for (int j = 0; j < test_labels[i].size(); j++)
					{
						fprintf(fp, "%f", test_labels[i][j]);
						if (j == test_labels[i].size() - 1)
						{
							fprintf(fp, "\n");
						}
						else
						{
							fprintf(fp, ",");
						}
					}
				}
				fclose(fp);
			}
			else
			{
				return;
			}

			fp = fopen(param, "a");
			if (fp)
			{
				fprintf(fp, "maxvalue:%f\n", maxvalue);
				fprintf(fp, "minvalue:%f\n", minvalue);
				fclose(fp);
			}
			else
			{
				return;
			}
		}

		tiny_dnn::adam				optimizer_adam;
		tiny_dnn::gradient_descent	optimizer_sgd;
		tiny_dnn::RMSprop			optimizer_rmsprop;
		tiny_dnn::adagrad			optimizer_adagrad;

		if (opt_type == "sgd")
		{
			std::cout << "optimizer:" << "sgd" << std::endl;
			optimizer_sgd.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_sgd.alpha << std::endl;

		}else
		if (opt_type == "rmsprop")
		{
			std::cout << "optimizer:" << "RMSprop" << std::endl;
			optimizer_rmsprop.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_rmsprop.alpha << std::endl;

		}else
		if (opt_type == "adagrad")
		{
			std::cout << "optimizer:" << "adagrad" << std::endl;
			optimizer_adagrad.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_adagrad.alpha << std::endl;

		}else
		if (opt_type == "adam" )
		{
			std::cout << "optimizer:" << "adam" << std::endl;

			//optimizer.alpha *=
			//	std::min(tiny_dnn::float_t(4),
			//		static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));
			optimizer_adam.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_adam.alpha << std::endl;

		}

		construct_net(n_layers);
		if (layer_graph_only)
		{
			return;
		}

		if (opt_type == "adam") optimizer_adam.reset();
		if (opt_type == "sgd")	optimizer_sgd.reset();
		if (opt_type == "rmsprop")	optimizer_rmsprop.reset();
		if (opt_type == "adagrad")optimizer_adagrad.reset();
		
		tiny_dnn::timer finish_predict;
		tiny_dnn::timer t;

		tiny_dnn::progress_display disp(nn.get_input_size());

		auto on_enumerate_epoch = [&]() {

			if (epoch % 10 == 0) {
				if (opt_type == "adam" && optimizer_adam.alpha > 1.0e-12)		optimizer_adam.alpha *= 0.97;
				if (opt_type == "sgd" && optimizer_sgd.alpha > 1.0e-12)			optimizer_sgd.alpha *= 0.97;
				if (opt_type == "rmsprop" && optimizer_rmsprop.alpha > 1.0e-12)	optimizer_rmsprop.alpha *= 0.97;
				if (opt_type == "adagrad" && optimizer_adagrad.alpha > 1.0e-12)	optimizer_adagrad.alpha *= 0.97;
			}
			if (epoch % 1 == 0)
			{
				std::cout << "\nEpoch " << epoch << "/" << n_train_epochs << " finished. "
					<< t.elapsed() << "s elapsed." << std::endl;
			}

#ifdef USE_LIBTORCH
#else
			if ( this->batch_shuffle)
			{
				tiny_dnn::tensor_t tmp_train_images = train_images;
				tiny_dnn::tensor_t tmp_train_labels = train_labels;

				std::vector<int> index(train_images.size());
				for (int i = 0; i < train_images.size(); i++)
				{
					index[i] = i;
				}
				std::mt19937 mt(epoch);
				std::shuffle(index.begin(), index.end(), mt);

#pragma omp parallel for
				for (int i = 0; i < train_images.size(); i++)
				{
					tmp_train_images[i] = train_images[index[i]];
					tmp_train_labels[i] = train_labels[index[i]];
				}
				train_labels = tmp_train_labels;
				train_images = tmp_train_images;
			}
#endif

			if (epoch >= 3 && plot && epoch % plot == 0)
			{
				gen_visualize_fit_state();
#ifdef USE_LIBTORCH
				//if (this->n_sampling > 0)
				//{
				//	std::mt19937 mt(epoch);
				//	std::uniform_real_distribution r(0.01, 0.6);
				//	for (int i = 0; i < this->n_sampling; i++)
				//	{
				//		set_sampling(r(mt));
				//		gen_visualize_fit_state(true);
				//	}
				//	reset_sampling();
				//}
#endif
			}
			if (convergence)
			{
				printf("convergence!!\n");
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				error = 0;

			}
			if (early_stopp)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				error = 1;
				printf("early_stopp!!\n");
			}
			if (is_stopping_solver())
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				printf("stop_ongoing_training!!\n");
			}

			if (progress) disp.restart(nn.get_input_size());
			t.restart();
			//rnn_state_reset(nn);
			++epoch;

			if (epoch % 5 == 0)
			{
				float one_ephch = finish_predict.elapsed() / 5.0;
				one_ephch *= (n_train_epochs - epoch);
				std::ofstream stream("Time_to_finish.txt");

				if (!stream.bad())
				{
					stream << "Time to finish:" << one_ephch << "[sec] = " << one_ephch / 60.0 << "[min]" << std::endl;
					stream.flush();
				}
				std::cout << "Time to finish:" << one_ephch << "[sec] = " << one_ephch / 60.0 << "[min]" << std::endl;
				finish_predict.restart();
			}
		};
		auto on_enumerate_minibatch = [&]() {
			if (progress) disp += n_minibatch;

			//if (epoch < 3 && plot && (batch+1) % plot == 0)
			//{
			//	gen_visualize_fit_state();
			//}
			if (convergence)
			{
				printf("convergence!!\n");
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				error = 0;
			}
			if (early_stopp)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				error = 1;
				printf("early_stopp!!\n");
			}
			if (is_stopping_solver())
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				printf("stop_ongoing_training!!\n");
			}

			if (ctr_c_stopping_nonlinear_regression)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_stop_ongoing_training();
				}
				else
#endif
				{
					nn.stop_ongoing_training();
				}
				error = 1;
				printf("CTR-C stopp!!\n");
			}
			++batch;
		};

		if (!test_mode)
		{
			if (!use_libtorch)
			{
				try
				{
					// training
					//nn.fit<tiny_dnn::mse>(*optimizer_, train_images, train_labels,
					//	n_minibatch,
					//	n_train_epochs,
					//	on_enumerate_minibatch,
					//	on_enumerate_epoch
					//	);
					if (opt_type == "adam")
					{
						// training
						if (classification < 2)
						{
							nn.fit<tiny_dnn::mse>(optimizer_adam, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
						else
						{
							nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer_adam, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
					}
					if (opt_type == "sgd")
					{
						// training
						if (classification < 2)
						{
							nn.fit<tiny_dnn::mse>(optimizer_sgd, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
						else
						{
							nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer_sgd, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
					}
					if (opt_type == "rmsprop")
					{
						// training
						if (classification < 2)
						{
							nn.fit<tiny_dnn::mse>(optimizer_rmsprop, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
						else
						{
							nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer_rmsprop, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
					}
					if (opt_type == "adagrad")
					{
						// training
						if (classification < 2)
						{
							nn.fit<tiny_dnn::mse>(optimizer_adagrad, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
						else
						{
							nn.fit<tiny_dnn::cross_entropy_multiclass>(optimizer_adagrad, train_images, train_labels,
								n_minibatch,
								n_train_epochs,
								on_enumerate_minibatch,
								on_enumerate_epoch
								);
						}
					}
				}
				catch (tiny_dnn::nn_error &err) {
					std::cerr << "Exception: " << err.what() << std::endl;
					error = -1;
					return;
				}
			}
		}

#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			if (!test_mode)
			{
				torch_read_train_params();
			}
			else
			{
				torch_read_test_params();
			}
			printf("****** pytorch (C++) mode ******\n");
			try
			{
				torch_train_fc(
					train_images,
					train_labels,
					n_minibatch,
					n_train_epochs,
					(char*)regression.c_str(),
					on_enumerate_minibatch, on_enumerate_epoch);
			}
			catch (std::exception& err)
			{
				std::cout << err.what() << std::endl;
			}
			catch (...)
			{
				printf("exception!\n");
			}
		}
#endif

		try
		{
			if (!fit_best_saved && !test_mode)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_save("fit_best.pt");
					fit_best_saved = true;
				}
				else
#endif
				{
					nn.save("fit_best.model");
					fit_best_saved = true;
					nn.save("fit_best.model.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
				}
			}

#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_load("fit_best.pt");
			}
			else
#endif
			{
				nn.load("fit_best.model");
			}
			gen_visualize_fit_state();
#ifdef USE_LIBTORCH
			//if (this->n_sampling > 0)
			//{
			//	std::mt19937 mt(epoch);
			//	std::uniform_real_distribution r(0.01, 0.6);
			//	for (int i = 0; i < this->n_sampling; i++)
			//	{
			//		set_sampling(r(mt));
			//		gen_visualize_fit_state(true);
			//	}
			//	reset_sampling();
			//}
#endif

		}
		catch (tiny_dnn::nn_error& msg)
		{
			printf("%s\n", msg.what());
			printf("fit_best.model open error.\n");
		}
		catch (std::exception& err)
		{
			std::cout << err.what() << std::endl;
		}
		catch (...)
		{
			printf("exception!\n");
		}

#ifndef USE_LIBTORCH
		std::cout << "end training." << std::endl;
#endif

		if (fp_error_loss)fclose(fp_error_loss);
		if (fp_error_vari_loss)fclose(fp_error_vari_loss);
		if (fp_accuracy)fclose(fp_accuracy);
		fp_error_loss = NULL;
		fp_error_vari_loss = NULL;
		fp_accuracy = NULL;

		// save network model & trained weights
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			torch_save(model_file.c_str());
		}
		else
#endif
		{
			nn.save(model_file + ".json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
			nn.save(model_file);
		}

		if (regression == "linear" || regression == "logistic")
		{
			int n = 0;
			std::vector<double> w;
			double bias;
			FILE* fp = fopen((model_file + ".json").c_str(), "r");
			if (fp)
			{
				double ww;

				char buf[256];
				while (fgets(buf, 256, fp) != NULL)
				{
					char* p;
					if ( p = strstr(buf, "\"in_size\":"))
					{
						sscanf(p, "\"in_size\": %d", &n);
						break;
					}
				}
				while (fgets(buf, 256, fp) != NULL)
				{
					if (strstr(buf, "\"value0\": ["))
					{
						break;
					}
				}
				for (int i = 0; i < n; i++)
				{
					fgets(buf, 256, fp);
					sscanf(buf, " %lf", &ww);
					w.push_back(ww);
				}
				while (fgets(buf, 256, fp) != NULL)
				{
					if (strstr(buf, "\"value1\": ["))
					{
						break;
					}
				}
				fgets(buf, 256, fp);
				sscanf(buf, " %lf", &bias);
				fclose(fp);
			}		
			
			fp = fopen("dnn_regression_fit.model", "w");
			if (fp)
			{
				fprintf(fp, "n %d\n", n);
				fprintf(fp, "bias %.16g\n", bias);
				for (int i = 0; i < n; i++)
				{
					fprintf(fp, "%.16g\n", w[i]);
				}
				fclose(fp);
			}
			if (regression == "logistic")
			{
				fp = fopen("dnn_libsvm.model", "w");
				if (fp)
				{
					fprintf(fp, "solver_type L2R_LR\n");
					fprintf(fp, "nr_class 2\n");
					fprintf(fp, "label 1 0\n");
					fprintf(fp, "nr_feature %d\n", n);
					fprintf(fp, "bias %.16g\n", 0.0);
					fprintf(fp, "w\n");
					for (int i = 0; i < n; i++)
					{
						fprintf(fp, "%.16g \n", w[i]);
					}
					fprintf(fp, "%.16g \n", bias);
					fclose(fp);
				}
			}
		}
	}

	tiny_dnn::vec_t predict_next(tiny_dnn::vec_t& x)
	{
		tiny_dnn::vec_t xx = x;
		if (zscore_normalization)
		{
			for (int k = 0; k < xx.size(); k++)
			{
				xx[k] = (x[k] - Mean_x[k]) / (Sigma_x[k] + 1.0e-10);
			}
		}
		else
		if (minmax_normalization)
		{
			for (int k = 0; k < xx.size(); k++)
			{
				xx[k] = (x[k] - Min_x[k]) / (MaxMin_x[k] + 1.0e-10);
			}
		}
		else
		if (_11_normalization)
		{
			for (int k = 0; k < xx.size(); k++)
			{
				xx[k] = (x[k] - Min_x[k])*2 / (MaxMin_x[k] + 1.0e-10) - 1;
			}
		}
		tiny_dnn::vec_t& y = nn.predict(xx);
		if (zscore_normalization)
		{
			for (int k = 0; k < y.size(); k++)
			{
				y[k] = y[k] * Sigma_y[k] + Mean_y[k];
			}
		}else
		if (minmax_normalization)
		{
			for (int k = 0; k < y.size(); k++)
			{
				y[k] = y[k] * MaxMin_y[k] + Min_y[k];
			}
		}else
		if (minmax_normalization)
		{
			for (int k = 0; k < y.size(); k++)
			{
				y[k] = 0.5*(y[k]+1) * MaxMin_y[k] + Min_y[k];
			}
		}
	}

	double get_accuracy(FILE* fp=NULL)
	{
		double accuracy = 0;
		double tot = 0;

		if (fp == NULL) fp = stdout;
		if (regression == "logistic")
		{
			for (int i = 0; i < nY.size(); i++)
			{
				tiny_dnn::vec_t& y = nn.predict(nX[i]);
				for (int k = 0; k < y.size(); k++)
				{
					auto z = nY[i][k];
					float_t d;
					if (y[k] < 0.5) d = 0.0;
					else d = 1.0;

					if (d == z)
					{
						accuracy++;
					}
					tot++;
				}
			}
			fprintf(fp, "accuracy:%.3f%%\n", 100.0*accuracy / tot);
		}

		//if (classification > 2)
		//{
		//	for (int i = 0; i < nY.size(); i++)
		//	{
		//		tiny_dnn::vec_t& y = nn.predict(nX[i]);
		//		for (int k = 0; k < y.size(); k++)
		//		{
		//			auto z = train_labels[i][k];
		//			//if (zscore_normalization)
		//			//{
		//			//	y[k] = y[k] * Sigma_y[k] + Mean_y[k];
		//			//	z = z * Sigma_y[k] + Mean_y[k];
		//			//}
		//			//if (minmax_normalization)
		//			//{
		//			//	y[k] = y[k] * MaxMin_y[k] + Min_y[k];
		//			//	z = z * MaxMin_y[k] + Min_y[k];
		//			//}

		//			float_t d;
		//			if (classification == 2)
		//			{
		//				if (y[k] < 0.5) d = 0.0;
		//				else d = 1.0;

		//				if (d == z)
		//				{
		//					accuracy++;
		//				}
		//			}
		//			tot++;
		//		}
		//	}
		//	fprintf(fp, "accuracy:%.3f%%\n", 100.0*accuracy / tot);
		//}
		return 100.0*accuracy / tot;
	}

	float get_loss(tiny_dnn::network2<tiny_dnn::sequential>& model, std::vector<tiny_dnn::vec_t>& images, std::vector<tiny_dnn::vec_t>& labels)
	{
		if (images.size() == 0) return 0;
		float_t loss = model.get_loss<tiny_dnn::cross_entropy_multiclass>(images, labels);
		std::cout << "loss " << loss << std::endl;
		return loss;
	}

	tiny_dnn::result  get_accuracy(tiny_dnn::network2<tiny_dnn::sequential>& model, std::vector<tiny_dnn::vec_t>& images, std::vector<tiny_dnn::vec_t>& labels)
	{
		tiny_dnn::result result;
		float accuracy = 0;

		if (images.size() == 0)
		{
			result.num_total = 1;
			return result;
		}
#if 0
		std::vector<tiny_dnn::label_t> tmp_labels(images.size());
		std::vector<tiny_dnn::vec_t> tmp_images(images.size());

#pragma omp parallel for
		for (int i = 0; i < images.size(); i++)
		{
			int actual = -1;
			for (int k = 0; k < labels[i].size(); k++)
			{
				if (actual < labels[i][k])
				{
					actual = labels[i][k];
					actual = k;
				}
			}
			tmp_labels[i] = actual;
			tmp_images[i] = images[i];
		}
		result = model.test(tmp_images, tmp_labels);
		std::cout << result.num_success << "/" << result.num_total << std::endl;

#else

		for (int i = 0; i < images.size(); i++)
		{
			auto predict_y = model.predict(images[i]);

			float max_value = 0;
			int predicted = -1;
			int actual = -1;
			for (int k = 0; k < labels[i].size(); k++)
			{
				if (actual < labels[i][k])
				{
					actual = labels[i][k];
					actual = k;
				}
				if (max_value < predict_y[k])
				{
					max_value = predict_y[k];
					predicted = k;
				}
			}
			if (predicted == actual) result.num_success++;
			result.num_total++;
			result.confusion_matrix[predicted][actual]++;
		}
		accuracy = 100.0*result.num_success / result.num_total;
		printf("%.3f%%\n", accuracy);
#endif
		//ConfusionMatrix
		std::cout << "ConfusionMatrix:" << std::endl;
		result.print_detail(std::cout);
		std::cout << result.num_success << "/" << result.num_total << std::endl;
		printf("accuracy:%.3f%%\n", result.accuracy());

		std::ofstream ofs("ConfusionMatrix.txt");
		result.print_detail(ofs);

		{
			FILE* fp = fopen("ConfusionMatrix.txt", "r");
			FILE* fp2 = fopen("ConfusionMatrix.r", "w");
			char* p = NULL;
			if (fp && fp2)
			{
				fprintf(fp2, "confusionMatrix <- data.frame(");

				char buf[4096];
				int nn = 0;
				fgets(buf, 4096, fp);
				fgets(buf, 4096, fp);
				fgets(buf, 4096, fp);
				do
				{
					int n = 0;
					p = buf;
					while (isspace(*p)) p++;

					fprintf(fp2, "c(");
					while (*p != '\n'&& *p != '\0')
					{
						if ( n > 1) fprintf(fp2, ",");
						do
						{
							if ( n >= 1 )fprintf(fp2, "%c", *p);
							p++;
						} while (!isspace(*p));
						while (isspace(*p)) p++;
						n++;
						if (*p == '\n')break;
					}
					if (n >= 1)fprintf(fp2, ")");
					p = fgets(buf, 4096, fp);
					if ( p != NULL )fprintf(fp2, ",");
					nn++;
				} while (p != NULL);
				fprintf(fp2, ")\r\n");

				fprintf(fp2, "colnames(confusionMatrix)<-c(");
				for (int i = 1; i <= nn; i++)
				{
					fprintf(fp2, "\"C%d\"", i);
					if (i < nn) fprintf(fp2, ",");
				}
				fprintf(fp2, ")\r\n");
				fprintf(fp2, "rownames(confusionMatrix)<-colnames(confusionMatrix)\r\n");
				fclose(fp);
				fclose(fp2);
			}
		}

		return result;
	}

	void report(double α=0.05, std::string& filename = std::string(""))
	{
		FILE* fp = fopen(filename.c_str(), "w");
		if (fp == NULL)
		{
			fp = stdout;
		}
		if (classification >= 2)
		{
			if (fp != stdout && fp != NULL) fclose(fp);
			tiny_dnn::network2<tiny_dnn::sequential> nn_test;

			tiny_dnn::result train_result;
			tiny_dnn::result test_result;
#ifdef USE_LIBTORCH
			void* torch_nn_test = NULL;
			if (use_libtorch)
			{
				torch_nn_test = torch_load_new("fit_best.pt");
				train_result = torch_get_accuracy_nn(torch_nn_test, train_images, train_labels, 1);
				test_result = torch_get_accuracy_nn(torch_nn_test, test_images, test_labels, 1);
			}
			else
#endif
			{
				nn_test.load("fit_best.model");

				train_result = get_accuracy(nn_test, train_images, train_labels);
				test_result = get_accuracy(nn_test, test_images, test_labels);
			}
			{
				if (fp) fclose(fp);
				fp = NULL;
				std::ofstream stream(filename);

				if (!stream.bad())
				{
					stream << "ConfusionMatrix(";
					if (test_mode)
					{
						stream << "test";
					}
					else
					{
						stream << "train";
					}
					stream << "):" << std::endl;
					train_result.print_detail(stream);
					stream << train_result.num_success << "/" << train_result.num_total << std::endl;
					stream << "accuracy:" << train_result.accuracy() << "%" << std::endl;
					stream << std::endl;
					stream << "ConfusionMatrix(test):" << std::endl;
					if (test_images.size() == 0)
					{
						stream << "----" << std::endl;
					}
					else
					{
						test_result.print_detail(stream);
						stream << test_result.num_success << "/" << test_result.num_total << std::endl;
						stream << "accuracy:" << test_result.accuracy() << "%" << std::endl;
					}
					stream.flush();
				}
			}
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_delete_load_model(torch_nn_test);
				torch_nn_test = NULL;
			}
#endif
			return;
		}

		if (regression == "logistic")
		{
			get_accuracy(fp);
			return;
		}

		std::vector<double> mer;
		std::vector<double> xx;
		std::vector<double> yy;
		std::vector<double> ff;
		std::vector<double> dd;
		double mse = 0.0;
		double rmse = 0.0;
		for (int i = 0; i < nY.size(); i++)
		{
			tiny_dnn::vec_t predict_y;
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				predict_y = torch_predict(nX[i]);
			}
			else
#endif
			{
				predict_y = nn.predict(nX[i]);
			}
			tiny_dnn::vec_t y(predict_y.size());
			tiny_dnn::vec_t z(predict_y.size());
			for (int k = 0; k < predict_y.size(); k++)
			{
				z[k] = nY[i][k];
				y[k] = predict_y[k];

				if (zscore_normalization)
				{
					y[k] = y[k] * Sigma_y[k] + Mean_y[k];
					z[k] = z[k] * Sigma_y[k] + Mean_y[k];
				}else
				if (minmax_normalization)
				{
					y[k] = y[k] * MaxMin_y[k] + Min_y[k];
					z[k] = z[k] * MaxMin_y[k] + Min_y[k];
				}
				else
				if (_11_normalization)
				{
					y[k] = 0.5*(y[k] + 1) * MaxMin_y[k] + Min_y[k];
					z[k] = 0.5*(z[k] + 1) * MaxMin_y[k] + Min_y[k];
				}

				double d = (y[k] - z[k]);
				mse += d*d;
				yy.push_back(y[k]);
				ff.push_back(z[k]);

				mer.push_back(fabs(d) / z[k]);
			}
		}
		double se = mse;
		mse /= train_images.size();
		rmse = sqrt(mse);
		double Maximum_likelihood_estimator = mse;
		double Maximum_log_likelihood = log(2.0*M_PI) + log(Maximum_likelihood_estimator) + 1.0;

		Maximum_log_likelihood *= -0.5*((train_images.size()*train_images.size()));

		double AIC = train_images.size()*(log(2.0*M_PI*se / train_images.size()) + 1) + 2.0*(nX[0].size() +2.0);
		if (true)	//bias use
		{
			AIC = train_images.size()*(log(2.0*M_PI*se / train_images.size()) + 1) + 2.0*(nX[0].size() + 1.0);
		}

		double d_sum = 0.0;
		double f_sum = 0.0;
		double mean_ff = 0.0;
		double mean_yy = 0.0;
		for (int i = 0; i < yy.size(); i++)
		{
			mean_yy += yy[i];
			mean_ff += ff[i];
		}
		f_sum = mean_ff;
		mean_ff /= yy.size();
		mean_yy /= yy.size();
		//printf("mean_ff:%f\n", mean_ff);
		//printf("mean_yy:%f\n", mean_yy);

		double syy = 0.0;
		double sff = 0.0;
		double y_yy_f_yy = 0.0;
		for (int i = 0; i < yy.size(); i++)
		{
			double y_yy = (yy[i] - mean_yy);
			double f_ff = (ff[i] - mean_ff);
			//printf("[%03d]:%f-%f=%f, %f-%f=%f\n", i, yy[i] , mean_yy, y_yy, ff[i] , mean_ff, f_ff);
			y_yy_f_yy += y_yy*f_ff;
			syy += y_yy*y_yy;
			sff += f_ff*f_ff;
		}
		//printf("se:%f\n", se);
		//printf("sff:%f\n", sff);
		//printf("syy:%f\n", syy);
		//printf("r2:%f\n", sff/syy);

		double r = y_yy_f_yy / sqrt(syy*sff);
		double R2 = 1.0 - se / sff;
		double adjustedR2 = 1.0 - (se / (yy.size() - this->x_idx.size() - 1)) / (sff / (yy.size() - 1));

		//double chi_square = 0.0;
		//for (int i = 0; i < yy.size(); i++)
		//{
		//	double d = dd[i] / d_sum;
		//	double f = ff[i] / f_sum;
		//	chi_square += (d / (f + 1.e-10))*(d / (f + 1.e-10));
		//}

		//Chi_distribution chi_distribution(yy.size() -1);
		//double chi_pdf = chi_distribution.p_value(α);

		fprintf(fp, "Status:%d\n", getStatus());
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "SE(残差)                            :%.4f\n", se);
		fprintf(fp, "MSE                                 :%.4f\n", mse);
		fprintf(fp, "RMSE                                :%.4f\n", rmse);
		fprintf(fp, "r(相関係数)                         :%.4f\n", r);
		fprintf(fp, "R^2(決定係数(寄与率))               :%.4f\n", R2);
		fprintf(fp, "R^2(自由度調整済み決定係数(寄与率)) :%.4f\n", adjustedR2);
		fprintf(fp, "MER                                 :%.4f\n", median(mer));
		//fprintf(fp, "AIC                                 :%.3f\n", AIC);
		//fprintf(fp, "chi square       :%f\n", chi_square);
		//fprintf(fp, "p value          :%f\n", chi_pdf);
		fprintf(fp, "--------------------------------------------------------------------\n");

		if (classification >= 2)
		{
			get_accuracy(fp);
		}
		//if (chi_distribution.status != 0)
		//{
		//	fprintf(fp, "chi_distribution status:%d\n", chi_distribution.status);
		//}
		//if (chi_square > chi_pdf)
		//{
		//	fprintf(fp, "χ2値:%f > χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
		//	fprintf(fp, "=>良いフィッティングでしょう。\n予測に有効と思われます\n");
		//	fprintf(fp, "ただし、データ点範囲外での予測が正しい保証はありません。\n");
		//}
		//else
		//{
		//	fprintf(fp, "χ2値:%f < χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
		//	fprintf(fp, "=>良いとは言えないフィッティングでしょう。\n予測に有効とは言えないと思われます\n");
		//}
		if (fp != stdout) fclose(fp);
	}

	void visualize_observed_predict()
	{
		if (!visualize_observed_predict_plot) return;
#ifdef USE_GNUPLOT
		{
			int win_size[2] = { 640 * 3, 480 * 3 };
			std::vector<std::string> header_names(2);
			header_names[0] = "observed";
			header_names[1] = "predict";

			gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 10);
			if (capture)
			{
				plot1.set_capture(win_size, std::string("observed_predict_NL.png"));
			}
			Matrix<dnn_double> T(Diff.size()*Diff[0].size() / 2, 2);
			for (int i = 0; i < Diff.size(); i++)
			{
				for (int j = 0; j < Diff[0].size() / 2; j++)
				{
					T(i*Diff[0].size() / 2 + j, 0) = Diff[i][2 * j];
					T(i*Diff[0].size() / 2 + j, 1) = Diff[i][2 * j + 1];
				}
			}

			plot1.scatter_xyrange_setting = false;
			plot1.scatter(T, 0, 1, 1, 30, header_names, 5);
			if (10)
			{
				double max_x = T.Col(0).Max();
				double min_x = T.Col(0).Min();
				double step = (max_x - min_x) / 5.0;
				Matrix<dnn_double> x(6, 2);
				Matrix<dnn_double> v(1, 1);
				for (int i = 0; i < 6; i++)
				{
					v(0, 0) = min_x + i * step;
					x(i, 0) = v(0, 0);
					x(i, 1) = v(0, 0);
				}
				plot1.set_label(0.5, 0.85, 1, "observed=predict");
				plot1.plot_lines2(x, header_names);
				plot1.draw();
			}
		}
#endif
	}

};

#endif
