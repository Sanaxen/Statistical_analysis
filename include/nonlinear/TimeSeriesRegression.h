#ifndef _TimeSeriesRegression_H
#define _TimeSeriesRegression_H

#include "../../include/util/mathutil.h"
#include "../../include/nonlinear/MatrixToTensor.h"
#include <signal.h>


#ifdef USE_LIBTORCH
#include "../pytorch_cpp/tiny_dnn2libtorch_dll.h"
#pragma comment(lib, "../../pytorch_cpp/lib/rnn6.lib")
#endif

#define TIME_MEASUR
#ifdef TIME_MEASUR
#include <chrono>
#endif

#pragma warning( disable : 4305 ) 

#include "../../include/nonlinear/f_distribution.h"

#define SAMPLING_RANGE_MIN 0.01
#define SAMPLING_RANGE_MAX 0.5

#define EARLY_STOPPING_	10

#define MINBATCH_MAX	1024
#define TARGET_POSITON			(target_position-1)
//#define TARGET_POSITON			100


/* シグナル受信/処理 */
bool ctr_c_stopping_time_series_regression = false;
inline void SigHandler_time_series_regression(int p_signame)
{
	static int sig_catch = 0;
	if (sig_catch)
	{
		printf("割り込みです。終了します\n");
		ctr_c_stopping_time_series_regression = true;
		fflush(stdout);
		//return;
		exit(0);
	}
	sig_catch++;
	return;
}

inline char* timeToStr(dnn_double t, const std::string& format, char* str, size_t sz)
{
	if (format == "")
	{
		sprintf(str, "%.2f", t); return str;
	}

	memset(str, '\0', sizeof(char)*sz);
	time_t timer;
	struct tm *timeptr;

	//timer = time(NULL);
	timer =(time_t)t;
	timeptr = localtime(&timer);
	strftime(str, sz, format.c_str(), timeptr);
	//strftime(str, sz, "%Y/%m/%d[%H:%M:%S]", timeptr);
	return str;
}

inline void multiplot_gnuplot_script_ts(int ydim, int step, std::vector<std::string>& names, std::vector<int>& idx, std::string& timeformat, bool putImage, int confidence)
{
	if (!putImage)
	{
		for (int i = 0; 1; i++)
		{
			char fname[256];
			sprintf(fname, "multi_data%03d_ts.plt", i);

			FILE* fp = fopen(fname, "r");
			if (fp == NULL) break;
			if (fp) fclose(fp);
			remove(fname);

			sprintf(fname, "multi_data%03d_ts_png.plt", i);
			fp = fopen(fname, "r");
			if (fp)
			{
				fclose(fp);
				remove(fname);
			}

			sprintf(fname, "multi_data%03d_ts.png", i);
			fp = fopen(fname, "r");
			if (fp)
			{
				fclose(fp);
				remove(fname);
			}
		}
	}

	FILE* genImg = NULL;
	if (putImage)
	{
		genImg = fopen("gen_multi_data_ts2png.bat", "w");
	}

	int count = 0;
	for (int i = 0; 1; i++)
	{
		char fname[256];
		if (!putImage)
		{
			sprintf(fname, "multi_data%03d_ts.plt", i);
		}
		else
		{
			sprintf(fname, "multi_data%03d_ts_png.plt", i);
		}

		FILE* fp = fopen(fname, "w");
		if (fp == NULL) return;

		fprintf(fp, "bind \"Close\" \"if (GPVAL_TERM eq \'wxt\') bind \'Close\' \'\'; exit gnuplot; else bind \'Close\' \'\'; exit\"\n");

		fprintf(fp, "set term windows size %d,%d font \"arial,10\"\n", (int)(640 * 2.8), (int)(380 * 1.5)*step / 2);
		if (putImage)
		{
			fprintf(fp, "set term pngcairo size %d,%d font \"arial,10\"\n", (int)(640 * 2.8), (int)(380 * 1.5)*step / 2);
			fprintf(fp, "set output \"multi_data%03d_ts.png\"\n", i);
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
			fprintf(fp, "set style circle radius 0.7\n");
		}

		char* timex =
			"x_timefromat=%d\n"
			"if (x_timefromat != 0) set xdata time\n"
			"if (x_timefromat != 0) set timefmt \"%s\"\n"
			"if (x_timefromat != 0) set xtics timedate\n"
			"if (x_timefromat != 0) set xtics format \"%%Y/%%m/%%d\"\n";
		fprintf(fp, timex, timeformat!=""?1:0, timeformat.c_str());

		for (int k = 0; k < step; k++)
		{
			fprintf(fp, "set title %s\n", names[idx[count]].c_str());
			std::string plot =
				"file = \"test.dat\"\n"
				"plot file using 1:%d   t \"observation\"  with lines linewidth 2 lc \"#0068b7\", \\\n"
				"\"predict1.dat\" using 1:%d   t \"observation\"  with lines linewidth 2 lc \"#0080ff\", \\\n"
				"\"predict2.dat\" using 1:%d   t \"observation\"  with lines linewidth 2 lc \"#0068b7\", \\\n"
				"\"prophecy.dat\" using 1:%d   t \"prophecy\"  with lines linewidth 2 lc \"#0068b7\" dt 3,\\\n"
				"file using 1:%d   t \"predict\"  with lines linewidth 2 lc \"#009944\", \\\n"
				"\"predict1.dat\" using 1:%d   t \"predict\"  with lines linewidth 2 lc \"#ff8000\", \\\n"
				"\"predict2.dat\" using 1:%d   t \"predict\"  with lines linewidth 2 lc \"plum\", \\\n"
				"\"prophecy.dat\" using 1:%d   t \"prophecy\"  with lines linewidth 2 lc \"#e4007f\"";
			
			if (confidence)
			{
				plot = plot + ",\\\n\"confidence.dat\" using 1:%d:%d  with filledcurves fc \"#5f9ea0\" t \"95%% confidence\""
					",\\\n\"distribution.dat\"  using 1:%d with circles lc \"#5f9ea0\" fs transparent  solid 0.2  noborder\n\n";
				fprintf(fp, plot.c_str(), count * 2 + 2, count * 2 + 2, count * 2 + 2, count * 2 + 2, count * 2 + 3, count * 2 + 3, count * 2 + 3, count * 2 + 3,
					count*ydim +3, count*ydim +4,
					count+2);
			}
			else
			{
				plot = plot + "\n\n";
				fprintf(fp, plot.c_str(), count * 2 + 2, count * 2 + 2, count * 2 + 2, count * 2 + 2, count * 2 + 3, count * 2 + 3, count * 2 + 3, count * 2 + 3);
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
				//system((gnuplot_exe_path + " " + fname).c_str());
				//printf("%s\n", (gnuplot_exe_path + " " + fname).c_str());
				if ( genImg ) fprintf(genImg, "%s\n", (gnuplot_exe_path + " " + fname).c_str());
			}
		}
		if (count >= ydim) break;
	}
	if (genImg)
	{
		fclose(genImg);
		system("gen_multi_data_ts2png.bat");
	}
}

class TimeSeriesRegression
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
			FILE* fp = fopen("Writing_TimeSeriesRegression_", "w");
			fclose(fp);
			fp = NULL;
		}
		~writing()
		{
			unlink("Writing_TimeSeriesRegression_");
		}
	};

public:
	std::string device_name = "cpu";
	int use_libtorch = 0;
	bool use_logdiffernce = false;
	int use_differnce = 0;
	bool use_differnce_output_only = false;
	bool use_differnce_auto_inv = false;
	bool use_defined_scale = false;
	bool use_trained_scale = true;
	bool use_latest_observations = true;	//Always use the latest observations
	bool normalized = false;
	std::vector<std::string> header;
	std::vector<int> normalize_skilp;		// non normalize var
	double xx_var_scale = 1.0;			// non normalize var scaling
	std::vector<int> x_idx;
	std::vector<int> y_idx;
	std::string timeformat = "";
	bool fit_best_saved = false;
	bool state_reset_mode = false;
	bool batch_shuffle = true;
	int target_position = 1;
	int mean_row = 1;
	std::string activation_fnc = "tanh";
	bool use_attention = true;
	int n_sampling = 0;
	std::mt19937 mt_distribution;

	int padding_prm = 0;
	int residual = 0;

private:

	void normalizeZ(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
	{
		if (!normalized)
		{
			mean = std::vector<float_t>(X[0].size(), 0.0);
			sigma = std::vector<float_t>(X[0].size(), 1.0);

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

			if (normalize_skilp.size() > 0)
			{
				for (int k = y_dim; k < X[0].size(); k++)
				{
					if (normalize_skilp[k- y_dim])
					{
						sigma[k] = 1.0 / xx_var_scale;
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
			if (normalize_skilp.size() > 0)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (normalize_skilp[k])
					{
						min_[k] = 0.0;
						maxmin_[k] = 1.0 / xx_var_scale;
					}
				}
			}
		}
#endif

		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) / (maxmin_[k] + 1.0e-10);
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

			if (normalize_skilp.size() > 0)
			{
				for (int k = 0; k < X[0].size(); k++)
				{
					if (normalize_skilp[k])
					{
						maxmin_[k] = 2.0*(1.0 / xx_var_scale);
						min_[k] = -1.0 * (1.0 / xx_var_scale);
					}
				}
			}
		}
#endif
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) * 2 / (maxmin_[k]+1.0e-10) - 1;
			}
		}
	}

	float cost_min = std::numeric_limits<float>::max();
	float cost_min0 = std::numeric_limits<float>::max();
	float cost_pre = 0;
	float accuracy_max = std::numeric_limits<float>::min();
	float accuracy_pre = 0;

	tiny_dnn::tensor_t st_sum;
	tiny_dnn::tensor_t st_sigma2;
	int n_samples_cnt = 0;

	size_t net_test_no_Improvement_count = 0;
	void net_test(bool sampling_only = false)
	{
		writing lock;

#ifdef TIME_MEASUR
		std::chrono::system_clock::time_point  start, end;
		double elapsed;
#endif
		tiny_dnn::network2<tiny_dnn::sequential> nn_test;

		void* torch_nn_test = NULL;
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			if (test_mode)
			{
				torch_nn_test = torch_load_new("fit_best_ts.pt");
			}
			else
			{
				torch_nn_test = torch_load_new("tmp_ts.pt");
			}
		}
		else
#endif
		{
			nn_test = nn;
			if (test_mode)
			{
				nn_test.load("fit_best_ts.model");
			}
			else
			{
				nn_test.load("tmp_ts.model");
			}
		}
		//printf("layers:%zd\n", nn_test.depth());

		set_test(nn_test, 1);
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			state_reset(this->rnn_type, torch_nn_test);
		}
#endif

		FILE* fp_predict = NULL;
		if (!sampling_only)
		{
			if (test_mode)
			{
				fp_predict = fopen("predict_dnn.csv", "w");
				if (fp_predict)
				{
					if (iX.size())
					{
						for (int i = 0; i < iX[0].size(); i++)
						{
							fprintf(fp_predict, "%s,", header[x_idx[i]].c_str());
						}
					}
					if (classification >= 2)
					{
						fprintf(fp_predict, "predict[%s],", header[y_idx[0]].c_str());
						fprintf(fp_predict, "probability\n");
					}
					else
					{
						for (int i = 0; i < y_dim - 1; i++)
						{
							fprintf(fp_predict, "predict[%s],%s,dy,", header[y_idx[i]].c_str(), header[y_idx[i]].c_str());
						}
						fprintf(fp_predict, "predict[%s],%s,dy\n", header[y_idx[y_dim - 1]].c_str(), header[y_idx[y_dim - 1]].c_str());
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
			if (test_images.size())
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					loss_test = torch_get_loss_nn(torch_nn_test, test_images, test_labels, n_eval_minibatch);
				}
				else
#endif
				{
					float loss_test = get_loss(nn_test, test_images, test_labels);
				}
			}

			tiny_dnn::result train_result;
			tiny_dnn::result test_result;

#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				train_result = torch_get_accuracy_nn(torch_nn_test, train_images, train_labels, n_eval_minibatch);
				confusion_matrix_outoput(train_result);
				if (test_images.size() > 0)
				{
					test_result = torch_get_accuracy_nn(torch_nn_test, test_images, test_labels, n_eval_minibatch);
					confusion_matrix_outoput(test_result);
				}
			}
			else
#endif
			{
				train_result = get_accuracy(nn_test, train_images, train_labels);
				confusion_matrix_outoput(train_result);
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
				FILE* fp = fopen("timeseries_error_vari_loss.txt", "w");
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
					torch_save_nn(torch_nn_test, "fit_best_ts.pt");
					fit_best_saved = true;
					//torch_save_nn(torch_nn_test, "tmp_ts.pt");
					accuracy_max = train_result.accuracy();
				}
				else
#endif
				{
					nn_test.save("fit_best_ts.model");
					fit_best_saved = true;
					nn_test.save("fit_best.model_ts.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
					accuracy_max = train_result.accuracy();
				}
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

			if (fp_predict)
			{
				std::vector<tiny_dnn::vec_t> YY = nY;
				YY.resize(nY.size() + prophecy + sequence_length + out_sequence_length);
				for (int i = nY.size(); i < YY.size(); i++)
				{
					YY[i].resize(nY[0].size());
				}

				std::vector<tiny_dnn::vec_t> y_predict_n;
				std::vector<tiny_dnn::vec_t> seq(nY.size());
#ifdef USE_LIBTORCH
#pragma omp parallel for
				for (int i = 0; i < nY.size(); i++)
				{
					seq[i] = seq_vec(YY, i);
				}
				y_predict_n = torch_model_predict_batch(torch_nn_test, seq, n_eval_minibatch);
#endif
				for (int i = 0; i < iY.size(); i++)
				{
					tiny_dnn::vec_t y_predict;
#ifdef USE_LIBTORCH
					if (use_libtorch)
					{
						//y_predict = torch_model_predict(torch_nn_test, seq_vec(YY, i));
						y_predict = y_predict_n[i];
					}
					else
#endif

					{
						y_predict = nn_test.predict(seq_vec(YY, i));
					}

					if (fp_predict)
					{
						if (iX.size())
						{
							for (int k = 0; k < iX[0].size(); k++)
							{
								fprintf(fp_predict, "%.3f,", iX[i][k]);
							}
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

			set_train(nn, sequence_length, n_bptt_max, default_backend_type);
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_delete_load_model(torch_nn_test);
				torch_nn_test = NULL;
			}
#endif
			return;
		}


		{
			const size_t time_str_sz = 80;
			char time_str[time_str_sz];
			std::vector<tiny_dnn::vec_t> train(nY.size() + prophecy + sequence_length + out_sequence_length + TARGET_POSITON);
			std::vector<tiny_dnn::vec_t> predict(nY.size() + prophecy + sequence_length + out_sequence_length + TARGET_POSITON);
			std::vector<tiny_dnn::vec_t> YY = nY;
			YY.resize(nY.size() + prophecy + sequence_length + out_sequence_length + TARGET_POSITON);

			std::vector<double> timver_tmp(YY.size());
#ifdef TIME_MEASUR
			start = std::chrono::system_clock::now();
#endif

#pragma omp parallel
			{
				const size_t sz = nY[0].size();
#pragma omp for nowait
				for (int i = nY.size(); i < YY.size(); i++)
				{
					YY[i].resize(sz);
				}
#pragma omp for nowait
				for (int i = 0; i < YY.size(); i++)
				{
					train[i].resize(sz);
					predict[i].resize(sz);
				}

#pragma omp for nowait
				for (int i = 0; i < iY.size(); i++)
				{
					timver_tmp[i] = timevar(i, 0);
				}
				for (int i = iY.size(); i < YY.size(); i++)
				{
					timver_tmp[i] = timver_tmp[i - 1] + timevar(1, 0) - timevar(0, 0);
				}
			}
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed1:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif

			//最初のシーケンス分は入力でしか無いのでそのまま
			for (int j = 0; j < sequence_length + TARGET_POSITON; j++)
			{
				tiny_dnn::vec_t y(y_dim);
				for (int k = 0; k < y_dim; k++)
				{
					y[k] = nY[j][k];
				}
				train[j] = predict[j] = y;
			}

			const int sz = iY.size() + prophecy - use_differnce - TARGET_POSITON;

			std::vector<tiny_dnn::vec_t> y_predict_n;
			std::vector<tiny_dnn::vec_t> seq(train_images.size());
#ifdef USE_LIBTORCH
			if (use_latest_observations || !test_mode)
			{

#pragma omp parallel for
				for (int i = 0; i < train_images.size(); i++)
				{
					seq[i] = seq_vec(YY, i);
				}
				y_predict_n = torch_model_predict_batch(torch_nn_test, seq, n_eval_minibatch);
			}
			else
			{
				y_predict_n.clear();
			}
#endif
			for (int i = 0; i < sz; i++)
			{
				//i...i+sequence_length-1 -> 
				//  i+sequence_length ... i+sequence_length+out_sequence_length-1


				tiny_dnn::vec_t next_y;
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					if (i >= train_images.size() - sequence_length || y_predict_n.size() == 0)
					{
						next_y = torch_model_predict(torch_nn_test, seq_vec(YY, i));
					}
					else
					{
						next_y = y_predict_n[i];
					}
				}
				else
#endif
				{
					next_y = nn_test.predict(seq_vec(YY, i));
				}

				//output sequence_length 
				for (int j = 0; j < out_sequence_length; j++)
				{
					tiny_dnn::vec_t yy(y_dim);
					tiny_dnn::vec_t y(y_dim);
					for (int k = 0; k < y_dim; k++)
					{
						y[k] = YY[i + sequence_length + j + TARGET_POSITON][k];

						//need this
						if (i + sequence_length + j + TARGET_POSITON < train_images.size())
						{
							y[k] = nY[i + sequence_length + j + TARGET_POSITON][k];
						}
						yy[k] = next_y[y_dim*j + k];
					}
					train[i + sequence_length + j+ TARGET_POSITON] = y;
					predict[i + sequence_length + j+ TARGET_POSITON] = yy;
				}

				//From the first sequence onwards, all are predicted from predicted values
				if (!use_latest_observations /*&& test_mode*/)
				{
					for (int j = 0; j < out_sequence_length; j++)
					{
						for (int k = 0; k < y_dim; k++)
						{
							YY[i + sequence_length + j+ TARGET_POSITON][k] = predict[i + sequence_length + j+ TARGET_POSITON][k];
						}
					}
				}
				if (use_latest_observations/* || !test_mode*/)
				{
					if (i >= train_images.size() - sequence_length)
					{
						for (int j = 0; j < out_sequence_length; j++)
						{
							if (i + sequence_length + j + TARGET_POSITON >= train_images.size())
							{
								for (int k = 0; k < y_dim; k++)
								{
									YY[i + sequence_length + j+ TARGET_POSITON][k] = predict[i + sequence_length + j+ TARGET_POSITON][k];
								}
							}
						}
					}
				}
			}
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed2:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif
			{
				//denormalize
				const size_t sz = iY.size() + prophecy - use_differnce;

#pragma omp parallel for
				for (int i = 0; i < sz; i++)
				{
					for (int k = 0; k < y_dim; k++)
					{
						if (zscore_normalization)
						{
							predict[i][k] = predict[i][k] * Sigma[k] + Mean[k];
							train[i][k] = train[i][k] * Sigma[k] + Mean[k];
						}else
						if (minmax_normalization)
						{
							predict[i][k] = predict[i][k] * MaxMin[k] + Min[k];
							train[i][k] = train[i][k] * MaxMin[k] + Min[k];
						}else
						if (_11_normalization)
						{
							predict[i][k] = 0.5*(predict[i][k] + 1) * MaxMin[k] + Min[k];
							train[i][k] = 0.5*(train[i][k] + 1) * MaxMin[k] + Min[k];
						}
					}
				}

				if (use_differnce > 0 && use_differnce_auto_inv)
				{
					predict = diffinv_vec(iY, predict, this->normalize_skilp, use_differnce, use_logdiffernce);
					train = diffinv_vec(iY, train, this->normalize_skilp, use_differnce, use_logdiffernce);

					if (use_logdiffernce)
					{
						predict = exp(predict, this->normalize_skilp);
						train = exp(train, this->normalize_skilp);
					}
				}
				
				if (distribution_only || sampling_only)
				{
					try
					{
						std::ofstream ofs_distri("distribution.dat", std::ios::app);;
						for (int i = sequence_length + 1; i < sz; i++)
						{
							ofs_distri << i << " ";
							for (int k = 0; k < y_dim; k++)
							{
								ofs_distri << predict[i][k] << " ";
							}
							ofs_distri << std::endl;
						}
						ofs_distri.close();
					}
					catch (...)
					{
					}
					if (distribution_only && !sampling_only) return;
				}

				//distribution_ts
				if (/*this->n_sampling > 0*/ sampling_only )
				{
					//printf("n_samples_cnt:%d\n", n_samples_cnt); fflush(stdout);
					if (n_samples_cnt == 0)
					{
						st_sum.resize(sz, tiny_dnn::vec_t(y_dim, 0));
					}
					for (int i = 0; i < sz; i++)
					{
						for (int k = 0; k < y_dim; k++)
						{
							st_sum[i][k] += predict[i][k];
						}
					}

					if (n_samples_cnt > 1)
					{
						st_sigma2.resize(sz, tiny_dnn::vec_t(y_dim, 0));
						for (int j = 0; j < sz; j++)
						{
							for (int k = 0; k < y_dim; k++)
							{
								auto d = (predict[j][k] - st_sum[j][k] / n_samples_cnt);
								st_sigma2[j][k] += d * d / (n_samples_cnt -1 );
							}
						}

						FILE* fp = fopen("confidence.dat", "w");
						if (fp)
						{
							//片側p = 0.05, 0.025, 0.005, 0.0005-> 0 1 2 3 4
							const float alp = stats_t_ppf(1, n_samples_cnt);

							for (int j = 0; j < sz; j++)
							{
								fprintf(fp, "%d", j);
								for (int k = 0; k < y_dim; k++)
								{
									if (j < sequence_length)
									{
										fprintf(fp, " %f %f %f",
											train[j][k],
											train[j][k],
											train[j][k]);
									}
									else
									{
										fprintf(fp, " %f %f %f",
											st_sum[j][k] / n_samples_cnt,
											st_sum[j][k] / n_samples_cnt - alp * sqrt(st_sigma2[j][k] / n_samples_cnt),
											st_sum[j][k] / n_samples_cnt + alp * sqrt(st_sigma2[j][k] / n_samples_cnt));
									}
								}
								fprintf(fp, "\n");
							}
							fclose(fp);
						}
					}
					n_samples_cnt++;
					printf("n_samples_cnt:%d\n", n_samples_cnt);
					if (sampling_only) return;
				}
			}

#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed3:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif

			if (this->test_mode)
			{
				if (fp_predict)
				{
					for (int i = 0; i < iY.size(); i++)
					{
						if (iX.size())
						{
							for (int k = 0; k < iX[0].size(); k++)
							{
								fprintf(fp_predict, "%.3f,", iX[i][k]);
							}
						}

						tiny_dnn::vec_t yy = predict[i];
						tiny_dnn::vec_t y = train[i];
						for (int k = 0; k < y_dim-1; k++)
						{
							fprintf(fp_predict, "%.3f,%.3f,%.3f,", yy[k], y[k], yy[k]-y[k]);
						}
						fprintf(fp_predict, "%.3f,%.3f,%.3f\n", yy[y_dim-1], y[y_dim - 1], yy[y_dim - 1]- y[y_dim - 1]);
					}

					for (int i = iY.size(); i < iY.size() + prophecy- use_differnce; i++)
					{
						if (iX.size())
						{
							for (int k = 0; k < iX[0].size(); k++)
							{
								fprintf(fp_predict, "%.3f,", 0.0);
							}
						}
						tiny_dnn::vec_t yy = predict[i];
						tiny_dnn::vec_t y = train[iY.size()-1];
						for (int k = 0; k < y_dim - 1; k++)
						{
							fprintf(fp_predict, "%.3f,%.3f,%.3f,", yy[k], yy[k], 0.0);
						}
						fprintf(fp_predict, "%.3f, %.3f, %.3f\n", yy[y_dim - 1], yy[y_dim - 1], 0.0);
					}
				}
				if(fp_predict) fclose(fp_predict);


				FILE* fp_test = fopen("predict1.dat", "w");
				if (fp_test)
				{
					for (int i = 0; i < iY.size() + prophecy- use_differnce; i++)
					{
						tiny_dnn::vec_t y = train[i];
						tiny_dnn::vec_t yy = predict[i];

						//fprintf(fp_test, "%f ", timver_tmp[i]);
						//fprintf(fp_test, "%s ", timeToStr(timver_tmp[i], timeformat, time_str, time_str_sz));
						if (timestamp.size() > i)
						{
							fprintf(fp_test, "%s ", timestamp[i].c_str());
						}
						else
						{
							fprintf(fp_test, "%.3f ",timver_tmp[i]);
						}

						for (int k = 0; k < y_dim - 1; k++)
						{
							if (i < sequence_length + TARGET_POSITON)
							{
								fprintf(fp_test, "NaN %f ", yy[k]);
							}else
							if (i >= iY.size())
							{
								fprintf(fp_test, "NaN %f ", yy[k]);
							}
							else
							{
								fprintf(fp_test, "%f %f ", y[k], yy[k]);
							}
							//if (fp_predict)
							//{
							//	fprintf(fp_test, "%f,", yy[k]);
							//}
						}
						if (i >= iY.size())
							fprintf(fp_test, "NaN %f\n", yy[y_dim - 1]);
						else
							fprintf(fp_test, "%f %f\n", y[y_dim - 1], yy[y_dim - 1]);

						//if (fp_predict)
						//{
						//	fprintf(fp_test, "%f\n", yy[y_dim - 1]);
						//}
					}
					fclose(fp_test);
				}
			}
			else
			{
				FILE* fp_test = fopen("test.dat", "w");
				if (fp_test)
				{
					for (int i = 0; i < sequence_length+ TARGET_POSITON; i++)
					{
						tiny_dnn::vec_t y = train[i];
						//fprintf(fp_test, "%f ", timver_tmp[i]);
						//fprintf(fp_test, "%s ", timeToStr(timver_tmp[i], timeformat, time_str, time_str_sz));
						if (timestamp.size() > i)
						{
							fprintf(fp_test, "%s ", timestamp[i].c_str());
						}
						else
						{
							fprintf(fp_test, "%.3f ", timver_tmp[i]);
						}
						for (int k = 0; k < y_dim - 1; k++)
						{
							fprintf(fp_test, "%f %f ", y[k], y[k]);
						}
						fprintf(fp_test, "%f %f\n", y[y_dim - 1], y[y_dim - 1]);
					}
					fclose(fp_test);
				}


				fp_test = fopen("predict1.dat", "w");
				if (fp_test)
				{
					for (int i = sequence_length - 1; i < train_images.size() - sequence_length; i++)
					{
						tiny_dnn::vec_t y = train[i];
						tiny_dnn::vec_t yy = predict[i];

						//if (i < iY.size())
						//{
						//	y = nY[i];
						//}
						//fprintf(fp_test, "%f ", timver_tmp[i]);
						//fprintf(fp_test, "%s ", timeToStr(timver_tmp[i], timeformat, time_str, time_str_sz));
						if (timestamp.size() > i)
						{
							fprintf(fp_test, "%s ", timestamp[i].c_str());
						}
						else
						{
							fprintf(fp_test, "%.3f ", timver_tmp[i]);
						}
						for (int k = 0; k < y_dim - 1; k++)
						{
							fprintf(fp_test, "%f %f ", y[k], yy[k]);
						}
						fprintf(fp_test, "%f %f\n", y[y_dim - 1], yy[y_dim - 1]);
					}
					fclose(fp_test);
				}

				fp_test = fopen("predict2.dat", "w");
				if (fp_test)
				{
					for (int i = train_images.size() - sequence_length - 1; i < iY.size() - sequence_length; i++)
					{
						tiny_dnn::vec_t y = train[i];
						tiny_dnn::vec_t yy = predict[i];
						//if (i < iY.size())
						//{
						//	y = nY[i];
						//}
						//fprintf(fp_test, "%f ", timver_tmp[i]);
						//fprintf(fp_test, "%s ", timeToStr(timver_tmp[i], timeformat, time_str, time_str_sz));
						if (timestamp.size() > i)
						{
							fprintf(fp_test, "%s ", timestamp[i].c_str());
						}
						else
						{
							fprintf(fp_test, "%.3f ", timver_tmp[i]);
						}
						for (int k = 0; k < y_dim - 1; k++)
						{
							fprintf(fp_test, "%f %f ", y[k], yy[k]);
						}
						fprintf(fp_test, "%f %f\n", y[y_dim - 1], yy[y_dim - 1]);
					}
					fclose(fp_test);
				}

				fp_test = fopen("prophecy.dat", "w");
				if (fp_test)
				{
					for (int i = iY.size() - sequence_length - 1; i < iY.size() + prophecy - use_differnce; i++)
					{
						tiny_dnn::vec_t y = train[i];
						tiny_dnn::vec_t yy = predict[i];
						//if (i < iY.size())
						//{
						//	y = nY[i];
						//}
						//fprintf(fp_test, "%f ", timver_tmp[i]);
						//fprintf(fp_test, "%s ", timeToStr(timver_tmp[i], timeformat, time_str, time_str_sz));
						if (timestamp.size() > i)
						{
							fprintf(fp_test, "%s ", timestamp[i].c_str());
						}
						else
						{
							fprintf(fp_test, "%.3f ", timver_tmp[i]);
						}

						if (i >= iY.size())
						{
							for (int k = 0; k < y_dim - 1; k++)
							{
								fprintf(fp_test, "NaN %f ", yy[k]);
							}
							fprintf(fp_test, "NaN %f\n", yy[y_dim - 1]);
						}
						else
						{
							for (int k = 0; k < y_dim - 1; k++)
							{
								fprintf(fp_test, "%f %f ", y[k], yy[k]);
							}
							fprintf(fp_test, "%f %f\n", y[y_dim - 1], yy[y_dim - 1]);
						}
					}
					fclose(fp_test);
				}
			}
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed4:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif

			Diff.clear();
			Diff.resize(iY.size());
			float vari_cost = 0.0;
			float cost = 0.0;
			float cost_tot = 0.0;
			const size_t train_sz = train_images.size();

#pragma omp parallel for reduction(+:cost,vari_cost,cost_tot)
			for (int i = 0; i < iY.size(); i++)
			{
				std::vector<double> diff(y_dim*2);
				tiny_dnn::vec_t y = train[i];
				tiny_dnn::vec_t yy = predict[i];
				for (int k = 0; k < y_dim; k++)
				{
					double d = (yy[k] - y[k])*(yy[k] - y[k]);
					cost_tot += d;
					if (i < train_sz)
					{
						cost += d;
					}
					else
					{
						vari_cost += d;
					}
					diff[2*k] = y[k];
					diff[2*k+1] = yy[k];
				}
				Diff[i] = diff;
			}
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed5:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif

			if (this->test_mode)
			{
				Matrix<dnn_double> x;
				Matrix<dnn_double> y;
				train.resize(iY.size() - sequence_length - use_differnce);
				predict.resize(iY.size() - sequence_length - use_differnce);
				TensorToMatrix(train, x);
				TensorToMatrix(predict, y);
				Matrix<dnn_double> xx = x;

				try
				{
					auto& mean = x.Mean();
					x = x.whitening(mean, x.Std(mean) + 1.0e-10);
					mean = y.Mean();
					y = y.whitening(mean, y.Std(mean) + 1.0e-10);
					mean = xx.Mean();
					xx = xx.whitening(mean, xx.Std(mean) + 1.0e-10);

					const size_t time_str_sz = 80;
					char time_str[time_str_sz];

					x = MahalanobisDist_Abnormality(x.appendCol(y));

					FILE* fp_test = fopen("mahalanobis_dist.csv", "w");
					if (fp_test)
					{
						fprintf(fp_test, "T,measured,predict,mahalanobis_dist\n");
						for (int i = 0; i < train.size(); i++)
						{
							//fprintf(fp_test, "%f,%f,%f\n", timver_tmp[i], y(i,0), x(i,0));
							if (timestamp.size() > i)
							{
								fprintf(fp_test, "%s, ", timestamp[i].c_str());
							}
							else
							{
								fprintf(fp_test, "%.3f, ", timver_tmp[i]);
							}
							fprintf(fp_test, "%f,%f,%f\n", xx(i, 0), y(i, 0), x(i, 0));
						}
						fclose(fp_test);
					}

					if (0)
					{
						gnuPlot plot1 = gnuPlot(std::string(GNUPLOT_PATH), 6);

						int win_size[] = { 640,480 };
						if (capture)
						{
							plot1.set_capture(win_size, std::string("timeSeries_scatter.png"));
						}
						int grid = 30;
						float pointsize = 1.0;
						char* palette = NULL;
						std::vector<std::string> header_names(2);
						header_names[0] = "train";
						header_names[1] = "predict";
						plot1.scatter(xx, 0, 1, pointsize, grid, header_names, 5, palette);
						if (palette != NULL)
						{
							plot1.set_palette(palette);
						}
						{
							for (float t = 0.05; t < 0.5; t += 0.1)
							{
								plot1.probability_ellipse(xx, 0, 1, t);
							}
						}
						plot1.draw();
					}
				}
				catch (...)
				{
					printf("MahalanobisDist error.\n");
					fflush(stderr);
				}
			}
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed6:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif

			cost /= (iY.size() - sequence_length);
			vari_cost /= (iY.size() - train_images.size());
			cost_tot /= iY.size();
			//printf("%f %f\n", cost_min, cost_tot);
			if (cost_tot < cost_min)
			{
#ifdef USE_LIBTORCH
				if (use_libtorch)
				{
					torch_save("fit_best_ts.pt");
				}
				else
#endif
				{
					nn_test.save("fit_best_ts.model");
					fit_best_saved = true;
					nn_test.save("fit_best.model_ts.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
				}
				//printf("!!=========== best model save ============!!\n");
				cost_min = cost_tot;
			}
			if (cost_min < tolerance)
			{
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

			{
				FILE* fp = fopen("timeseries_error_vari_loss.txt", "w");
				if (fp)
				{
					fprintf(fp, "best.model loss:%f\n", cost_min);
					fprintf(fp, "total loss:%f\n", cost_tot);
					fprintf(fp, "validation loss:%f\n", vari_cost);
					fclose(fp);
				}
			}
			cost_pre = cost_tot;

#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed7:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif
			set_train(nn, sequence_length, n_bptt_max, default_backend_type);
			visualize_observed_predict();
#ifdef TIME_MEASUR
			end = std::chrono::system_clock::now();
			elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			printf("elapsed8:%f\n", elapsed);
			start = std::chrono::system_clock::now();
#endif
		}
#ifdef USE_LIBTORCH
		torch_delete_load_model(torch_nn_test);
#endif
	}

public:
	bool distribution_only = false;
	void gen_visualize_fit_state(bool sampling_only = false)
	{
		set_test(nn, 1);
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			state_reset(this->rnn_type, torch_getNet());
		}
#endif

#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			torch_save("tmp_ts.pt");
		}
		else
#endif
		{
			nn.save("tmp_ts.model");
		}

		net_test(sampling_only);
		set_train(nn, sequence_length, n_bptt_max, default_backend_type);

#ifdef USE_GNUPLOT
		//if (capture)
		//{
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
	float_t support_length = 1.0;
	int epoch = 1;
	int plot_count = 0;
	int batch = 0;
	bool early_stopp = false;

public:
	float class_minmax[2] = { 0,0 };
	float dropout = 0;
	int classification = -1;
	bool visualize_observed_predict_plot = false;
	std::vector < std::vector<double>> Diff;
	Matrix<dnn_double> timevar;
	std::vector<std::string> timestamp;

	size_t x_dim;
	size_t y_dim;
	size_t prophecy = 0;
	size_t freedom = 0;
	bool minmax_normalization = false;
	bool zscore_normalization = false;
	bool _11_normalization = false;
	bool early_stopping = true;
	bool test_mode = false;
	bool capture = false;
	bool progress = true;
	float tolerance = 1.0e-6;
	int use_cnn = 1;
	int use_cnn_add_bn = 1;
	int fc_hidden_size = -1;
	std::string weight_init_type = "xavier";
	bool layer_graph_only = false;

	tiny_dnn::core::backend_t default_backend_type = tiny_dnn::core::backend_t::internal;
	//tiny_dnn::core::backend_t default_backend_type = tiny_dnn::core::backend_t::intel_mkl;

	tiny_dnn::tensor_t iX;
	tiny_dnn::tensor_t iY;
	//tiny_dnn::tensor_t nX;
	tiny_dnn::tensor_t nY;
	std::vector<float_t> Mean;
	std::vector<float_t> Sigma;
	std::vector<float_t> Min;
	std::vector<float_t> MaxMin;
	tiny_dnn::network2<tiny_dnn::sequential> nn;
	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	float_t clip_gradients = 0;
	std::string opt_type = "adam";
	std::string rnn_type = "lstm";
	size_t input_size = 32;
	size_t sequence_length = 100;
	size_t out_sequence_length = 1;

	size_t n_minibatch = 30;
	size_t n_eval_minibatch = 30;
	size_t n_train_epochs = 3000;
	float_t learning_rate = 1.0;
	size_t n_bptt_max = 1e9;
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

		FILE* fp = fopen("normalize_info_t.txt", "w");
		if (fp)
		{
			fprintf(fp, "%s\n", normalize_type.c_str());
			fprintf(fp, "%d\n", nY[0].size());
			for (int i = 0; i < y_dim; i++)
			{
				fprintf(fp, "目的変数(%d)平均.分散:%.16f %.16f\n", i, Mean[i], Sigma[i]);
				fprintf(fp, "目的変数(%d)Min.Max:%.16f %.16f\n", i, Min[i], MaxMin[i] + Min[i]);
			}
			for (int i = y_dim; i < nY[0].size(); i++)
			{
				fprintf(fp, "説明変数(%d)平均.分散:%.16f %.16f\n", i, Mean[i], Sigma[i]);
				fprintf(fp, "説明変数(%d)Min.Max:%.16f %.16f\n", i, Min[i], MaxMin[i]+ Min[i]);
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

		//if (!test_mode) return;

		FILE* fp = fopen("normalize_info_t.txt", "r");
		if (fp)
		{
			char buf[256];
			char dummy[128];
			double a = 0.0, b = 0.0;
			int d = 0;
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
			sscanf(buf, "%d\n", &d);
			if (d != nY[0].size())
			{
				printf("ERROR:dimension miss match!\n");
				exit(0);
			}
			Mean = std::vector<float_t>(d, 0.0);
			Sigma = std::vector<float_t>(d, 0.0);
			Min = std::vector<float_t>(d, 0.0);
			MaxMin = std::vector<float_t>(d, 0.0);
			for (int i = 0; i < y_dim; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "目的変数(%d)平均.分散:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Mean[i] = a;
				Sigma[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "目的変数(%d)Min.Max:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Min[i] = a;
				MaxMin[i] = b-a;
			}
			for (int i = y_dim; i < d; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "説明変数(%d)平均.分散:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Mean[i] = a;
				Sigma[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "説明変数(%d)Min.Max:%lf %lf\n", &tmp, &a, &b);
				printf(buf);
				Min[i] = a;
				MaxMin[i] = b - a;
			}
			fclose(fp);
			printf("load scaling data\n");
		}
		normalized = true;
	}

	TimeSeriesRegression(tiny_dnn::tensor_t& X, tiny_dnn::tensor_t& Y, std::vector<int>& normalize_skilp_, float scale = 1.0, int ydim = 1, int xdim = 1, std::string normalize_type = "", int classification_ = -1, bool test_mode_ = false, bool use_trained_scale_ = true, bool use_defined_scale_ = false, int use_differnce_=0, bool use_logdiffernce_ = true, bool use_differnce_output_only_ = false)
	{
		use_differnce_output_only = use_differnce_output_only_;
		use_differnce = use_differnce_;
		use_logdiffernce = use_logdiffernce_;
		use_defined_scale = use_defined_scale_;
		use_trained_scale = use_trained_scale_;
		test_mode = test_mode_;

		normalize_skilp = normalize_skilp_;
		xx_var_scale = scale;

		y_dim = ydim;
		x_dim = xdim;
		iX = X;
		iY = Y;
		nY = Y;
		if (normalize_type == "zscore") zscore_normalization = true;
		if (normalize_type == "minmax") minmax_normalization = true;
		if (normalize_type == "[-1..1]") _11_normalization = true;

		classification = classification_;
		printf("classification:%d\n", classification);

		if (use_differnce > 0)
		{
			if (use_logdiffernce)
			{
				nY = log(nY, this->normalize_skilp);
			}
			nY = diff_vec(nY, this->normalize_skilp, use_differnce);

			Matrix<dnn_double> tmp;
			TensorToMatrix(nY, tmp);
			tmp.print_csv("differnce_.csv");
			if (use_differnce_output_only)
			{
				return;
			}
		}


		if (_11_normalization)
		{
			normalize_info_load(normalize_type);

			normalize1_1(nY, Min, MaxMin);
			printf("[-1,1] normalization\n");

			//get Mean, Sigma
			auto dmy = iY;
			normalizeZ(dmy, Mean, Sigma);

			if (!test_mode) normalize_info_save(normalize_type);

			if (classification > 0)
			{
				printf("ERROR:no!! [-1, 1] normalization");
				exit(0);
			}
		}
		if (minmax_normalization)
		{
			normalize_info_load(normalize_type);

			normalizeMinMax(nY, Min, MaxMin);
			printf("minmax_normalization\n");

			//get Mean, Sigma
			auto dmy = iY;
			normalizeZ(dmy, Mean, Sigma);

			if (!test_mode) normalize_info_save(normalize_type);

			if (classification > 0)
			{
				nY = iY;
				for (int k = 0; k < nY[0].size(); k++)
				{
					Min[k] = 0.0;
					MaxMin[k] = 1.0;
				}
			}
		}
		if (zscore_normalization)
		{
			normalize_info_load(normalize_type);
			normalizeZ(nY, Mean, Sigma);
			printf("zscore_normalization\n");

			auto dmy = iY;
			normalizeMinMax(dmy, Min, MaxMin);

			if (!test_mode) normalize_info_save(normalize_type);

			if (classification > 0)
			{
				nY = iY;
				for (int k = 0; k < nY[0].size(); k++)
				{
					Mean[k] = 0.0;
					Sigma[k] = 1.0;
				}
			}
			if (classification >= 0)
			{
				bool class_id_number = true;
				int class_num = classification;
				float class_min = 10000000;
				float class_max = -1;
				for (int i = 0; i < nY.size(); i++)
				{
					for (int k = 0; k < y_dim; k++)
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
				if (class_min > class_max)
				{
					error = -1;
					return;
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
						for (int k = 0; k < y_dim; k++)
						{
							nY[i][k] = (int)((class_num - 1)*(nY[i][k] - class_min) / (class_max - class_min));
							nY[i][k] = std::min((float_t)(class_num - 1), std::max(float_t(0.0), float_t(nY[i][k])));
							iY[i][k] = nY[i][k];
						}
					}
					std::ofstream stream("classification_warning.txt");
					if (!stream.bad())
					{
						stream << class_minmax[0] << "---" << class_minmax[1] << std::endl;
						double dt = class_num / (class_max - class_min);
						for (int i = 0; i < class_num; i++)
						{
							stream << "class index:" << i << " (class number:" << i + 1 << ") " << (i*dt + class_min) << " " << (i + 1)*dt + class_min << std::endl;
						}
						for (int i = 0; i < class_num; i++)
						{
							stream << i + 1 << " " << (i*dt + class_min) << " " << (i + 1)*dt + class_min << std::endl;
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
		if (Mean.size() == nY[0].size())
		{
			for (int i = 0; i < nY[0].size(); i++)
			{
				printf("%.16f %.16f\n", Mean[i], Sigma[i]);
				printf("%.16f %.16f\n", Min[i], MaxMin[i]);
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

	void add_seq(tiny_dnn::vec_t& y, tiny_dnn::vec_t& Y)
	{
#if 10
		const size_t sz = Y.size();
		Y.resize(sz + y.size());
		for (int i = 0; i < y.size(); i++)
		{
			Y[sz+i] = y[i];
		}
#else
		for (int i = 0; i < y.size(); i++)
		{
			Y.push_back(y[i]);
		}
#endif

	}
	tiny_dnn::vec_t seq_vec(tiny_dnn::tensor_t& ny, const int start)
	{
		const int dim = ny[0].size();

		tiny_dnn::vec_t seq(sequence_length*dim);
		for (int k = 0; k < sequence_length; k++)
		{
			if (ny.size() <= start + k)
			{
				printf("range over");
				throw tiny_dnn::nn_error("seq_vec:range over");
			}
			//add_seq(ny[start + k], seq);
			for (int i = 0; i < dim; i++)
			{
				seq[k*dim + i] = ny[start + k][i];
			}
		}
		return seq;
	}

	int data_set(float test = 0.3f)
	{
		train_images.clear();
		train_labels.clear();
		test_images.clear();
		test_images.clear();

		printf("n_minibatch:%d sequence_length:%d\n", n_minibatch, sequence_length);
		printf("out_sequence_length:%d\n", out_sequence_length);

		int dataAll = iY.size() - sequence_length - out_sequence_length - use_differnce - TARGET_POSITON;
		printf("dataset All:%d->", dataAll);

		if (dataAll <= 0 && test_mode)
		{
			printf("ERROR:Too many min_batch or Sequence length\n");
			error = -1;
			return error;
		}
		size_t test_Num = dataAll*test;
		int datasetNum = dataAll - test_Num;

		//datasetNum = datasetNum - datasetNum % n_minibatch;
		if (!test_mode)
		{
			if (datasetNum == 0 || datasetNum < this->n_minibatch)
			{
				printf("ERROR:Too many min_batch or Sequence length\n");
				error = -1;
				return error;
			}
		}
		else
		{
			if (datasetNum == 0)
			{
				printf("ERROR:Too many Sequence length\n");
				error = -1;
				return error;
			}
		}
		size_t train_num_max = datasetNum;
		if (datasetNum < 0)
		{
			printf("ERROR:Insufficient data length\n");
			error = -1;
			return error;
		}
		printf("train:%d test:%d\n", datasetNum, test_Num);

		for (int i = 0; i < train_num_max; i++)
		{
			train_images.push_back(seq_vec(nY, i));


			tiny_dnn::vec_t y;
			for (int j = 0; j < out_sequence_length; j++)
			{
				const auto& ny = nY[i + sequence_length + j + TARGET_POSITON];
				for (int k = 0; k < y_dim; k++)
				{
					y.emplace_back(ny[k]);
				}
			}

			if (classification >= 2)
			{
				train_labels.push_back(label2tensor(y[0], classification));
			}
			else
			{
				train_labels.push_back(y);
			}
		}

		for (int i = train_num_max; i < dataAll; i++)
		{
			test_images.push_back(seq_vec(nY, i));

			tiny_dnn::vec_t y;
			for (int j = 0; j < out_sequence_length; j++)
			{
				const auto& ny = nY[i + sequence_length + j + TARGET_POSITON];
				for (int k = 0; k < y_dim; k++)
				{
					y.emplace_back(ny[k]);
				}
			}

			if (classification >= 2)
			{
				test_labels.push_back(label2tensor(y[0], classification));
			}
			else
			{
				test_labels.push_back(y);
			}
		}
		printf("train:%d test:%d\n", train_images.size(), test_images.size());
		return 0;
	}

	void construct_net(int n_rnn_layers = 2, int n_layers = 2, int n_hidden_size = -1)
	{
		SigHandler_time_series_regression(0);
		signal(SIGINT, SigHandler_time_series_regression);
		signal(SIGTERM, SigHandler_time_series_regression);
		signal(SIGBREAK, SigHandler_time_series_regression);
		signal(SIGABRT, SigHandler_time_series_regression);

		using tanh = tiny_dnn::activation::tanh;
		using recurrent = tiny_dnn::recurrent_layer;

		int hidden_size = n_hidden_size;
		if (hidden_size <= 0) hidden_size = 64;

		input_size = train_images[0].size();

		// clip gradients
		tiny_dnn::recurrent_layer_parameters params;
		params.clip = clip_gradients;	// 1～5?

		if (n_bptt_max == 0) n_bptt_max = sequence_length;
		params.bptt_max = n_bptt_max;
		printf("bptt_max:%d\n", n_bptt_max);

		size_t in_w = train_images[0].size();
		size_t in_h = 1;
		size_t in_map = 1;

		LayerInfo layers(in_w, in_h, in_map);
		nn << layers.add_input(input_size);

		size_t usize = input_size * 1;
		nn << layers.add_fc(usize, false);
		if (this->activation_fnc == "tanh") nn << layers.tanh();
		if (this->activation_fnc == "relu") nn << layers.relu();

		if (use_cnn > 0)
		{
#if 10
			const int cnn_win_size = 3;
			const int pool_size = 2;
			size_t sz = 0;
			bool error = false;
			printf("////////////////////////\n");
			do {

				LayerInfo tmp_layers(nn.out_data_size(), 1, 1);
				tiny_dnn::network2<tiny_dnn::sequential> tmp_nn;

				tmp_nn << tmp_layers.add_fc(usize);
				if (this->activation_fnc == "tanh") tmp_nn << tmp_layers.tanh();
				if (this->activation_fnc == "relu") tmp_nn << tmp_layers.relu();

				error = false;
				try {
					for (int i = 0; i < use_cnn; i++)
					{
						if (tmp_nn.out_data_size() < 2)
						{
							error = true;
							break;
						}
						tmp_nn << tmp_layers.add_cnv(1, cnn_win_size, 1, 1, 1, tiny_dnn::padding::valid);
						if (tmp_nn.out_data_size() < 2)
						{
							error = true;
							break;
						}
						////Aliasing occurs in downsampling and high frequency components are impaired
						////Anomaly Detection in Control Valves by 1d CNN-LSTM

						//tmp_nn << tmp_layers.add_maxpool(pool_size, 1, 2, 1, tiny_dnn::padding::valid);
					}
					sz = tmp_nn.out_data_size();
				}catch(...)
				{ }
				usize += 10;
			} while (sz < 2 || error);
			printf("////////////////////////\n\n");
#endif
			nn << layers.add_fc(usize);
			if (this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();

			for (int i = 0; i < use_cnn; i++)
			{
				nn << layers.add_cnv(1, cnn_win_size, 1, 1, 1, tiny_dnn::padding::valid);
				if (this->activation_fnc == "tanh") nn << layers.tanh();
				if (this->activation_fnc == "relu") nn << layers.relu();
				
				////Aliasing occurs in downsampling and high frequency components are impaired
				////Anomaly Detection in Control Valves by 1d CNN-LSTM
				
				//nn << layers.add_maxpool(pool_size, 1, 2, 1, tiny_dnn::padding::valid);
				//nn << layers.add_dropout(0.25);
			}
			nn << layers.add_fc(input_size, false);
		}
		else
		{
			nn << layers.add_fc(input_size, false);
			if (this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();
		}

#if 0
		//printf("n_rnn_layers:%d\n", n_rnn_layers);
		for (int i = 0; i < n_rnn_layers; i++) 
		{
			nn << layers.add_fc(input_size/*, false*/);
			//nn << layers.add_batnorm(nn.template at<tiny_dnn::layer>(nn.layer_size() - 1), 0.00001, 0.9);
			nn << layers.add_rnn(rnn_type, hidden_size, sequence_length, params);
			input_size = hidden_size;
			//Scaled Exponential Linear Unit. (Klambauer et al., 2017)
			//nn << layers.selu_layer();
			nn << layers.tanh();
			//nn << layers.relu();
		}
#else
		//printf("n_rnn_layers:%d\n", n_rnn_layers);
		for (int i = 0; i < n_rnn_layers; i++)
		{
			//nn << layers.add_batnorm(nn.template at<tiny_dnn::layer>(nn.layer_size() - 1), 0.00001, 0.9);
			nn << layers.add_rnn(rnn_type, hidden_size, sequence_length, params);
			input_size = hidden_size;
		}
#endif


		int n_layers_tmp = n_layers;
		size_t sz = hidden_size / 2;

		sz = train_labels[0].size() * 10;
		for (int i = 0; i < n_layers_tmp; i++) {
			if (dropout && i == n_layers_tmp-1) nn << layers.add_dropout(dropout);
			if (fc_hidden_size > 0)
			{
				nn << layers.add_fc(fc_hidden_size);
			}
			else
			{
				nn << layers.add_fc(sz);
			}
			if (this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();
		}
		nn << layers.add_fc(sz);
		if (this->activation_fnc == "tanh") nn << layers.tanh();
		if (this->activation_fnc == "relu") nn << layers.relu();

		if (classification >= 2)
		{
			if (dropout) nn << layers.add_dropout(dropout);
			nn << layers.add_fc(std::min((int)input_size, classification * 2));
			if (this->activation_fnc == "tanh") nn << layers.tanh();
			if (this->activation_fnc == "relu") nn << layers.relu();
			nn << layers.add_fc(classification);
			nn << layers.softmax(classification);
		}
		else
		{
			nn << layers.add_fc(train_labels[0].size());
			nn << layers.add_linear(train_labels[0].size());
			printf("output size:%d\n", train_labels[0].size());
		}

#if 10
#ifdef CNN_USE_AVX
		for (auto n : nn)
		{
			if (n->layer_type() == "fully-connected")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::avx);
			}
			if (n->layer_type() == "recurrent-layer")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::avx);
			}
			if (n->layer_type() == "lstm-cell")
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
			if (n->layer_type() == "recurrent-layer")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::avx);
			}
			if (n->layer_type() == "lstm-cell")
			{
				n->set_backend_type(tiny_dnn::core::backend_t::avx);
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
		//printf("dot -Tgif graph_net.txt -o graph.gif\n");
#endif
	}


	void fit(int seq_length = 30, int rnn_layers = 2, int n_layers = 2, int n_hidden_size = -1)
	{
		if (seq_length < 0) seq_length = 30;
		if (rnn_layers < 0) rnn_layers = 2;
		if (n_layers < 0) n_layers = 2;

		mt_distribution = std::mt19937(1234);
		sequence_length = seq_length;
		nn.set_input_size(train_images.size());
		using train_loss = tiny_dnn::mse;

	
		{
			char *param = "train_params.txt";
			if (test_mode) param = "test_params.txt";
			FILE* fp = fopen(param, "w");
			fprintf(fp, "test_mode:%d\n", test_mode);
			fprintf(fp, "learning_rate:%f\n", learning_rate);
			fprintf(fp, "opt_type:%s\n", opt_type.c_str());
			fprintf(fp, "n_train_epochs:%d\n", n_train_epochs);
			fprintf(fp, "n_minibatch:%d\n", n_minibatch);
			fprintf(fp, "n_eval_minibatch:%d\n", n_eval_minibatch);
			fprintf(fp, "x_dim:%d\n", x_dim);
			fprintf(fp, "y_dim:%d\n", y_dim);
			fprintf(fp, "sequence_length:%d\n", sequence_length);
			fprintf(fp, "out_sequence_length:%d\n", this->out_sequence_length);
			fprintf(fp, "target_position:%d\n", this->target_position);
			fprintf(fp, "clip_gradients:%f\n", this->clip_gradients);
			fprintf(fp, "early_stopping:%d\n", this->early_stopping?1:0);
			fprintf(fp, "rnn_layers:%d\n", rnn_layers);
			fprintf(fp, "n_layers:%d\n", n_layers);
			fprintf(fp, "n_hidden_size:%d\n", n_hidden_size);
			fprintf(fp, "fc_hidden_size:%d\n", fc_hidden_size);
			fprintf(fp, "dropout:%f\n", dropout);
			fprintf(fp, "prophecy:%d\n", prophecy);
			fprintf(fp, "tolerance:%f\n", tolerance);
			fprintf(fp, "rnn_type:%s\n", rnn_type.c_str());
			fprintf(fp, "state_reset_mode:%d\n", state_reset_mode);
			fprintf(fp, "classification:%d\n", classification);
			fprintf(fp, "batch_shuffle:%d\n", batch_shuffle);
			fprintf(fp, "weight_init_type:%s\n", this->weight_init_type.c_str());
			fprintf(fp, "activation_fnc:%s\n", this->activation_fnc.c_str());
			fprintf(fp, "use_attention:%d\n", this->use_attention?1:0);
			fprintf(fp, "use_cnn:%d\n", this->use_cnn);
			fprintf(fp, "use_cnn_add_bn:%d\n", this->use_cnn_add_bn);
			fprintf(fp, "padding_prm:%d\n", this->padding_prm);
			fprintf(fp, "residual:%d\n", this->residual);
			fclose(fp);
			
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
 
		tiny_dnn::adam optimizer_adam;
		tiny_dnn::gradient_descent optimizer_sgd;
		tiny_dnn::RMSprop optimizer_rmsprop;
		tiny_dnn::adagrad optimizer_adagrad;

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

		construct_net(rnn_layers, n_layers, n_hidden_size);
		if (layer_graph_only)
		{
			return;
		}

		if (opt_type == "adam") optimizer_adam.reset();
		if (opt_type == "sgd" )	optimizer_sgd.reset();
		if (opt_type == "rmsprop" )	optimizer_rmsprop.reset();
		if (opt_type == "adagrad")optimizer_adagrad.reset();

		tiny_dnn::timer finish_predict;
		tiny_dnn::timer t;

		float_t loss_min = std::numeric_limits<float>::max();
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
			if (this->batch_shuffle)
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
			if (plot && epoch % plot == 0)
			{
				gen_visualize_fit_state();
#ifdef USE_LIBTORCH
				if (this->n_sampling > 0)
				{
					distribution_only = true;

					std::uniform_real_distribution r(SAMPLING_RANGE_MIN, SAMPLING_RANGE_MAX);
					set_sampling(r(mt_distribution));
					gen_visualize_fit_state(true);
					reset_sampling();

					distribution_only = false;
			}
#endif
		}


			if (convergence)
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
			
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				state_reset(this->rnn_type, torch_getNet());
			}
			else
#endif
			{
				rnn_state_reset(nn);
			}
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

			//if (plot && (batch+1) % plot == 0)
			//{
			//	gen_visualize_fit_state();
			//}
			if (convergence)
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
			if (ctr_c_stopping_time_series_regression)
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
			else
			{
				/* empty*/
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
				torch_train(
					train_images,
					train_labels,
					n_minibatch,
					n_train_epochs,
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
					torch_save("fit_best_ts.pt");
					fit_best_saved = true;
				}
				else
#endif
				{
					nn.save("fit_best_ts.model");
					fit_best_saved = true;
					nn.save("fit_best.model_ts.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
				}
			}
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				torch_load("fit_best_ts.pt");
			}
			else
#endif
			{
				nn.load("fit_best_ts.model");
			}
			gen_visualize_fit_state();
			if (test_mode)
			{
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
		}
		catch (tiny_dnn::nn_error& msg)
		{
			printf("%s\n", msg.what());
			printf("fit_best.model open error.\n");
		}
		catch (std::exception &err)
		{
			printf("%s\n", err.what());
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
			nn.save(model_file);
			nn.save(model_file + ".json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
		}

	}

	//tiny_dnn::vec_t predict_next(tiny_dnn::vec_t& pre)
	//{
	//	set_test(nn, 1);
	//	tiny_dnn::vec_t& y_predict = nn.predict(pre);
	//	for (int k = 0; k < y_predict.size(); k++)
	//	{
	//		if (zscore_normalization)
	//		{
	//			y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
	//		}
	//		if (minmax_normalization)
	//		{
	//			y_predict[k] = y_predict[k] * MaxMin[k] + Min[k];
	//		}
	//		if (_11_normalization)
	//		{
	//			y_predict[k] = 0.5*(y_predict[k] + 1) * MaxMin[k] + Min[k];
	//		}
	//	}
	//	return y_predict;
	//}

	double get_accuracy(FILE* fp = NULL)
	{
		if (classification < 2) return 0.0;
		if (fp == NULL) fp = stdout;

		double accuracy = 0;
		double tot = 0;

		for (int i = 1; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t y = nn.predict(train_images[i - 1]);
			for (int k = 0; k < y.size(); k++)
			{
				auto z = train_labels[i][k];

				//if (zscore_normalization)
				//{
				//	y[k] = y[k] * Sigma[k] + Mean[k];
				//	z = z * Sigma[k] + Mean[k];
				//}
				//if (minmax_normalization)
				//{
				//	y[k] = y[k] * MaxMin[k] + Min[k];
				//	z = z * MaxMin[k] + Min[k];
				//}

				float_t d;
				if (classification == 2)
				{
					if (y[k] < 0.5) d = 0.0;
					else d = 1.0;

					if (d == z)
					{
						accuracy++;
					}
				}
				tot++;
			}
			fprintf(fp, "accuracy:%.3f%%\n", 100.0*accuracy / tot);
		}
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
		std::vector<tiny_dnn::label_t> tmp_labels(train_images.size());
		std::vector<tiny_dnn::vec_t> tmp_images(train_images.size());

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
			tmp_images[i] = train_images[i];
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

		//std::ofstream ofs("ConfusionMatrix.txt");
		//result.print_detail(ofs);
		//{
		//	FILE* fp = fopen("ConfusionMatrix.txt", "r");
		//	FILE* fp2 = fopen("ConfusionMatrix.r", "w");
		//	char* p = NULL;
		//	if (fp && fp2)
		//	{
		//		fprintf(fp2, "confusionMatrix <- data.frame(");

		//		char buf[4096];
		//		int nn = 0;
		//		fgets(buf, 4096, fp);
		//		fgets(buf, 4096, fp);
		//		fgets(buf, 4096, fp);
		//		do
		//		{
		//			int n = 0;
		//			p = buf;
		//			while (isspace(*p)) p++;

		//			fprintf(fp2, "c(");
		//			while (*p != '\n'&& *p != '\0')
		//			{
		//				if (n > 1) fprintf(fp2, ",");
		//				do
		//				{
		//					if (n >= 1)fprintf(fp2, "%c", *p);
		//					p++;
		//				} while (!isspace(*p));
		//				while (isspace(*p)) p++;
		//				n++;
		//				if (*p == '\n')break;
		//			}
		//			if (n >= 1)fprintf(fp2, ")");
		//			p = fgets(buf, 4096, fp);
		//			if (p != NULL)fprintf(fp2, ",");
		//			nn++;
		//		} while (p != NULL);
		//		fprintf(fp2, ")\r\n");

		//		fprintf(fp2, "colnames(confusionMatrix)<-c(");
		//		for (int i = 1; i <= nn; i++)
		//		{
		//			fprintf(fp2, "\"C%d\"", i);
		//			if (i < nn) fprintf(fp2, ",");
		//		}
		//		fprintf(fp2, ")\r\n");
		//		fprintf(fp2, "rownames(confusionMatrix)<-colnames(confusionMatrix)\r\n");
		//		fclose(fp);
		//		fclose(fp2);
		//	}
		//}
		return result;
	}

	void confusion_matrix_outoput(tiny_dnn::result& result)
	{
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
						if (n > 1) fprintf(fp2, ",");
						do
						{
							if (n >= 1)fprintf(fp2, "%c", *p);
							p++;
						} while (!isspace(*p));
						while (isspace(*p)) p++;
						n++;
						if (*p == '\n')break;
					}
					if (n >= 1)fprintf(fp2, ")");
					p = fgets(buf, 4096, fp);
					if (p != NULL)fprintf(fp2, ",");
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
	}

	void report(double α=0.05, std::string& filename = std::string(""))
	{
		FILE* fp = fopen(filename.c_str(), "w");
		if (fp == NULL)
		{
			fp = stdout;
		}

		set_test(nn, 1);
#ifdef USE_LIBTORCH
		if (use_libtorch)
		{
			state_reset(this->rnn_type, torch_getNet());
		}
#endif


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
				torch_nn_test = torch_load_new("fit_best_ts.pt");
				train_result = torch_get_accuracy_nn(torch_nn_test, train_images, train_labels, n_eval_minibatch);
				if (test_images.size() > 0)
				{
					test_result = torch_get_accuracy_nn(torch_nn_test, test_images, test_labels, n_eval_minibatch);
				}
			}
			else
#endif
			{
				nn_test.load("fit_best_ts.model");
				train_result = get_accuracy(nn_test, train_images, train_labels);
				if (test_images.size() > 0)
				{
					test_result = get_accuracy(nn_test, test_images, test_labels);
				}
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
						stream << "test" ;
					}
					else
					{
						stream << "train";
					}
					stream <<  "):" << std::endl;
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
		
		tiny_dnn::tensor_t train, predict;

		std::vector<tiny_dnn::vec_t> y_predict_n;

#ifdef USE_LIBTORCH
		y_predict_n = torch_predict_batch(train_images, n_eval_minibatch);
#endif

		for (int i = 0; i < train_images.size() - 1; i++)
		{
			tiny_dnn::vec_t next_y;
#ifdef USE_LIBTORCH
			if (use_libtorch)
			{
				if (y_predict_n.size() > 0)
				{
					next_y = y_predict_n[i];
				}
				else
				{
					next_y = torch_predict(seq_vec(nY, i));
				}
			}
			else
#endif
			{
				next_y = nn.predict(seq_vec(nY, i));
			}

			//output sequence_length 
			for (int j = 0; j < out_sequence_length; j++)
			{
				tiny_dnn::vec_t z(y_dim);
				tiny_dnn::vec_t y(y_dim);
				for (int k = 0; k < y_dim; k++)
				{
					z[k] = nY[i + sequence_length + j + TARGET_POSITON][k];
					y[k] = next_y[y_dim*j + k];

					if (zscore_normalization)
					{
						y[k] = y[k] * Sigma[k] + Mean[k];
						z[k] = z[k] * Sigma[k] + Mean[k];
					}
					else
					if (minmax_normalization)
					{
						y[k] = y[k] * MaxMin[k] + Min[k];
						z[k] = z[k] * MaxMin[k] + Min[k];
					}
					else
					if (_11_normalization)
					{
						y[k] = 0.5*(y[k] + 1) * MaxMin[k] + Min[k];
						z[k] = 0.5*(z[k] + 1) * MaxMin[k] + Min[k];
					}
				}
				train.push_back(z);
				predict.push_back(y);
			}
		}
		if (use_differnce > 0 && use_differnce_auto_inv)
		{
			predict = diffinv_vec(iY, predict, this->normalize_skilp, use_differnce, use_logdiffernce);
			train = diffinv_vec(iY, train, this->normalize_skilp, use_differnce, use_logdiffernce);

			if (use_logdiffernce)
			{
				predict = exp(predict, this->normalize_skilp);
				train = exp(train, this->normalize_skilp);
			}
		}


		std::vector<double> yy;
		std::vector<double> ff;
		std::vector<double> mer;
		double mse = 0.0;
		double rmse = 0.0;
		for (int i = 0; i < train_images.size()- 1; i++)
		{
			//output sequence_length 
			for (int j = 0; j < out_sequence_length; j++)
			{
				tiny_dnn::vec_t z = train[i + j + TARGET_POSITON];
				tiny_dnn::vec_t y = predict[i + j + TARGET_POSITON];
				for (int k = 0; k < y_dim; k++)
				{
					double d = (y[k] - z[k]);
					mse += d*d;
					yy.push_back(y[k]);
					ff.push_back(z[k]);

					mer.push_back(fabs(d) / z[k]);
				}
			}
		}
		double se = mse;
		mse /= (yy.size());
		rmse = sqrt(mse);
		double Maximum_likelihood_estimator = mse;
		double Maximum_log_likelihood = log(2.0*M_PI) + log(Maximum_likelihood_estimator) + 1.0;

		Maximum_log_likelihood *= -0.5*(train_images.size()*train_labels[0].size());

		double AIC = train_images.size()*(log(2.0*M_PI*se / train_images.size()) + 1) + 2.0*(iX.size() == 0 ? 0 : iX[0].size() + 2.0);
		if (true)	//bias use
		{
			AIC = train_images.size()*(log(2.0*M_PI*se / train_images.size()) + 1) + 2.0*(iX.size() == 0 ? 0 : iX[0].size() + 1.0);
		}

		double mean_ff = 0.0;
		double mean_yy = 0.0;
		for (int i = 0; i < yy.size(); i++)
		{
			mean_yy += yy[i];
			mean_ff += ff[i];
		}
		mean_ff /= yy.size();
		mean_yy /= yy.size();

		double syy = 0.0;
		double sff = 0.0;
		double y_yy_f_yy = 0.0;
		for (int i = 0; i < yy.size(); i++)
		{
			double y_yy = (yy[i] - mean_yy);
			double f_ff = (ff[i] - mean_ff);
			y_yy_f_yy += y_yy*f_ff;
			syy += y_yy*y_yy;
			sff += f_ff*f_ff;
		}

		double r = y_yy_f_yy / sqrt(syy*sff);
		double R2 = 1.0 - se / sff;
		double adjustedR2 = 1.0 - (se/(yy.size()-this->x_dim-1)) / (sff/(yy.size() - 1));


		//set_test(nn, out_sequence_length);
		//double chi_square = 0.0;
		//for (int i = 1; i < train_images.size(); i++)
		//{
		//	tiny_dnn::vec_t y = nn.predict(train_images[i-1]);
		//	for (int k = 0; k < y.size(); k++)
		//	{
		//		float_t d;
		//		auto yy = train_labels[i][k];

		//		if (zscore_normalization)
		//		{
		//			y[k] = y[k] * Sigma[k] + Mean[k];
		//			yy = yy * Sigma[k] + Mean[k];
		//		}
		//		if (minmax_normalization)
		//		{
		//			y[k] = y[k] * MaxMin[k] + Min[k];
		//			yy = yy * MaxMin[k] + Min[k];
		//		}

		//		d = (y[k] - yy);

		//		chi_square += d*d / (y[k] + 1.e-10);
		//	}
		//}

		//Chi_distribution chi_distribution(freedom);
		//double chi_pdf = chi_distribution.p_value(α);

		fprintf(fp, "Status:%d\n", getStatus());
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "SE(残差)                :%.4f\n", se);
		fprintf(fp, "MSE                     :%.4f\n", mse);
		fprintf(fp, "RMSE                    :%.4f\n", rmse);
		fprintf(fp, "r(相関係数)             :%.4f\n", r);
		fprintf(fp, "R^2(決定係数(寄与率))   :%.4f\n", R2);
		fprintf(fp, "R^2(自由度調整済み決定係数(寄与率))   :%.4f\n", adjustedR2);
		fprintf(fp, "MER                     :%.4f\n", median(mer));
		//fprintf(fp, "AIC                     :%.3f\n", AIC);
		//fprintf(fp, "chi square       :%f\n", chi_square);
		//fprintf(fp, "p value          :%f\n", chi_pdf);
		fprintf(fp, "--------------------------------------------------------------------\n");
		//if (chi_distribution.status != 0)
		//{
		//	fprintf(fp, "chi_distribution status:%d\n", chi_distribution.status);
		//}
		//if (chi_square < chi_pdf)
		//{
		//	fprintf(fp, "χ2値:%f < χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
		//	fprintf(fp, "=>良いフィッティングでしょう。\n予測に有効と思われます\n");
		//}
		//else
		//{
		//	fprintf(fp, "χ2値:%f > χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
		//	fprintf(fp, "=>良いとは言えないフィッティングでしょう。\n予測に有効とは言えないと思われます\n");
		//}

		if (classification == 2)
		{
		}

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

			const int diff0_sz2 = Diff[0].size() / 2;
			const int diff_sz = Diff.size();
#pragma omp parallel for
			for (int i = 0; i < diff_sz; i++)
			{
				for (int j = 0; j < diff0_sz2; j++)
				{
					T(i*diff0_sz2 + j, 0) = Diff[i][2 * j];
					T(i*diff0_sz2 + j, 1) = Diff[i][2 * j + 1];
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
