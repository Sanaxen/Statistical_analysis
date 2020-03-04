#ifndef _NonLinearRegression_H

#define _NonLinearRegression_H

#include "../../include/util/mathutil.h"
#include "../../include/nonlinear/MatrixToTensor.h"
#include <signal.h>
#pragma warning( disable : 4305 ) 

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
	}
	sig_catch++;
	//exit(0);
	return;
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
	bool normalized = false;
	std::vector<std::string> header;
	std::vector<int> xx_idx;			// non normalize var
	double xx_var_scale = 1.0;			// non normalize var scaling
	std::vector<int> x_idx;
	std::vector<int> y_idx;

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
		}

		for (int i = 0; i < X.size(); i++)
		{
			if (xx_idx.size() && xx_idx[i])
			{
				sigma[i] = xx_var_scale;
				mean[i] = 0.0;
			}
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
			for (int i = 0; i < X.size(); i++)
			{
				if (xx_idx.size() && xx_idx[i])
				{
					maxmin_[i] = xx_var_scale;
					min_[i] = 0.0;
				}
				for (int k = 0; k < X[0].size(); k++)
				{
					X[i][k] = (X[i][k] - min_[k]) / maxmin_[k];
				}
			}
#endif
		}
		for (int i = 0; i < X.size(); i++)
		{
			if (xx_idx.size() && xx_idx[i])
			{
				maxmin_[i] = xx_var_scale;
				min_[i] = 0.0;
			}
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) / maxmin_[k];
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
#endif
		}
		for (int i = 0; i < X.size(); i++)
		{
			if (xx_idx.size() && xx_idx[i])
			{
				maxmin_[i] = 2.0*xx_var_scale;
				min_[i] = -1.0 * xx_var_scale;
			}
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) * 2 / maxmin_[k] - 1;
			}
		}
	}

	float cost_min = std::numeric_limits<float>::max();
	float cost_min0 = std::numeric_limits<float>::max();
	float cost_pre = 0;
	float accuracy_max = std::numeric_limits<float>::min();
	float accuracy_pre = 0;

	size_t net_test_no_Improvement_count = 0;
	void net_test()
	{
		writing lock;

		tiny_dnn::network2<tiny_dnn::sequential> nn_test;

		if (test_mode)
		{
			nn_test.load("fit_best.model");
		}
		else
		{
			nn_test.load("tmp.model");
		}
		printf("layers:%zd\n", nn_test.depth());

		set_test(nn_test, 1);

		FILE* fp_predict = NULL;
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

		if (classification >= 2)
		{
			float loss_train = get_loss(nn_test, train_images, train_labels);
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
			float loss_test = get_loss(nn_test, test_images, test_labels);


			tiny_dnn::result train_result;
			tiny_dnn::result test_result;

			train_result = get_accuracy(nn_test, train_images, train_labels);
			test_result = get_accuracy(nn_test, test_images, test_labels);


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
				nn_test.save("fit_best.model");
				nn_test.save("fit_best.model.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
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

			if (fp_predict)
			{
				for (int i = 0; i < nX.size(); i++)
				{
					tiny_dnn::vec_t x = nX[i];
					tiny_dnn::vec_t& y_predict = nn_test.predict(x);


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
			return;
		}


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
				tiny_dnn::vec_t& y_predict = nn_test.predict(x);


				if (fp_predict)
				{
					for (int k = 0; k < nX[0].size(); k++)
					{
						fprintf(fp_predict, "%.3f,", iX[i][k]);
					}
				}

				tiny_dnn::vec_t& y = iY[i];
				fprintf(fp_test, "%d ", i);
				for (int k = 0; k < y_predict.size()-1; k++)
				{
					float_t yy = y_predict[k];
					if (zscore_normalization)
					{
						yy = y_predict[k] * Sigma_y[k] + Mean_y[k];
					}
					if (minmax_normalization)
					{
						yy = y_predict[k] * MaxMin_y[k] + Min_y[k];
					}
					if (_11_normalization)
					{
						yy = 0.5*(y_predict[k]+1) * MaxMin_y[k] + Min_y[k];
					}
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

				float_t yy = y_predict[y_predict.size() - 1];
				if (zscore_normalization)
				{
					yy = y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1];
				}
				if (minmax_normalization)
				{
					yy = y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1];
				}
				if (_11_normalization)
				{
					yy = 0.5*(y_predict[y_predict.size() - 1]+1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1];
				}

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
				tiny_dnn::vec_t& y_predict = nn_test.predict(x);

				tiny_dnn::vec_t& y = iY[i];
				fprintf(fp_test, "%d ", i);
				for (int k = 0; k < y_predict.size() - 1; k++)
				{
					if (zscore_normalization)
					{
						fprintf(fp_test, "%f %f ", y_predict[k] * Sigma_y[k] + Mean_y[k], y[k]);
						vari_cost += (y_predict[k] * Sigma_y[k] + Mean_y[k] - y[k])*(y_predict[k] * Sigma_y[k] + Mean_y[k] - y[k]);
						cost_tot += (y_predict[k] * Sigma_y[k] + Mean_y[k] - y[k])*(y_predict[k] * Sigma_y[k] + Mean_y[k] - y[k]);
					}else				
					if (minmax_normalization)
					{
						fprintf(fp_test, "%f %f ", y_predict[k] * MaxMin_y[k] + Min_y[k], y[k]);
						vari_cost += (y_predict[k] * MaxMin_y[k] + Min_y[k] - y[k])*(y_predict[k] * MaxMin_y[k] + Min_y[k] - y[k]);
						cost_tot += (y_predict[k] * MaxMin_y[k] + Min_y[k] - y[k])*(y_predict[k] * MaxMin_y[k] + Min_y[k] - y[k]);
					}else
					if (_11_normalization)
					{
						fprintf(fp_test, "%f %f ", 0.5*(y_predict[k]+1) * MaxMin_y[k] + Min_y[k], y[k]);
						vari_cost += (0.5*(y_predict[k] + 1) * MaxMin_y[k] + Min_y[k] - y[k])*(0.5*(y_predict[k] + 1) * MaxMin_y[k] + Min_y[k] - y[k]);
						cost_tot += (0.5*(y_predict[k] + 1) * MaxMin_y[k] + Min_y[k] - y[k])*(0.5*(y_predict[k] + 1) * MaxMin_y[k] + Min_y[k] - y[k]);
					}
					else
					{
						fprintf(fp_test, "%f %f ", y_predict[k], y[k]);
					}
				}
				if (zscore_normalization)
				{
					fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1], y[y_predict.size() - 1]);
					vari_cost += (y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
					cost_tot += (y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
				}else
				if (minmax_normalization)
				{
					fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1], y[y_predict.size() - 1]);
					vari_cost += (y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
					cost_tot += (y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(y_predict[y_predict.size() - 1] * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
				}else
				if (_11_normalization)
				{
					fprintf(fp_test, "%f %f\n", 0.5*(y_predict[y_predict.size() - 1]+1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1], y[y_predict.size() - 1]);
					vari_cost += (0.5*(y_predict[y_predict.size() - 1] + 1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(0.5*(y_predict[y_predict.size() - 1] + 1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
					cost_tot += (0.5*(y_predict[y_predict.size() - 1] + 1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1])*(0.5*(y_predict[y_predict.size() - 1] + 1) * MaxMin_y[y_predict.size() - 1] + Min_y[y_predict.size() - 1] - y[y_predict.size() - 1]);
				}
				else
				{
					fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1], y[y_predict.size() - 1]);
				}
			}
			fclose(fp_test);
		}
		cost /= train_images.size();
		vari_cost /= test_images.size();
		cost_tot /= iY.size();
		//printf("%f %f\n", cost_min, cost);
		if (cost_tot < cost_min)
		{
			nn_test.save("fit_best.model");
			nn_test.save("fit_best.model.json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
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
		visualize_observed_predict();
	}

public:
	void gen_visualize_fit_state()
	{
		set_test(nn, 1);
		nn.save("tmp.model");
		net_test();
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
	size_t n_train_epochs = 2000;
	float_t learning_rate = 1.0;
	int plot = 100;
	std::string model_file = "fit.model";

	int getStatus() const
	{
		return error;
	}

	void normalize_info_save(std::string& normalize_type)
	{
		if (test_mode) return;

		FILE* fp = fopen("normalize_info.dat", "w");
		if (fp)
		{
			fprintf(fp, "%s\n", normalize_type.c_str());
			fprintf(fp, "%d %d\n", nX[0].size(), nY[0].size());
			for (int i = 0; i < nX[0].size(); i++)
			{
				fprintf(fp, "%.16f %.16f\n", Mean_x[i], Sigma_x[i]);
				fprintf(fp, "%.16f %.16f\n", Min_x[i], MaxMin_x[i]);
			}
			for (int i = 0; i < nY[0].size(); i++)
			{
				fprintf(fp, "%.16f %.16f\n", Mean_y[i], Sigma_y[i]);
				fprintf(fp, "%.16f %.16f\n", Min_y[i], MaxMin_y[i]);
			}
			fclose(fp);
		}
	}
	void normalize_info_load(std::string& normalize_type)
	{
		normalized = false;
		if (!test_mode) return;

		FILE* fp = fopen("normalize_info.dat", "r");
		if (fp)
		{
			char buf[256];
			char dummy[128];
			double a = 0.0, b = 0.0;
			int d = 0, dd = 0;

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
				sscanf(buf, "%lf %lf\n", &a, &b);
				Mean_x[i] = a;
				Sigma_x[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "%lf %lf\n", &a, &b);
				Min_x[i] = a;
				MaxMin_x[i] = b;
			}
			for (int i = 0; i < dd; i++)
			{
				fgets(buf, 256, fp);
				sscanf(buf, "%lf %lf\n", &a, &b);
				Mean_y[i] = a;
				Sigma_y[i] = b;
				fgets(buf, 256, fp);
				sscanf(buf, "%lf %lf\n", &a, &b);
				Min_y[i] = a;
				MaxMin_y[i] = b;
			}
			printf("load scaling data\n");
			fclose(fp);
		}
		normalized = true;
	}

	NonLinearRegression(tiny_dnn::tensor_t& Xi, tiny_dnn::tensor_t& Yi, std::string& normalize_type= std::string("zscore"), double dec_random_=0.0, double fluctuation_=0.0, std::string regression_type = "", int classification_ = -1, bool test_mode_ = false)
	{		
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
				printf("ERROR:classification:%d < class_max:%d\n", class_num, (int)class_max);
				fflush(stdout);
				error = -1;
			}
			if (class_min < 0)
			{
				printf("ERROR:class_min:%f\n", class_min);
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
			nn << layers.tanh();

			for (int i = 0; i < n_layers; i++) {
				if (dropout) nn << layers.add_dropout(dropout);
				nn << layers.add_fc(input_size);
				nn << layers.tanh();
			}
		}
		if (classification >= 2)
		{
			if (dropout ) nn << layers.add_dropout(dropout);
			nn << layers.add_fc(std::min((int)input_size, classification*2));
			nn << layers.tanh();
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

		tiny_dnn::adam				optimizer_adam;
		tiny_dnn::gradient_descent	optimizer_sgd;
		tiny_dnn::RMSprop			optimizer_rmsprop;
		tiny_dnn::adagrad			optimizer_adagrad;

		if (opt_type == "SGD")
		{
			std::cout << "optimizer:" << "SGD" << std::endl;
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
			if (epoch >= 3 && plot && epoch % plot == 0)
			{
				gen_visualize_fit_state();
			}
			if (convergence)
			{
				printf("convergence!!\n");
				nn.stop_ongoing_training();
				error = 0;

			}
			if (early_stopp)
			{
				nn.stop_ongoing_training();
				error = 1;
				printf("early_stopp!!\n");
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

			if (epoch < 3 && plot && batch % plot == 0)
			{
				gen_visualize_fit_state();
			}
			if (convergence)
			{
				printf("convergence!!\n");
				nn.stop_ongoing_training();
				error = 0;
			}
			if (early_stopp)
			{
				nn.stop_ongoing_training();
				error = 1;
				printf("early_stopp!!\n");
			}
			if (ctr_c_stopping_nonlinear_regression)
			{
				nn.stop_ongoing_training();
				error = 1;
				printf("CTR-C stopp!!\n");
			}
			++batch;
		};

		if (!test_mode)
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

		try
		{
			nn.load("fit_best.model");
			gen_visualize_fit_state();
		}
		catch (tiny_dnn::nn_error& msg)
		{
			printf("%s\n", msg.what());
			printf("fit_best.model open error.\n");
		}

		std::cout << "end training." << std::endl;

		if (fp_error_loss)fclose(fp_error_loss);
		if (fp_error_vari_loss)fclose(fp_error_vari_loss);
		if (fp_accuracy)fclose(fp_accuracy);
		fp_error_loss = NULL;
		fp_error_vari_loss = NULL;
		fp_accuracy = NULL;

		// save network model & trained weights
		nn.save(model_file+".json", tiny_dnn::content_type::weights_and_model, tiny_dnn::file_format::json);
		nn.save(model_file);


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
		if (minmax_normalization)
		{
			for (int k = 0; k < xx.size(); k++)
			{
				xx[k] = (x[k] - Min_x[k]) / (MaxMin_x[k] + 1.0e-10);
			}
		}
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
			nn_test.load("fit_best.model");

			tiny_dnn::result train_result = get_accuracy(nn_test, train_images, train_labels);
			tiny_dnn::result test_result = get_accuracy(nn_test, test_images, test_labels);

			{
				std::ofstream stream(filename);

				if (!stream.bad())
				{
					stream << "ConfusionMatrix(train):" << std::endl;
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
			return;
		}

		if (regression == "logistic")
		{
			get_accuracy(fp);
			return;
		}

		std::vector<double> xx;
		std::vector<double> yy;
		std::vector<double> ff;
		std::vector<double> dd;
		double mse = 0.0;
		double rmse = 0.0;
		for (int i = 0; i < nY.size(); i++)
		{
			tiny_dnn::vec_t& predict_y = nn.predict(nX[i]);
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
				}
				if (minmax_normalization)
				{
					y[k] = y[k] * MaxMin_y[k] + Min_y[k];
					z[k] = z[k] * MaxMin_y[k] + Min_y[k];
				}
				if (_11_normalization)
				{
					y[k] = 0.5*(y[k] + 1) * MaxMin_y[k] + Min_y[k];
					z[k] = 0.5*(z[k] + 1) * MaxMin_y[k] + Min_y[k];
				}

				double d = (y[k] - z[k]);
				mse += d*d;
				yy.push_back(y[k]);
				ff.push_back(z[k]);
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
		fprintf(fp, "R^2(自由度調整済み決定係数(寄与率))   :%.4f\n", adjustedR2);
		//fprintf(fp, "AIC                                 :%.3f\n", AIC);
		//fprintf(fp, "chi square       :%f\n", chi_square);
		//fprintf(fp, "p value          :%f\n", chi_pdf);
		fprintf(fp, "--------------------------------------------------------------------\n");

		get_accuracy(fp);

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
