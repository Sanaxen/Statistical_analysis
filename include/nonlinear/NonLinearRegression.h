#ifndef _NonLinearRegression_H

#define _NonLinearRegression_H

#include "../../include/util/mathutil.h"

class NonLinearRegression
{
	int error = 0;
	FILE* fp_error_loss = NULL;
	bool visualize_state_flag = true;

	void normalize(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
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
			sigma[k] /= X.size();
		}
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - mean[k]) / (sigma[k] + 1.0e-10);
			}
		}
	}

	void net_test()
	{
		tiny_dnn::network2<tiny_dnn::sequential> nn_test;

		nn_test.load(model_file);
		printf("layers:%zd\n", nn_test.depth());

		set_test(nn_test, 1);


		char plotName[256];
		//sprintf(plotName, "test%04d.dat", plot_count);
		sprintf(plotName, "test.dat", plot_count);
		FILE* fp_test = fopen(plotName, "w");

		if (fp_test)
		{
			for (int i = 0; i < nY.size(); i++)
			{
				tiny_dnn::vec_t x = nX[i];
				tiny_dnn::vec_t& y_predict = nn_test.predict(x);

				tiny_dnn::vec_t& y = iY[i];
				fprintf(fp_test, "%d ", i);
				for (int k = 0; k < y_predict.size()-1; k++)
				{
					fprintf(fp_test, "%f %f ", y_predict[k]*Sigma_y[k] + Mean_y[k], y[k]);
				}
				fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1] * Sigma_y[y_predict.size() - 1] + Mean_y[y_predict.size() - 1], y[y_predict.size() - 1]);
			}
			fclose(fp_test);
		}
	}

	void gen_visualize_fit_state()
	{
		set_test(nn, 1);
		nn.save(model_file);
		net_test();
		set_train(nn, 1);


#ifdef USE_GNUPLOT
		if (capture)
		{
			printf("capture\n");
			std::string plot = std::string(GNUPLOT_PATH);
			plot += " test_plot_capture1.plt";
			system(plot.c_str());

			char buf[256];
			sprintf(buf, "images\\test_%04d.png", plot_count);
			std::string cmd = "cmd.exe /c ";
			cmd += "copy images\\test.png " + std::string(buf);
			system(cmd.c_str());
			printf("%s\n", cmd.c_str());
		}
		plot_count++;
#endif
	}

	//int load_count = 1;
	int epoch = 1;
	int plot_count = 0;
	int batch = 0;
public:
	bool capture = false;
	bool progress = true;
	float tolerance = 1.0e-6;
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::backend_t::internal;
	tiny_dnn::tensor_t iX;
	tiny_dnn::tensor_t iY;
	tiny_dnn::tensor_t nX;
	tiny_dnn::tensor_t nY;
	std::vector<float_t> Mean_x, Mean_y;
	std::vector<float_t> Sigma_x, Sigma_y;
	bool NormalizeData = true;
	tiny_dnn::network2<tiny_dnn::sequential> nn;
	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

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
	NonLinearRegression(tiny_dnn::tensor_t& Xi, tiny_dnn::tensor_t& Yi)
	{
		iX = Xi;
		iY = Yi;
		nX = Xi;
		nY = Yi;
		if (NormalizeData)
		{
			normalize(nX, Mean_x, Sigma_x);
			normalize(nY, Mean_y, Sigma_y);
		}
		else
		{
			tiny_dnn::tensor_t dmyX = nX;
			tiny_dnn::tensor_t dmyY = nY;
			normalize(dmyX, Mean_x, Sigma_x);
			normalize(dmyY, Mean_y, Sigma_y);
		}
	}
	void visualize_loss(int n)
	{
		visualize_state_flag = n;
		if (n > 0)
		{
			fp_error_loss = fopen("error_loss.dat", "w");
		}
		else
		{
			if (fp_error_loss)
			{
				fclose(fp_error_loss);
				fp_error_loss = NULL;
			}
		}
	}

	int data_set(float test = 0.3f)
	{
		size_t dataAll = iY.size();
		printf("dataset All:%d->", dataAll);
		size_t test_Num = dataAll*test;
		int datasetNum = dataAll - test_Num;

		if (datasetNum == 0)
		{
			printf("Too many min_batch or Sequence length\n");
			error = -1;
			return error;
		}
		size_t train_num_max = datasetNum;
		printf("train:%d test:%d\n", datasetNum, test_Num);

		std::random_device rnd;     // 非決定的な乱数生成器を生成
		std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタ
		std::uniform_int_distribution<> rand1(0, train_num_max - 1);

		for (int i = 0; i < train_num_max; i++)
		{
			int ii = rand1(mt);
			train_images.push_back(nX[ii]);
			train_labels.push_back(nY[ii]);
		}
		std::uniform_int_distribution<> rand2(train_num_max, dataAll - 1);
		for (int i = train_num_max - 1; i < dataAll - 1; i++)
		{
			int ii = rand2(mt);
			test_images.push_back(nX[ii]);
			test_labels.push_back(nY[ii]);
		}
	}

	void construct_net(int n_layers = 5)
	{
		using tanh = tiny_dnn::activation::tanh;
		using recurrent = tiny_dnn::recurrent_layer;

		int hidden_size = iX[0].size() * 50;

		// clip gradients
		tiny_dnn::recurrent_layer_parameters params;
		params.clip = 0;
		params.bptt_max = 1e9;

		size_t in_w = iX[0].size();
		size_t in_h = 1;
		size_t in_map = 1;

		LayerInfo layers(in_w, in_h, in_map);
		nn << layers.add_fc(input_size);
		nn << layers.tanh();
		for (int i = 0; i < n_layers; i++) {
			nn << layers.add_fc(input_size);
			nn << layers.tanh();
		}
		nn << layers.add_fc(iY[0].size());

		nn.weight_init(tiny_dnn::weight_init::xavier());
		for (auto n : nn) n->set_parallelize(true);
		printf("layers:%zd\n", nn.depth());

#ifdef USE_GRAPHVIZ_DOT
		// generate graph model in dot language
		std::ofstream ofs("graph_net.txt");
		tiny_dnn::graph_visualizer viz(nn, "graph");
		viz.generate(ofs);
		printf("dot -Tgif graph_net.txt -o graph.gif\n");
#endif
	}


	void fit(int n_layers = 5, int input_unit = 32)
	{
		if (n_layers < 0) n_layers = 5;
		if (input_unit < 0) input_unit = 32;
		
		input_size = input_unit;

		nn.set_input_size(train_images.size());
		using train_loss = tiny_dnn::mse;


		tiny_dnn::adam optimizer;
		std::cout << "optimizer:" << "adam" << std::endl;

		optimizer.alpha *= learning_rate;
		std::cout << "optimizer.alpha:" << optimizer.alpha << std::endl;

		construct_net(n_layers);
		optimizer.reset();
		tiny_dnn::timer t;

		tiny_dnn::progress_display disp(nn.get_input_size());

		auto on_enumerate_epoch = [&]() {

			if (epoch % 10 == 0) {
				optimizer.alpha *= 0.97;
			}
			if (epoch % 1 == 0)
			{
				std::cout << "\nEpoch " << epoch << "/" << n_train_epochs << " finished. "
					<< t.elapsed() << "s elapsed." << std::endl;

				if (fp_error_loss)
				{
					set_test(nn, 1);
					float_t loss;
#if 0
					float_t loss = nn.get_loss<train_loss>(train_images, train_labels) / train_images.size();
#else
					loss = 0;
					for (int i = 0; i < train_images.size(); i++)
					{
						tiny_dnn::vec_t& y = nn.predict(train_images[i]);
						for (int k = 0; k < y.size(); k++)
						{
							float_t d;
							if (NormalizeData)
							{
								d = (y[k] - train_labels[i][k])* Sigma_y[k];
							}
							else
							{
								d = (y[k] - train_labels[i][k]);
							}

							loss += d*d;
						}
					}
					loss /= train_images.size();
#endif
					set_train(nn, 1);

					fprintf(fp_error_loss, "%d %.3f\n", epoch, loss);
					fflush(fp_error_loss);

					error = 1;
					if (loss < tolerance)
					{
						nn.stop_ongoing_training();
						error = 0;
					}
				}
			}
			if (epoch >= 3 && plot && epoch % plot == 0)
			{
				gen_visualize_fit_state();
			}

			if (progress) disp.restart(nn.get_input_size());
			t.restart();
			//rnn_state_reset(nn);
			++epoch;
		};
		auto on_enumerate_minibatch = [&]() {
			if (progress) disp += n_minibatch;

			if (epoch < 3 && plot && batch % plot == 0)
			{
				gen_visualize_fit_state();
			}
			++batch;
		};

		try
		{
			// training
			nn.fit<tiny_dnn::mse>(optimizer, train_images, train_labels,
				n_minibatch,
				n_train_epochs,
				on_enumerate_minibatch,
				on_enumerate_epoch
				);
		}
		catch (tiny_dnn::nn_error &err) {
			std::cerr << "Exception: " << err.what() << std::endl;
			error = -1;
			return;
		}
		set_test(nn, 1);
		//float_t loss = nn.get_loss<train_loss>(train_images, train_labels) / train_images.size();
		float_t loss = 0.0;
		for (int i = 0; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t& y = nn.predict(train_images[i]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				if (NormalizeData)
				{
					d = (y[k] - train_labels[i][k])* Sigma_y[k];
				}
				else
				{
					d = (y[k] - train_labels[i][k]);
				}
				loss += d*d;
			}
		}
		loss /= train_images.size();

		printf("loss:%.3f\n", loss);
		set_train(nn, 1);

		gen_visualize_fit_state();
		std::cout << "end training." << std::endl;

		if (fp_error_loss)fclose(fp_error_loss);

		// save network model & trained weights
		nn.save(model_file);
	}

	tiny_dnn::vec_t predict_next(tiny_dnn::vec_t& x)
	{
		tiny_dnn::vec_t xx = x;
		if (NormalizeData)
		{
			for (int k = 0; k < xx.size(); k++)
			{
				xx[k] = (x[k] - Mean_x[k]) / (Sigma_x[k] + 1.0e-10);
			}
		}
		tiny_dnn::vec_t& y = nn.predict(xx);
		if (NormalizeData)
		{
			for (int k = 0; k < y.size(); k++)
			{
				y[k] = y[k] * Sigma_y[k] + Mean_y[k];
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

		double rmse = 0.0;
		for (int i = 0; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t& y = nn.predict(train_images[i]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				if (NormalizeData)
				{
					d = (y[k] - train_labels[i][k])* Sigma_y[k];
				}
				else
				{
					d = (y[k] - train_labels[i][k]);
				}
				rmse += d*d;
			}
		}
		rmse /= (train_images.size()*train_labels[0].size());
		rmse = sqrt(rmse);

		double chi_square = 0.0;
		for (int i = 0; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t& y = nn.predict(train_images[i]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				if (NormalizeData)
				{
					d = (y[k] - train_labels[i][k])* Sigma_y[k];
				}
				else
				{
					d = (y[k] - train_labels[i][k]);
				}
				chi_square += d*d/(Sigma_y[k]* Sigma_y[k]);
			}
		}

		Chi_distribution chi_distribution(train_images.size()*train_labels[0].size());
		double chi_pdf = chi_distribution.p_value(α);

		fprintf(fp, "Status:%d\n", getStatus());
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "RMSE             :%f\n", rmse);
		fprintf(fp, "chi square       :%f\n", chi_square);
		fprintf(fp, "p value          :%f\n", chi_pdf);
		fprintf(fp, "--------------------------------------------------------------------\n");


		if (chi_distribution.status != 0)
		{
			fprintf(fp, "chi_distribution status:%d\n", chi_distribution.status);
		}
		if (chi_square < chi_pdf)
		{
			fprintf(fp, "χ2値:%f < χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
			fprintf(fp, "=>良いフィッティングでしょう。\n予測に有効と思われます\n");
			fprintf(fp, "ただし、データ点範囲外での予測が正しい保証はありません。\n");
		}
		else
		{
			fprintf(fp, "χ2値:%f < χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
			fprintf(fp, "=>良いとは言えないフィッティングでしょう。\n予測に有効とは言えないと思われます\n");
		}
		if (fp != stdout) fclose(fp);
	}

};

#endif
