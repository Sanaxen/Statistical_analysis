#ifndef _TimeSeriesRegression_H
#define _TimeSeriesRegression_H

#include "../../include/util/mathutil.h"
#include <signal.h>

#define OUT_SEQ_LEN	1
#pragma warning( disable : 4305 ) 

#define EARLY_STOPPING_	60

class TimeSeriesRegression
{
	bool convergence = false;
	int error = 0;
	FILE* fp_error_loss = NULL;
	bool visualize_state_flag = true;

	void normalize(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
	{
		mean = std::vector<float_t>(X[0].size(), 0.0);
		sigma = std::vector<float_t>(X[0].size(), 0.0);

#if 0
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
#else
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
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_value) / (max_value - min_value);
			}
		}
		for (int k = 0; k < X[0].size(); k++)
		{
			mean[k] = min_value;
			sigma[k] = (max_value - min_value);
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
			mean[k] = min_value;
			sigma[k] = (max_value - min_value);
		}
		for (int i = 0; i < X.size(); i++)
		{
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - mean[k]) / (sigma[k] - mean[k]);
			}
		}
#endif
#endif
	}

	float cost_min = std::numeric_limits<float>::max();
	float cost_pre = 0;
	size_t net_test_no_Improvement_count = 0;
	void net_test()
	{
		tiny_dnn::network2<tiny_dnn::sequential> nn_test;

		nn_test.load("tmp.model");
		printf("layers:%zd\n", nn_test.depth());

		set_test(nn_test, OUT_SEQ_LEN);


		char plotName[256];
		//sprintf(plotName, "test%04d.dat", plot_count);
		sprintf(plotName, "test.dat", plot_count);
		tiny_dnn::vec_t y_pre = nY[0];
		FILE* fp_test = fopen(plotName, "w");

		float cost = 0.0;
		if (fp_test)
		{
			std::vector<tiny_dnn::vec_t> YY = nY;
			std::vector<tiny_dnn::vec_t> ZZ = nY;
			for (int i = 0; i < iY.size()- sequence_length; i++)
			{
				if (i + sequence_length == train_images.size())
				{
					fclose(fp_test);
					sprintf(plotName, "predict.dat", plot_count);
					fp_test = fopen(plotName, "w");
				}

				tiny_dnn::vec_t y_predict;
				//y_predict = nn_test.predict((i < train_images.size()) ? nY[i] : y_pre);
				//y_predict = nn_test.predict((i < train_images.size()) ? train_images[i] : seq_vec(YY, i));
				y_predict = nn_test.predict( seq_vec(YY, i));

				//if (YY[i + sequence_length][1] != 0)
				//{
				//	y_predict[1] = YY[i + sequence_length][1];
				//	y_predict[0] = YY[i + sequence_length][0];
				//}
				if (i + sequence_length >= train_images.size())
				{
					YY[i + sequence_length] = y_predict;
				}
				y_pre = y_predict;

				tiny_dnn::vec_t y = nY[i + sequence_length];
				for (int k = 0; k < y_predict.size(); k++)
				{
					y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
					y[k] = y[k] * Sigma[k] + Mean[k];

					cost += (y_predict[k] - y[k])*(y_predict[k] - y[k]);

				}

				fprintf(fp_test, "%f ", iX[i + sequence_length][0]);
				for (int k = 0; k < y_predict.size() - 1; k++)
				{
					fprintf(fp_test, "%f %f ", y_predict[k], y[k]);
				}
				fprintf(fp_test, "%f %f\n", y_predict[y_predict.size() - 1], y[y_predict.size() - 1]);
			}
			fclose(fp_test);
		}

		cost /= (iY.size() - sequence_length);
		printf("%f %f\n", cost_min, cost);
		if (cost < cost_min)
		{
			nn_test.save("fit_bast.model");
			printf("!!=========== bast model save ============!!\n");
			cost_min = cost;
		}
		if (cost_min < tolerance)
		{
			convergence = true;
		}
		if (fp_error_loss)
		{
			fprintf(fp_error_loss, "%.10f %.10f %.4f\n", cost, cost_min, tolerance);
			fflush(fp_error_loss);
		}
		if (cost_pre <= cost || fabs(cost_pre - cost) < 1.0e-3)
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
		cost_pre = cost;
	}

	void gen_visualize_fit_state()
	{
		set_test(nn, OUT_SEQ_LEN);
		nn.save("tmp.model");

		net_test();
		set_train(nn, sequence_length, default_backend_type);

#ifdef USE_GNUPLOT
		if (capture)
		{
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

	float_t support_length = 1.0;
	int epoch = 1;
	int plot_count = 0;
	int batch = 0;
	bool early_stopp = false;

public:
	bool early_stopping = true;
	bool test_mode = false;
	bool capture = false;
	bool progress = true;
	float tolerance = 1.0e-6;

	tiny_dnn::core::backend_t default_backend_type = tiny_dnn::core::backend_t::internal;

	tiny_dnn::tensor_t iX;
	tiny_dnn::tensor_t iY;
	//tiny_dnn::tensor_t nX;
	tiny_dnn::tensor_t nY;
	std::vector<float_t> Mean;
	std::vector<float_t> Sigma;
	tiny_dnn::network2<tiny_dnn::sequential> nn;
	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	size_t support_epochs = 0;
	std::string opt_type = "adam";
	std::string rnn_type = "gru";
	size_t input_size = 32;
	size_t sequence_length = 100;
	size_t n_minibatch = 30;
	size_t n_train_epochs = 3000;
	float_t learning_rate = 0.01;
	int plot = 100;
	std::string model_file = "fit.model";

	int getStatus() const
	{
		return error;
	}
	TimeSeriesRegression(tiny_dnn::tensor_t& X, tiny_dnn::tensor_t& Y)
	{
		iX = X;
		iY = Y;
		nY = Y;
		normalize(nY, Mean, Sigma);
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

	void add_seq(tiny_dnn::vec_t& y, tiny_dnn::vec_t& Y)
	{
		for (int i = 0; i < y.size(); i++)
		{
			Y.push_back(y[i]);
		}
	}
	tiny_dnn::vec_t seq_vec(tiny_dnn::tensor_t& ny, int start)
	{

		tiny_dnn::vec_t seq;
		for (int k = 0; k < sequence_length; k++)
		{
			add_seq(ny[start + k], seq);
		}
		return seq;
	}

	int data_set(int sequence_length_, float test = 0.3f)
	{
		train_images.clear();
		train_labels.clear();
		test_images.clear();
		test_images.clear();

		sequence_length = sequence_length_;
		printf("n_minibatch:%d sequence_length:%d\n", n_minibatch, sequence_length);

		size_t dataAll = iY.size() - sequence_length;
		printf("dataset All:%d->", dataAll);
		size_t test_Num = dataAll*test;
		int datasetNum = dataAll - test_Num;

		datasetNum = datasetNum - datasetNum % n_minibatch;
		if (datasetNum == 0)
		{
			printf("Too many min_batch or Sequence length\n");
			error = -1;
			return error;
		}
		size_t train_num_max = datasetNum;
		printf("train:%d test:%d\n", datasetNum, test_Num);

		for (int i = 0; i < train_num_max; i++)
		{
			train_images.push_back(seq_vec(nY, i));
			train_labels.push_back(nY[i + sequence_length]);
		}
		for (int i = train_num_max; i < dataAll; i++)
		{
			test_images.push_back(seq_vec(nY, i));
			test_labels.push_back(nY[i + sequence_length]);
		}
	}

	void construct_net(int n_rnn_layers = 2, int n_layers = 2, int n_hidden_size = -1)
	{
		using tanh = tiny_dnn::activation::tanh;
		using recurrent = tiny_dnn::recurrent_layer;

		int hidden_size = n_hidden_size;
		if (hidden_size <= 0) hidden_size = 64;

		input_size = train_images[0].size();

		// clip gradients
		tiny_dnn::recurrent_layer_parameters params;
		params.clip = 0;	// 1～5
		params.bptt_max = 1e9;

		size_t in_w = train_images[0].size();
		size_t in_h = 1;
		size_t in_map = 1;

		LayerInfo layers(in_w, in_h, in_map);
		nn << layers.add_fc(input_size);
		for (int i = 0; i < n_rnn_layers; i++) {
			nn << layers.add_rnn(rnn_type, hidden_size, sequence_length, params);
			input_size = hidden_size;
			nn << layers.relu();
		}

		size_t sz = hidden_size;
		if (sz > train_labels[0].size() * 10)
		{
			sz = train_labels[0].size() * 10;
		}
		for (int i = 0; i < n_layers; i++) 
		{
			nn << layers.add_fc(sz);
			nn << layers.relu();
		}
		sz =  10;
		for (; sz > 2; sz /= 2)
		{
			nn << layers.add_fc(train_labels[0].size()*sz);
		nn << layers.relu();
		}
		nn << layers.add_fc(train_labels[0].size() * 2);
		nn << layers.tanh();
		nn << layers.add_fc(train_labels[0].size());

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


	void fit(int seq_length = 30, int rnn_layers = 2, int n_layers = 2, int n_hidden_size = -1)
	{
		if (seq_length < 0) seq_length = 30;
		if (rnn_layers < 0) rnn_layers = 2;
		if (n_layers < 0) n_layers = 2;

		sequence_length = seq_length;
		nn.set_input_size(train_images.size());
		using train_loss = tiny_dnn::mse;

		tiny_dnn::optimizer* optimizer_ = NULL;
		
		tiny_dnn::adam optimizer_adam;
		tiny_dnn::gradient_descent optimizer_sgd;
		tiny_dnn::RMSprop optimizer_rmsprop;
		tiny_dnn::adagrad optimizer_adagrad;

		if (opt_type == "SGD")
		{
			std::cout << "optimizer:" << "SGD" << std::endl;
			optimizer_sgd.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_sgd.alpha << std::endl;

			optimizer_ = &optimizer_sgd;
		}else
		if (opt_type == "rmsprop")
		{
			std::cout << "optimizer:" << "RMSprop" << std::endl;
			optimizer_rmsprop.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_rmsprop.alpha << std::endl;
			
			optimizer_ = &optimizer_rmsprop;
		}else
		if (opt_type == "adagrad")
		{
			std::cout << "optimizer:" << "adagrad" << std::endl;
			optimizer_adagrad.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_adagrad.alpha << std::endl;

			optimizer_ = &optimizer_adagrad;
		}else
		if (opt_type == "adam" || optimizer_ == NULL)
		{
			std::cout << "optimizer:" << "adam" << std::endl;

			//optimizer.alpha *=
			//	std::min(tiny_dnn::float_t(4),
			//		static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));
			optimizer_adam.alpha *= learning_rate;
			std::cout << "optimizer.alpha:" << optimizer_adam.alpha << std::endl;

			optimizer_ = &optimizer_adam;
		}

		construct_net(rnn_layers, n_layers, n_hidden_size);
		optimizer_->reset();
		tiny_dnn::timer t;

		float_t loss_min = std::numeric_limits<float>::max();
		tiny_dnn::progress_display disp(nn.get_input_size());
		
		size_t train_data_size = 0;
		if (support_epochs)
		{
			train_data_size = train_images.size();
			for (int i = 0; i < test_images.size(); i++)
			{
				train_images.push_back(test_images[i]);
				train_labels.push_back(test_labels[i]);
			}
		}
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
			if (plot && epoch % plot == 0)
			{
				gen_visualize_fit_state();
			}
			if (convergence)
			{
				nn.stop_ongoing_training();
				error = 0;
			}
			if (early_stopp)
			{
				nn.stop_ongoing_training();
				error = 1;
				printf("early_stopp!!\n");
			}

			if (support_epochs)
			{
				float_t a = std::min(float_t(1.0), float_t((float_t)epoch / (float_t)support_epochs));
				support_length = support_length*(float_t(1.0) - a*a);
				size_t sz = train_data_size + size_t(support_length*test_images.size());

				if (sz != train_images.size())
				{
					if (sz == train_data_size)
					{
						train_images.resize(train_data_size);
						train_labels.resize(train_data_size);
					}
					else if (sz < train_images.size())
					{
						printf("%d -> support_length:%d\n", train_images.size(), sz);
						train_images.resize(sz);
						train_labels.resize(sz);
					}
				}
			}

			if (progress) disp.restart(nn.get_input_size());
			t.restart();
			rnn_state_reset(nn);
			++epoch;
		};
		auto on_enumerate_minibatch = [&]() {
			if (progress) disp += n_minibatch;

			if (plot && batch % plot == 0)
			{
				gen_visualize_fit_state();
			}
			if (convergence)
			{
				nn.stop_ongoing_training();
				error = 0;
			}
			if (early_stopp)
			{
				nn.stop_ongoing_training();
				error = 1;
				printf("early_stopp!!\n");
			}
			++batch;
		};

		if (!test_mode)
		{
			try
			{
				// training
				nn.fit<tiny_dnn::mse>(*optimizer_, train_images, train_labels,
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

			set_test(nn, OUT_SEQ_LEN);
			//float_t loss = nn.get_loss<train_loss>(train_images, train_labels) / train_images.size();

			float_t loss = 0.0;
			for (int i = 0; i < train_images.size(); i++)
			{
				tiny_dnn::vec_t& y = nn.predict(train_images[i]);
				for (int k = 0; k < y.size(); k++)
				{
					float_t d = (y[k] - train_labels[i][k])* Sigma[k];
					loss += d*d;
				}
			}
			for (int i = 0; i < test_images.size(); i++)
			{
				tiny_dnn::vec_t& y = nn.predict(test_images[i]);
				for (int k = 0; k < y.size(); k++)
				{
					float_t d = (y[k] - test_labels[i][k])* Sigma[k];
					loss += d*d;
				}
			}
			loss /= (train_images.size() + test_images.size());
			printf("loss:%.3f\n", loss);

			set_train(nn, seq_length, default_backend_type);

		}

		try
		{
			nn.load("fit_bast.model");
			gen_visualize_fit_state();
		}
		catch (tiny_dnn::nn_error& msg)
		{
			printf("%s\n", msg.what());
			printf("fit_bast.model open error.\n");
		}
		std::cout << "end training." << std::endl;

		if (fp_error_loss)fclose(fp_error_loss);

		// save network model & trained weights
		nn.save(model_file);
	}

	tiny_dnn::vec_t predict_next(tiny_dnn::vec_t& pre)
	{
		set_test(nn, OUT_SEQ_LEN);
		tiny_dnn::vec_t& y_predict = nn.predict(pre);
		for (int k = 0; k < y_predict.size(); k++)
		{
			y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
		}
		return y_predict;
	}

	void report(double α=0.05, std::string& filename = std::string(""))
	{
		FILE* fp = fopen(filename.c_str(), "w");
		if (fp == NULL)
		{
			fp = stdout;
		}

		set_test(nn, OUT_SEQ_LEN);
		double rmse = 0.0;
		for (int i = 1; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t& y = nn.predict(train_images[i-1]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				d = (y[k] - train_labels[i][k])* Sigma[k];
				rmse += d*d;
			}
		}
		rmse /= (train_images.size()*train_labels[0].size());
		rmse = sqrt(rmse);

		set_test(nn, OUT_SEQ_LEN);
		double chi_square = 0.0;
		for (int i = 1; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t& y = nn.predict(train_images[i-1]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				d = (y[k] - train_labels[i][k])* Sigma[k];

				chi_square += d*d / (Sigma[k] * Sigma[k]);
			}
		}

		Chi_distribution chi_distribution(train_images.size()*train_labels[0].size());
		double chi_pdf = chi_distribution.p_value(α);

		fprintf(fp, "Status:%d\n", getStatus());
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "RMSE             :%f\n", rmse);
		fprintf(fp, "chi square       :%f\n", chi_square);
		fprintf(fp, "freedom          :%d\n", train_images.size()*train_labels[0].size());
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
		}
		else
		{
			fprintf(fp, "χ2値:%f > χ2(%.2f)=[%.2f]", chi_square, α, chi_pdf);
			fprintf(fp, "=>良いとは言えないフィッティングでしょう。\n予測に有効とは言えないと思われます\n");
		}


		if (fp != stdout) fclose(fp);
	}

};

#endif
