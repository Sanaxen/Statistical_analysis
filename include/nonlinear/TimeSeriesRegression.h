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
	FILE* fp_error_vari_loss = NULL;
	bool visualize_state_flag = true;

	void normalizeZ(tiny_dnn::tensor_t& X, std::vector<float_t>& mean, std::vector<float_t>& sigma)
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

	void normalizeMinMax(tiny_dnn::tensor_t& X, std::vector<float_t>& min_, std::vector<float_t>& maxmin_)
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
			for (int k = 0; k < X[0].size(); k++)
			{
				X[i][k] = (X[i][k] - min_[k]) / maxmin_[k];
			}
		}
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

		float vari_cost = 0.0;
		float cost = 0.0;
		float cost_tot = 0.0;
		if (fp_test)
		{
			std::vector<tiny_dnn::vec_t> YY = nY;

			//The first sequence is only referenced
			for (int i = 0; i < sequence_length; i++)
			{
				tiny_dnn::vec_t y = nY[i];
				for (int k = 0; k < y_dim; k++)
				{
					if (zscore_normalization)
					{
						y[k] = y[k] * Sigma[k] + Mean[k];
					}
					if (minmax_normalization)
					{
						y[k] = y[k] * MaxMin[k] + Min[k];
					}
				}
				fprintf(fp_test, "%f ", iX[i][0]);
				for (int k = 0; k < y_dim - 1; k++)
				{
					fprintf(fp_test, "%f %f ", y[k], y[k]);
				}
				fprintf(fp_test, "%f %f\n", y[y_dim - 1], y[y_dim - 1]);
			}

			float_t x_value;
			// {y(0) y(1) ... y(sequence_length-1)} -> y(sequence_length)
			for (int i = 0; i < iY.size()- sequence_length + prophecy; i++)
			{
				//prophecy
				if (i + sequence_length == iY.size())
				{
					fclose(fp_test);
					sprintf(plotName, "prophecy.dat", plot_count);
					fp_test = fopen(plotName, "w");

					tiny_dnn::vec_t y = YY[i + sequence_length - 1];
					//Plot Output of only data concatenation
					for (int k = 0; k < y_dim; k++)
					{
						if (zscore_normalization)
						{
							y[k] = y[k] * Sigma[k] + Mean[k];
						}
						if (minmax_normalization)
						{
							y[k] = y[k] * MaxMin[k] + Min[k];
						}
					}
					

					fprintf(fp_test, "%f ", x_value);
					for (int k = 0; k < y_dim - 1; k++)
					{
						fprintf(fp_test, "%f ", y[k]);
					}
					fprintf(fp_test, "%f\n", y[y_dim - 1]);
				}

				if (i + sequence_length == train_images.size()+ sequence_length)
				{
					fclose(fp_test);
					sprintf(plotName, "predict_all.dat", plot_count);
					fp_test = fopen(plotName, "w");

					tiny_dnn::vec_t y = nY[i + sequence_length - 1];

					//Plot Output of only data concatenation
					for (int k = 0; k < y_dim; k++)
					{
						if (zscore_normalization)
						{
							y_pre[k] = y_pre[k] * Sigma[k] + Mean[k];
							y[k] = y[k] * Sigma[k] + Mean[k];
						}
						if (minmax_normalization)
						{
							y_pre[k] = y_pre[k] * MaxMin[k] + Min[k];
							y[k] = y[k] * MaxMin[k] + Min[k];
						}
					}

					fprintf(fp_test, "%f ", iX[i + sequence_length - 1][0]);
					for (int k = 0; k < y_dim - 1; k++)
					{
						fprintf(fp_test, "%f %f ", y_pre[k], y[k]);
					}
					fprintf(fp_test, "%f %f\n", y_pre[y_dim - 1], y[y_dim - 1]);
				}
				//After the index reaches train, test data not used for training after that
				if (i + sequence_length == train_images.size())
				{
					//From this point on, test data not used for training
					fclose(fp_test);
					sprintf(plotName, "predict.dat", plot_count);
					fp_test = fopen(plotName, "w");

					tiny_dnn::vec_t y = nY[i + sequence_length-1];

					//Plot Output of only data concatenation
					for (int k = 0; k < y_dim; k++)
					{
						if (zscore_normalization)
						{
							y_pre[k] = y_pre[k] * Sigma[k] + Mean[k];
							y[k] = y[k] * Sigma[k] + Mean[k];
						}
						if (minmax_normalization)
						{
							y_pre[k] = y_pre[k] * MaxMin[k] + Min[k];
							y[k] = y[k] * MaxMin[k] + Min[k];
						}
					}

					fprintf(fp_test, "%f ", iX[i + sequence_length-1][0]);
					for (int k = 0; k < y_dim - 1; k++)
					{
						fprintf(fp_test, "%f %f ", y_pre[k], y[k]);
					}
					fprintf(fp_test, "%f %f\n", y_pre[y_dim - 1], y[y_dim - 1]);

				}

				tiny_dnn::vec_t y_predict;

				// {y(0) y(1) ... y(sequence_length-1)} -> y(sequence_length)
				if (support == 0)
				{
					y_predict = nn_test.predict(seq_vec(YY, i));
				}else
				{
					if (support > sequence_length) support = sequence_length;

					tiny_dnn::vec_t& obs = seq_vec(nY, i);
					tiny_dnn::vec_t x = seq_vec(YY, i);
					if (support)
					{
						size_t l = nY[0].size();
						for (int j = 0; j < support; j++)
						{
							for (int k = 0; k < y_dim; k++)
							{
								x[l*j + k] = obs[l*j + k];
							}
						}
					}
					y_predict = nn_test.predict(x);
				}

				//prophecy
				if (i + sequence_length >= iY.size())
				{
					YY.push_back(y_predict);
					//Add an explanatory variable
					for (int k = y_dim; k < YY[0].size(); k++)
					{
						YY[i + sequence_length].push_back(0);
					}
				}
				else
				{
				//Change to the predicted value
				if (i + sequence_length >= train_images.size())
				{
					YY[i + sequence_length] = y_predict;
					//Add an explanatory variable
					for (int k = y_dim; k < YY[0].size(); k++)
					{
						YY[i + sequence_length].push_back(nY[i + sequence_length][k]);
					}
				}
				}
				y_pre = y_predict;

				tiny_dnn::vec_t y;
				if (i + sequence_length < iY.size())
				{
					y = nY[i + sequence_length];
				for (int k = 0; k < y_dim; k++)
				{
					if (zscore_normalization)
					{
						y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
						y[k] = y[k] * Sigma[k] + Mean[k];
					}
					if (minmax_normalization)
					{
						y_predict[k] = y_predict[k] * MaxMin[k] + Min[k];
						y[k] = y[k] * MaxMin[k] + Min[k];
					}
					if (i < train_images.size())
					{
						cost += (y_predict[k] - y[k])*(y_predict[k] - y[k]);
					}
					else
					{
						vari_cost += (y_predict[k] - y[k])*(y_predict[k] - y[k]);
					}
					cost_tot += (y_predict[k] - y[k])*(y_predict[k] - y[k]);
				}

				fprintf(fp_test, "%f ", iX[i + sequence_length][0]);
				for (int k = 0; k < y_dim - 1; k++)
				{
					fprintf(fp_test, "%f %f ", y_predict[k], y[k]);
				}
				fprintf(fp_test, "%f %f\n", y_predict[y_dim - 1], y[y_dim - 1]);
					x_value = iX[i + sequence_length][0];
				}
				else
				{
					for (int k = 0; k < y_dim; k++)
					{
						if (zscore_normalization)
						{
							y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
						}
						if (minmax_normalization)
						{
							y_predict[k] = y_predict[k] * MaxMin[k] + Min[k];
						}
					}
					fprintf(fp_test, "%f ", x_value+(iX[iX.size() - 1][0]-iX[iX.size()-2][0]));
					for (int k = 0; k < y_dim - 1; k++)
					{
						fprintf(fp_test, "%f ", y_predict[k]);
					}
					fprintf(fp_test, "%f\n", y_predict[y_dim - 1]);
					x_value = x_value + (iX[iX.size() - 1][0] - iX[iX.size() - 2][0]);
				}
			}
			fclose(fp_test);
		}

		cost /= (train_images.size() + sequence_length);
		vari_cost /= (iY.size() - train_images.size() - sequence_length);
		cost_tot /= (iY.size() -  sequence_length);
		printf("%f %f\n", cost_min, cost_tot);
		if (cost_tot < cost_min)
		{
			nn_test.save("fit_bast.model");
			printf("!!=========== bast model save ============!!\n");
			cost_min = cost_tot;
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
		if (fp_error_vari_loss)
		{
			fprintf(fp_error_vari_loss, "%.10f\n", vari_cost);
			fflush(fp_error_vari_loss);
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
	}

	void gen_visualize_fit_state()
	{
		set_test(nn, OUT_SEQ_LEN);
		nn.save("tmp.model");

		net_test();
		set_train(nn, sequence_length, n_bptt_max, default_backend_type);

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
	size_t x_dim;
	size_t y_dim;
	size_t prophecy = 0;
	size_t freedom = 0;
	bool minmax_normalization = false;
	bool zscore_normalization = false;
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
	std::vector<float_t> Min;
	std::vector<float_t> MaxMin;
	tiny_dnn::network2<tiny_dnn::sequential> nn;
	std::vector<tiny_dnn::vec_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	size_t support = 0;
	std::string opt_type = "adam";
	std::string rnn_type = "lstm";
	size_t input_size = 32;
	size_t sequence_length = 100;
	size_t n_minibatch = 30;
	size_t n_train_epochs = 3000;
	float_t learning_rate = 1.0;
	size_t n_bptt_max = 1e9;
	int plot = 100;
	std::string model_file = "fit.model";

	int getStatus() const
	{
		return error;
	}
	TimeSeriesRegression(tiny_dnn::tensor_t& X, tiny_dnn::tensor_t& Y, std::string normalize_type="")
	{
		iX = X;
		iY = Y;
		nY = Y;
		if (normalize_type == "zscore") zscore_normalization = true;
		if (normalize_type == "minmax") minmax_normalization = true;

		if (minmax_normalization)
		{
			normalizeMinMax(nY, Min, MaxMin);
			printf("minmax_normalization\n");

			//get Mean, Sigma
			auto dmy = iY;
			normalizeZ(dmy, Mean, Sigma);
		}
		if (zscore_normalization)
		{
			normalizeZ(nY, Mean, Sigma);
			printf("zscore_normalization\n");
		}
	}
	void visualize_loss(int n)
	{
		visualize_state_flag = n;
		if (n > 0)
		{
			fp_error_loss = fopen("error_loss.dat", "w");
			fp_error_vari_loss = fopen("error_var_loss.dat", "w");
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

			auto ny = nY[i + sequence_length];
			tiny_dnn::vec_t y;
			for (int k = 0; k < y_dim; k++)
			{
				y.push_back(ny[k]);
			}
			train_labels.push_back(y);
		}

		for (int i = train_num_max; i < dataAll; i++)
		{
			test_images.push_back(seq_vec(nY, i));

			auto ny = nY[i + sequence_length];
			tiny_dnn::vec_t y;
			for (int k = 0; k < y_dim; k++)
			{
				y.push_back(ny[k]);
			}
			test_labels.push_back(y);
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

		if (n_bptt_max == 0) n_bptt_max = sequence_length;
		params.bptt_max = n_bptt_max;
		printf("bptt_max:%d\n", n_bptt_max);

		size_t in_w = train_images[0].size();
		size_t in_h = 1;
		size_t in_map = 1;

		LayerInfo layers(in_w, in_h, in_map);
		nn << layers.add_input(input_size);
		nn << layers.add_fc(input_size, false);
		//nn << layers.relu();
		//nn << layers.add_fc(input_size);
		//nn << layers.relu();
		for (int i = 0; i < n_rnn_layers; i++) {
			nn << layers.add_rnn(rnn_type, hidden_size, sequence_length, params);
			input_size = hidden_size;
			//Scaled Exponential Linear Unit. (Klambauer et al., 2017)
			//nn << layers.selu_layer();
			nn << layers.tanh();
		}
		//nn << layers.add_batnorm();
		//nn << layers.add_cnv(1, hidden_size, 1, tiny_dnn::padding::same);
		//nn << layers.add_maxpool(2, 1, tiny_dnn::padding::same);

		size_t sz = hidden_size;

		if (sz > train_labels[0].size() * 10)
		{
			sz = train_labels[0].size() * 10;
		}
		for (int i = 0; i < n_layers; i++) {
			nn << layers.add_fc(sz);
			nn << layers.tanh();
		}
		nn << layers.add_fc(sz);
		//nn << layers.relu();
		nn << layers.tanh();
		nn << layers.add_fc(train_labels[0].size());
		nn << layers.add_linear(train_labels[0].size());

		nn.weight_init(tiny_dnn::weight_init::xavier());
		for (auto n : nn) n->set_parallelize(true);
		printf("layers:%zd\n", nn.depth());
		freedom = layers.get_parameter_num();
		printf("freedom:%zd\n", freedom);

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

	
		tiny_dnn::adam optimizer_adam;
		tiny_dnn::gradient_descent optimizer_sgd;
		tiny_dnn::RMSprop optimizer_rmsprop;
		tiny_dnn::adagrad optimizer_adagrad;

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

		construct_net(rnn_layers, n_layers, n_hidden_size);
		
		if (opt_type == "adam") optimizer_adam.reset();
		if (opt_type == "sgd" )	optimizer_sgd.reset();
		if (opt_type == "rmsprop" )	optimizer_rmsprop.reset();
		if (opt_type == "adagrad")optimizer_adagrad.reset();

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
				if (opt_type == "adam")
				{
					// training
					nn.fit<tiny_dnn::mse>(optimizer_adam, train_images, train_labels,
						n_minibatch,
						n_train_epochs,
						on_enumerate_minibatch,
						on_enumerate_epoch
						);
				}
				if (opt_type == "sgd")
				{
					// training
					nn.fit<tiny_dnn::mse>(optimizer_sgd, train_images, train_labels,
						n_minibatch,
						n_train_epochs,
						on_enumerate_minibatch,
						on_enumerate_epoch
						);
				}
				if (opt_type == "rmsprop")
				{
					// training
					nn.fit<tiny_dnn::mse>(optimizer_rmsprop, train_images, train_labels,
						n_minibatch,
						n_train_epochs,
						on_enumerate_minibatch,
						on_enumerate_epoch
						);
				}
				if (opt_type == "adagrad")
				{
					// training
					nn.fit<tiny_dnn::mse>(optimizer_adagrad, train_images, train_labels,
						n_minibatch,
						n_train_epochs,
						on_enumerate_minibatch,
						on_enumerate_epoch
						);
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
		if (fp_error_vari_loss)fclose(fp_error_vari_loss);

		// save network model & trained weights
		nn.save(model_file);
	}

	tiny_dnn::vec_t predict_next(tiny_dnn::vec_t& pre)
	{
		set_test(nn, OUT_SEQ_LEN);
		tiny_dnn::vec_t& y_predict = nn.predict(pre);
		for (int k = 0; k < y_predict.size(); k++)
		{
			if (zscore_normalization)
			{
				y_predict[k] = y_predict[k] * Sigma[k] + Mean[k];
			}
			if (minmax_normalization)
			{
				y_predict[k] = y_predict[k] * MaxMin[k] + Min[k];
			}
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
		std::vector<double> yy;
		std::vector<double> ff;
		double mse = 0.0;
		double rmse = 0.0;
		for (int i = 1; i < train_images.size(); i++)
		{
			tiny_dnn::vec_t y = nn.predict(train_images[i-1]);
			for (int k = 0; k < y.size(); k++)
			{
				float_t d;
				auto z = train_labels[i][k];

				if (zscore_normalization)
				{
					y[k] = y[k] * Sigma[k] + Mean[k];
					z = z * Sigma[k] + Mean[k];
				}
				if (minmax_normalization)
				{
					y[k] = y[k] * MaxMin[k] + Min[k];
					z = z * MaxMin[k] + Min[k];
				}

				d = (y[k] - z);
				mse += d*d;
				yy.push_back(y[k]);
				ff.push_back(train_labels[i][k]);
			}
		}
		mse /= (train_images.size()*train_labels[0].size());
		rmse = sqrt(mse);
		double Maximum_likelihood_estimator = mse;
		double Maximum_log_likelihood = log(2.0*M_PI) + log(Maximum_likelihood_estimator) + 1.0;

		Maximum_log_likelihood *= -0.5*(train_images.size()*train_labels[0].size());

		double AIC = -2.0*Maximum_log_likelihood + 2.0*freedom;
		double SE = sqrt(mse / std::max(1, (int)(train_images.size()*train_labels[0].size()) - (int)freedom));

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
		double R2 = 1.0 - mse / syy;

		//set_test(nn, OUT_SEQ_LEN);
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

		Chi_distribution chi_distribution(freedom);
		double chi_pdf = chi_distribution.p_value(α);

		fprintf(fp, "Status:%d\n", getStatus());
		fprintf(fp, "--------------------------------------------------------------------\n");
		fprintf(fp, "MSE            :%.4f\n", mse);
		fprintf(fp, "RMSE           :%.4f\n", rmse);
		fprintf(fp, "SE(標準誤差)            :%.4f\n", SE);
		fprintf(fp, "r(相関係数)             :%.4f\n", r);
		fprintf(fp, "R^2(決定係数(寄与率))   :%.4f\n", R2);
		fprintf(fp, "Maximum log likelihood(最大対数尤度):%.4f\n", Maximum_log_likelihood);
		fprintf(fp, "AIC          :%.3f\n", AIC);
		fprintf(fp, "freedom          :%d\n", freedom);
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


		if (fp != stdout) fclose(fp);
	}

};

#endif
