#define _cublas_Init_def

#include "../../../include/Matrix.hpp"
#include "../../../include/statistical/fastICA.h"
#include "../../../include/util/csvreader.h"
#ifdef USE_GNUPLOT
#include "../../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
#endif

#include <iostream>
#include <string.h>
#include "../../../include/util/dnn_util.hpp"
#include "../../../include/nonlinear/TimeSeriesRegression.h"
#include "../../../include/nonlinear/MatrixToTensor.h"

#include "gen_test_data.h"


//#define X_DIM	2
//#define Y_DIM	3

#define IN_SEQ_LEN	15



int main(int argc, char** argv)
{
	int start_col = 0;
	int x_dim, y_dim;
	std::string csvfile("sample.csv");

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--x") {
			x_dim = atoi(argv[count + 1]);
		}
		else if (argname == "--y") {
			y_dim = atoi(argv[count + 1]);
		}
		else if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
	}

	FILE* fp = fopen(csvfile.c_str(), "r");
	if (fp == NULL)
	{
		make_data_set(csvfile, 1000);
	}
	else
	{
		fclose(fp);
	}

	CSVReader csv1(csvfile, ',', false);
	Matrix<dnn_double> z = csv1.toMat();
	z = csv1.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			z = z.removeCol(0);
		}
	}


	Matrix<dnn_double> x = z.Col(0);
	for (int i = 1; i < x_dim; i++)
	{
		x = x.appendCol(z.Col(i));
	}

	Matrix<dnn_double> y = z.Col(x_dim);
	for (int i = x_dim + 1; i < z.n; i++)
	{
		y = y.appendCol(z.Col(i));
	}
	x.print();
	y.print();

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X);
	MatrixToTensor(y, Y);


	TimeSeriesRegression timeSeries(X, Y);

	timeSeries.tolerance = 0.009;
	timeSeries.learning_rate = 0.01;
	timeSeries.data_set(0.4);
	timeSeries.visualize_loss(10);
	timeSeries.plot = 10;

	int n_layers = -1;
	int n_rnn_layers = -1;
	int input_unit = -1;
	int sequence_length = -1;

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--x") {
			continue;
		}
		else if (argname == "--y") {
			continue;
		}
		else if (argname == "--csv") {
			continue;
		} else	if (argname == "--col") {
			continue;
		}
		else if (argname == "--progress") {
			timeSeries.progress = (0 < atoi(argv[count + 1])) ? true : false;
		} else if (argname == "--tol") {
			timeSeries.tolerance = atof(argv[count + 1]);
		}
		else if (argname == "--learning_rate") {
			timeSeries.learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--test") {
			timeSeries.data_set(atof(argv[count + 1]));
		}
		else if (argname == "--epochs") {
			timeSeries.n_train_epochs = atoi(argv[count + 1]);
		}
		else if (argname == "--minibatch_size") {
			timeSeries.n_minibatch = atoi(argv[count + 1]);
		}
		else if (argname == "--plot") {
			timeSeries.plot = atoi(argv[count + 1]);
		}
		else if (argname == "--seq_len") {
			sequence_length = atoi(argv[count + 1]);
		}
		else if (argname == "--n_layers") {
			n_layers = atoi(argv[count + 1]);
		}
		else if (argname == "--n_rnn_layers") {
			n_rnn_layers = atoi(argv[count + 1]);
		}
		else if (argname == "--input_unit") {
			input_unit = atoi(argv[count + 1]);
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}

	}
	std::cout << "Running with the following parameters:" << std::endl
		<< "Learning rate   :    " << timeSeries.learning_rate << std::endl
		<< "Minibatch size  :   " << timeSeries.n_minibatch << std::endl
		<< "Number of epochs: " << timeSeries.n_train_epochs << std::endl
		<< "plotting cycle  :  " << timeSeries.plot << std::endl
		<< "tolerance       :       " << timeSeries.tolerance << std::endl
		<< "sequence_length :       " << timeSeries.sequence_length << std::endl
		
		<< std::endl;

	timeSeries.fit(sequence_length, n_rnn_layers, n_layers, input_unit);

	return 0;
}
