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
#include "../../../include/nonlinear/NonLinearRegression.h"
#include "../../../include/nonlinear/MatrixToTensor.h"

#include "gen_test_data.h"


int main(int argc, char** argv)
{
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

	Matrix<dnn_double> x = z.Col(0);
	for (int i = 1; i < x_dim; i++)
	{
		x = x.appendCol(z.Col(i));
	}

	Matrix<dnn_double> y = z.Col(x_dim);
	for (int i = x_dim+1; i < z.n; i++)
	{
		y = y.appendCol(z.Col(i));
	}
	printf("x_dim:%d y_dim:%d\n", x_dim, y_dim);
	//x.print();
	//y.print();

	tiny_dnn::tensor_t X, Y;
	MatrixToTensor(x, X);
	MatrixToTensor(y, Y);

	NonLinearRegression regression(X, Y);

	regression.tolerance = 1.0e-3;
	regression.learning_rate = 1;
	regression.data_set(0.0);
	regression.visualize_loss(10);
	regression.plot = 10;

	int n_layers = -1;
	int input_unit = -1;
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
		}
		else if (argname == "--progress") {
			regression.progress = (0 < atoi(argv[count + 1])) ? true : false;
		}
		else if (argname == "--tol") {
			regression.tolerance = atof(argv[count + 1]);
		}
		else if (argname == "--learning_rate") {
			regression.learning_rate = atof(argv[count + 1]);
		}
		else if (argname == "--test") {
			regression.data_set(atof(argv[count + 1]));
		}
		else if (argname == "--epochs") {
			regression.n_train_epochs = atoi(argv[count + 1]);
		}
		else if (argname == "--minibatch_size") {
			regression.n_minibatch = atoi(argv[count + 1]);
		}
		else if (argname == "--plot") {
			regression.plot = atoi(argv[count + 1]);
		}
		else if (argname == "--n_layers") {
			n_layers = atoi(argv[count + 1]);
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
		<< "Learning rate:    " << regression.learning_rate << std::endl
		<< "Minibatch size:   " << regression.n_minibatch << std::endl
		<< "Number of epochs: " << regression.n_train_epochs << std::endl
		<< "plotting cycle :  " << regression.plot << std::endl
		<< "tolerance :       " << regression.tolerance << std::endl
		<< std::endl;

	regression.fit(n_layers, input_unit);

	return 0;
}
