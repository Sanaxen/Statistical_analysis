#include <iostream>
/*
主成分 0.118306 係数 0.796646 -0.008174 -0.462969 0.388520
主成分 0.987518 係数 -0.050663 0.995047 -0.080488 0.028908
主成分 2.551440 係数 0.601096 0.092426 0.563442 -0.559172
主成分 0.342736 係数 0.038354 0.035656 0.679496 0.731808
*/
#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#endif

int main()
{
	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;

	FILE* fp = fopen("data.txt", "r");
	if (fp == NULL)
	{
		return -1;
	}
	fscanf(fp, "%d %d", &variablesNum, &n);   // 変数の数とデータの数

	x = Matrix<dnn_double>(n, variablesNum);

#ifndef USE_FLOAT
	for (int i = 0; i < n; i++) {   // データ
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%lf", &x(i,j));
	}
#else
	for (int i = 0; i < n; i++) {   // データ
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%f", &x(i, j));
	}
#endif

	PCA pca;
	
	pca.set(variablesNum);
	int stat = pca.fit(x, true);


	/*
	主成分 2.551440 係数 0.601096 0.092426 0.563442 -0.559172
	主成分 0.987518 係数 -0.050663 0.995047 -0.080488 0.028908
	主成分 0.342736 係数 0.038354 0.035656 0.679496 0.731808
	主成分 0.118306 係数 0.796646 -0.008174 -0.462969 0.388520
	*/
	pca.Report();

#ifdef USE_GNUPLOT
	gnuPlot plot1(std::string(GNUPLOT_PATH), 1);
	plot1.set_label_x("主成分");
	plot1.set_label_y("固有値");
	plot1.plot_lines(Matrix<dnn_double>(pca.component), std::vector<std::string>());
	plot1.draw();
#endif

}

