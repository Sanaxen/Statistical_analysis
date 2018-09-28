#include <iostream>
/*
�听�� 0.118306 �W�� 0.796646 -0.008174 -0.462969 0.388520
�听�� 0.987518 �W�� -0.050663 0.995047 -0.080488 0.028908
�听�� 2.551440 �W�� 0.601096 0.092426 0.563442 -0.559172
�听�� 0.342736 �W�� 0.038354 0.035656 0.679496 0.731808
*/
#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#define GNUPLOT_PATH "\"C:\\Program Files (x86)\\gnuplot\\bin\\wgnuplot.exe\""
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
	fscanf(fp, "%d %d", &variablesNum, &n);   // �ϐ��̐��ƃf�[�^�̐�

	x = Matrix<dnn_double>(n, variablesNum);

#ifndef USE_FLOAT
	for (int i = 0; i < n; i++) {   // �f�[�^
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%lf", &x(i,j));
	}
#else
	for (int i = 0; i < n; i++) {   // �f�[�^
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%f", &x(i, j));
	}
#endif

	PCA pca;
	
	pca.set(variablesNum);
	int stat = pca.fit(x, true);


	/*
	�听�� 2.551440 �W�� 0.601096 0.092426 0.563442 -0.559172
	�听�� 0.987518 �W�� -0.050663 0.995047 -0.080488 0.028908
	�听�� 0.342736 �W�� 0.038354 0.035656 0.679496 0.731808
	�听�� 0.118306 �W�� 0.796646 -0.008174 -0.462969 0.388520
	*/
	pca.Report();

#ifdef USE_GNUPLOT
	gnuPlot plot1(std::string(GNUPLOT_PATH), 1);
	plot1.set_label_x("�听��");
	plot1.set_label_y("�ŗL�l");
	plot1.plot_lines(Matrix<dnn_double>(pca.component), std::vector<std::string>());
	plot1.draw();
#endif

}

