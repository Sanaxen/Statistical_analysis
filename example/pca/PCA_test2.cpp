#include <iostream>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"

int main()
{
	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;

	FILE* fp = fopen("2-3c.csv", "r");
	if (fp == NULL)
	{
		return -1;
	}
	variablesNum = 3;
	n = 38;
	char buf[256];
	fgets(buf, 256, fp);

	x = Matrix<dnn_double>(n, variablesNum);
	int id;
	for (int i = 0; i < n; i++) {   // �f�[�^
		fscanf(fp, "%d,", &id);
		for (int j = 0; j < variablesNum - 1; j++)
		{
			dnn_double s;
			fscanf(fp, "%lf ,", &s);
			x(i, j) = s;
		}
		{
			dnn_double s;
			fscanf(fp, "%lf", &s);
			x(i, variablesNum - 1) = s;
		}
	}
	x.print();

	PCA pca2;
	pca2.set(variablesNum);

	int stat = pca2.fit(x, true);

	pca2.Report();

	pca2.principal_component().print("�听��");
	pca2.principal_component().print_csv("test.csv");

	stat = pca2.fit(x, true, false);

	pca2.Report();

	pca2.principal_component().print("�听��");
	pca2.principal_component().print_csv("test.csv");

	//Matrix<dnn_double>&w = pca2.whitening();
	//Matrix<dnn_double> pca_w(w.m, w.n);
	//for (int i = 0; i < w.m; i++)
	//{
	//	for (int j = 0; j < w.n; j++)
	//	{
	//		double s = 0.0;
	//		for (int k = 0; k < variablesNum; k++)
	//		{
	//			s += w(i, k)*pca2.coef(j, k);
	//		}
	//		pca_w(i, j) = s;
	//	}
	//}
	//pca2.variance_covariance().print("���֍s��");

	//for (int i = 0; i < variablesNum; i++)
	//{
	//	pca2.getEigen().getRightVector(i)[0].print("�ŗL�x�N�g��");
	//	//pca.getEigen().getRightVector(i)[1].print("�ŗL�x�N�g��");
	//}
	////w.print();
	//pca_w.print();
	////pca_w.print_csv("test.csv");

	//fp = fopen("test.csv", "w");
	//for (int i = 0; i < w.m; i++)
	//{
	//	for (int j = 0; j < w.n-1; j++)
	//	{
	//		fprintf(fp, "%f,", pca_w(i, j));
	//	}
	//	fprintf(fp, "%f\n", pca_w(i, w.n - 1));
	//}
	//fclose(fp);

	//printf("��^��(%%)\n");
	//for (int j = 0; j < w.n; j++)
	//{
	//	printf("%f%%\n", 100*pca2.contribution_rate(j));
	//}
	//pca2.mean.print("mean");
	//pca2.variance.print("variance");
}

