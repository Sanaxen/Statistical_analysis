#include <iostream>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"
#include "../../include/util/csvreader.h"

int main()
{
	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;

#if 10
	CSVReader csv("2-3c.csv");
	x = csv.toMat_removeEmptyRow();
	x = x.removeCol(0);
	variablesNum = x.n;
	n = x.m;

#else
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
	for (int i = 0; i < n; i++) {   // ƒf[ƒ^
		fscanf(fp, "%d,", &id);
		for (int j = 0; j < variablesNum - 1; j++)
		{
			double s;
			fscanf(fp, "%f ,", &s);
			x(i, j) = s;
		}
		{
			double s;
			fscanf(fp, "%lf", &s);
			x(i, variablesNum - 1) = s;
		}
	}
#endif
	x.print();

	PCA pca2;
	pca2.set(variablesNum);

	int stat = pca2.fit(x, true);

	pca2.Report();

	pca2.principal_component().print("Žå¬•ª");
	pca2.principal_component().print_csv("test.csv");

	stat = pca2.fit(x, true, false);

	pca2.Report();

	pca2.principal_component().print_csv("test.csv");
}

