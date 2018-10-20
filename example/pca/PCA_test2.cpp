#include <iostream>

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/pca.h"
#include "../../include/util/csvreader.h"

int main(int argc, char** argv)
{
	Matrix<dnn_double> x, coef;
	std::vector<dnn_double> component;
	int n, variablesNum;

	std::string csvfile("2-3c.csv");

	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
		}
	}

	CSVReader csv(csvfile, ',', header);
	x = csv.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			x = x.removeCol(0);
		}
	}
	variablesNum = x.n;
	n = x.m;
	x.print();

	PCA pca2;
	pca2.set(variablesNum);

	int stat = pca2.fit(x, true);

	pca2.Report();

	pca2.principal_component().print("Žå¬•ª");
	pca2.principal_component().print_csv("output1.csv");

	stat = pca2.fit(x, true, false);

	pca2.Report();

	pca2.principal_component().print_csv("output2.csv");
}

