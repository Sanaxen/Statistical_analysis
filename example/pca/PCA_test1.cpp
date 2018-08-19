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

	for (int i = 0; i < n; i++) {   // データ
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%lf", &x(i,j));
	}

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

	//if (stat == 0) {
	//	for (int i = 0; i < variablesNum; i++) {
	//		printf("主成分 %f", pca.component[i]);
	//		printf(" 係数");
	//		for (int j = 0; j < variablesNum; j++)
	//			printf(" %f", coef(i,j));
	//		printf("\n");
	//	}
	//}
	//else
	//	printf("error:%d\n", stat);

	//for (int i = 0; i < variablesNum; i++)
	//{
	//	pca.getEigen().getRightVector(i)[0].print("固有ベクトル");
	//}

	//pca.variance_covariance().print("相関行列");
	//{
	//	//pca.getEigen().getImageValue().print("Im");
	//	Matrix<dnn_double>& value = pca.getEigen().getRealValue();
	//	//value.print();
	//	for (int i = 0; i < variablesNum; i++)
	//	{
	//		std::vector<Matrix<dnn_double>>&tmp2 = pca.getEigen().getRightVector(i);

	//		//tmp2[1].print("I,m");
	//		(pca.variance_covariance()*tmp2[0] - value(0, i) * tmp2[0]).chop(1.0e-6).print("check");
	//	}
	//}
}

