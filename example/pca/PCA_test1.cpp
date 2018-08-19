#include <iostream>
/*
å¬•ª 0.118306 ŒW” 0.796646 -0.008174 -0.462969 0.388520
å¬•ª 0.987518 ŒW” -0.050663 0.995047 -0.080488 0.028908
å¬•ª 2.551440 ŒW” 0.601096 0.092426 0.563442 -0.559172
å¬•ª 0.342736 ŒW” 0.038354 0.035656 0.679496 0.731808
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
	fscanf(fp, "%d %d", &variablesNum, &n);   // •Ï”‚Ì”‚Æƒf[ƒ^‚Ì”

	x = Matrix<dnn_double>(n, variablesNum);

	for (int i = 0; i < n; i++) {   // ƒf[ƒ^
		for (int j = 0; j < variablesNum; j++)
			fscanf(fp, "%lf", &x(i,j));
	}

	PCA pca;
	
	pca.set(variablesNum);
	int stat = pca.fit(x, true);


	/*
	å¬•ª 2.551440 ŒW” 0.601096 0.092426 0.563442 -0.559172
	å¬•ª 0.987518 ŒW” -0.050663 0.995047 -0.080488 0.028908
	å¬•ª 0.342736 ŒW” 0.038354 0.035656 0.679496 0.731808
	å¬•ª 0.118306 ŒW” 0.796646 -0.008174 -0.462969 0.388520
	*/
	pca.Report();

	//if (stat == 0) {
	//	for (int i = 0; i < variablesNum; i++) {
	//		printf("å¬•ª %f", pca.component[i]);
	//		printf(" ŒW”");
	//		for (int j = 0; j < variablesNum; j++)
	//			printf(" %f", coef(i,j));
	//		printf("\n");
	//	}
	//}
	//else
	//	printf("error:%d\n", stat);

	//for (int i = 0; i < variablesNum; i++)
	//{
	//	pca.getEigen().getRightVector(i)[0].print("ŒÅ—LƒxƒNƒgƒ‹");
	//}

	//pca.variance_covariance().print("‘ŠŠÖs—ñ");
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

