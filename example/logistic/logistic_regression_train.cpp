#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#define main logistic_regression_train
#include "train.c"
#undef main


int main(int argc, char** argv)
{
	printf("logistic_regression START\n");

	int argc2 = 0;
	char argv2[32][640];


	std::string file, model, output;
	int s = 0;
	double c = 0.0;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--L1") {
			s = 6;
			c = atof(argv[count + 1]);
			continue;
		}
		if (argname == "--L2") {
			s = 0;
			c = atof(argv[count + 1]);
			continue;
		}
		if (argname == "--L2dual") {
			s = 7;
			c = atof(argv[count + 1]);
			continue;
		}
		if (argname == "--file") {
			file = argv[count + 1];
			continue;
		}
		if (argname == "--model") {
			model = argv[count + 1];
			continue;
		}
		if (argname == "--output") {
			output = argv[count + 1];
			continue;
		}
	}


	strcpy(argv2[0], "");
	strcpy(argv2[1], "-s");
	sprintf(argv2[2], "%d", s);

	strcpy(argv2[3], "-c");
	sprintf(argv2[4], "%f", c);
	strcpy(argv2[5], file.c_str());
	strcpy(argv2[6], model.c_str());
	argc2 = 7;

	logistic_regression_train(argc2, (char**)argv2);

	printf("logistic_regression END\n\n");
	return 0;
}