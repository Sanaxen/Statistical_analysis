#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>


#define main logistic_regression_predict
#include "predict.c"
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
	strcpy(argv2[1], "-b");
	sprintf(argv2[3], "%d", 1);
	strcpy(argv2[5], file.c_str());
	strcpy(argv2[6], model.c_str());
	strcpy(argv2[7], output.c_str());
	argc2 = 8;

	logistic_regression_predict(argc2, (char**)argv2);

	printf("logistic_regression END\n\n");
	return 0;
}