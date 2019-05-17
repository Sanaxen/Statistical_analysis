#ifndef _gen_test_data_H

#define _gen_test_data_H

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

inline void test_function(tiny_dnn::vec_t& x, tiny_dnn::vec_t& y)
{
	float y1 = sin(x[0]);
	float y2 = cos(x[1]);
	float y3 = sin(x[0]) + cos(3.0*x[1]);
	y.push_back(y1);
	y.push_back(y2);
	y.push_back(y3);
}

inline void make_data_set(std::string&  csvfile, int datanum = 1000)
{
	FILE* fp = fopen(csvfile.c_str(), "w");
	for (int i = 0; i < datanum; i++)
	{
		tiny_dnn::vec_t x;
		tiny_dnn::vec_t y;
		float x1 = 4 * M_PI*i / datanum;
		float x2 = 4 * M_PI*i / datanum - M_PI;
		x.push_back(x1);
		x.push_back(x2);

		test_function(x, y);
		fprintf(fp, "%f,%f,%f,%f,%f\n", x[0], x[1], y[0], y[1], y[2]);
	}
	fclose(fp);
}

inline int filelist_to_csv(std::string& csvfilename, std::string& dir, bool is_train, int class_num = 10, bool header = false, std::string& normalize = std::string("zscore"))
{
	FILE* fp = fopen((dir + "\\train_image_list.txt").c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}
	char image_file[640];
	fgets(image_file, 640, fp);
	fclose(fp);

	bool isImage = false;
	bool isCsv = false;
	bool islabel = false;

	if (strstr(image_file, ".bmp") || strstr(image_file, ".BMP")
		|| strstr(image_file, ".jpg") || strstr(image_file, ".JPEG")
		|| strstr(image_file, ".png") || strstr(image_file, ".PNG")
		)
	{
		isImage = true;
	}
	if (strstr(image_file, ".csv") || strstr(image_file, ".CSV"))
	{
		isCsv = true;
	}

	fp = fopen((dir + "\\train_label_list.txt").c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}
	char buf[640];
	fgets(buf, 640, fp);
	fclose(fp);

	int number;
	islabel = ( 1 == sscanf(buf, "%d\n", &number));

	if (class_num > 2 && isImage && islabel)
	{
		images_labes_to_csv(csvfilename, dir, is_train, class_num);
		return 0;
	}
	if (class_num > 2 && isCsv && islabel)
	{
		csvs_labes_to_csv(csvfilename, dir, is_train, class_num, header, normalize);
		return 0;
	}

	printf("unsupport file format!\n");
	return 1;
}

#endif
