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

#endif
