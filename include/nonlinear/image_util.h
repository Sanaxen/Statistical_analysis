#ifndef _IMAGE_UTIL_H
#define _IMAGE_UTIL_H

#include "../../../include/util/Image.hpp"

template<class T>
inline void image_to_vec(const T* img, const int w, const int h, const int channel, tiny_dnn::vec_t& vec, const float_t scale_min = -1.0, const float_t scale_max = -1.0)
{
	bool use_scale = (scale_min < scale_max);

	for (int c = 0; c < channel; c++)
	{
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				const int ii = y*w + x;
				if (use_scale)
				{
					vec[c * h*w + y*w + x] = scale_min + (scale_max - scale_min) *img[3 * ii + c] / 255.0;
				}
				else
				{
					vec[c * h*w + y*w + x] = img[3 * ii + c];
				}
			}
		}
	}
}

inline int load_images(std::string& image_filelist, std::string& dir, int channel, std::vector<tiny_dnn::vec_t>& images)
{
	char buf[640];
	FILE* fp = fopen(image_filelist.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}
	while (fgets(buf, 640, fp) != NULL)
	{
		char* p = strchr(buf, '\n');
		if (p)*p = '\0';

		std::string path = dir + "\\" + buf;
		Image* img = readImage((char*)path.c_str());
		unsigned char* img_f = ImageTo<unsigned char>(img);

		tiny_dnn::vec_t tmp(img->height*img->width*channel);
		image_to_vec<unsigned char>(img_f, img->width, img->height, channel, tmp);

		images.push_back(tmp);

		delete img;
		delete[] img_f;
	}
	fclose(fp);

	return 0;
}

inline int load_labels(std::string& label_filelist, int class_num, std::vector<tiny_dnn::vec_t>& labels)
{
	char buf[640];
	FILE* fp = fopen(label_filelist.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}
	while (fgets(buf, 640, fp) != NULL)
	{
		int label;
		sscanf(buf, "%d", &label);

		tiny_dnn::vec_t tmp(class_num, 0);
		tmp[label] = 1;

		labels.push_back(tmp);
	}
	fclose(fp);
	return 0;
}
inline int load_labels(std::string& label_filelist, int class_num, std::vector<int>& labels)
{
	char buf[640];
	FILE* fp = fopen(label_filelist.c_str(), "r");
	if (fp == NULL)
	{
		return -1;
	}
	while (fgets(buf, 640, fp) != NULL)
	{
		int label;
		sscanf(buf, "%d", &label);
		labels.push_back(label);
	}
	fclose(fp);
	return 0;
}


/**
image_dataset-+
              |
              +-- test-+-image1.bmp
              |        | image2.bmp
              |        |  .......
			  ..
              +-- train-+-imageA.bmp
              |         | imageB.bmp
              |         | .......
			  |
			  +-test_image_list.txt
			  +-test_label_list.txt
			  +-train_image_list.txt
			  +-train_label_list.txt

test_image_list.txt
image1.bmp
image2.bmp
test_label_list.txt
0
1
**/

/**
std::string dir = "image_dataset";

images_labes_to_csv(std::string("train.csv"), dir, true, channel, class_num);
images_labes_to_csv(std::string("test.csv"), dir, false, channel, class_num);
}
**/
inline void images_labes_to_csv(std::string& csvfilename, std::string& dir, bool is_train, int channel = 3, int class_num = 10)
{
	std::vector<tiny_dnn::vec_t> images;
	std::vector<int> labels;

	if (is_train)
	{
		load_images(dir + "\\train_image_list.txt", dir + "\\train", channel, images);
		load_labels(dir + "\\train_label_list.txt", class_num, labels);
	}
	else
	{
		load_images(dir + "\\test_image_list.txt", dir + "\\test", channel, images);
		load_labels(dir + "\\test_label_list.txt", class_num, labels);
	}
	printf("%d %d\n", images.size(), labels.size());
	FILE* fp = fopen(csvfilename.c_str(), "w");
	for (int i = 0; i < images.size(); i++)
	{
		fprintf(fp, "%d,", labels[i]);
		for (int k = 0; k < images[i].size(); k++)
		{
			fprintf(fp, "%.4f", images[i][k]);
			if (k == images[i].size() - 1)
			{
				fprintf(fp, "\n");
			}
			else
			{
				fprintf(fp, ",");
			}
		}
	}
	fclose(fp);
}

#endif
