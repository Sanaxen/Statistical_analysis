#ifndef _PLOT_H__
#define _PLOT_H__

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/statistical/LinearRegression.h"
#include <string>

//#define GNUPLOT_PATH "\"C:\\Program Files\\gnuplot\\bin\\wgnuplot.exe\""
#define GNUPLOT_PATH "start wgnuplot.exe -persist"

#pragma warning( disable : 4244 ) 

class SJIStoUTF8
{
	inline bool convSJIStoUTF8(BYTE* pSource, BYTE* pDist, int* pSize)
	{
		*pSize = 0;

		// Convert SJIS -> UTF-16
		const int nSize = ::MultiByteToWideChar(CP_ACP, 0, (LPCSTR)pSource, -1, NULL, 0);

		BYTE* buffUtf16 = new BYTE[nSize * 2 + 2];
		::MultiByteToWideChar(CP_ACP, 0, (LPCSTR)pSource, -1, (LPWSTR)buffUtf16, nSize);

		// Convert UTF-16 -> UTF-8
		const int nSizeUtf8 = ::WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)buffUtf16, -1, NULL, 0, NULL, NULL);
		if (!pDist) {
			*pSize = nSizeUtf8;
			delete buffUtf16;
			return true;
		}

		BYTE* buffUtf8 = new BYTE[nSizeUtf8 * 2];
		ZeroMemory(buffUtf8, nSizeUtf8 * 2);
		::WideCharToMultiByte(CP_UTF8, 0, (LPCWSTR)buffUtf16, -1, (LPSTR)buffUtf8, nSizeUtf8, NULL, NULL);

		*pSize = lstrlenA((char*)(buffUtf8));
		memcpy(pDist, buffUtf8, *pSize);

		delete buffUtf16;
		delete buffUtf8;

		return true;
	}

	/*
	* convert: sjis -> utf8
	*/
	inline bool sjis2utf8(BYTE* source, BYTE** dest) {
		// Calculate result size
		int size = 0;
		convSJIStoUTF8(source, NULL, &size);

		// Peform convert
		*dest = new BYTE[size + 1];
		ZeroMemory(*dest, size + 1);
		convSJIStoUTF8(source, *dest, &size);

		return true;
	}
	BYTE* Dest;

public:
	SJIStoUTF8() {}
	~SJIStoUTF8()
	{
		delete[] Dest;
	}
	inline void conv(char* sjis)
	{
		sjis2utf8((BYTE*)sjis, &Dest);
	}
	inline char* get()
	{
		return (char*)Dest;
	}
};

inline void convf(const char* filename)
{
	FILE* fp = fopen(filename, "r");
	if (!fp) return;

	char fname[640];
	sprintf(fname, "%s_%s", filename, ".tmp");

	FILE* fp2 = fopen(fname, "w");
	if (!fp2) return;

	SJIStoUTF8 sjis2utf8;
	sjis2utf8.conv("set encoding utf8\n");
	fprintf(fp2, "%s", sjis2utf8.get());

	char buf[1024];
	while (fgets(buf, 1024, fp) != NULL)
	{
		{
			SJIStoUTF8 sjis2utf8;
			sjis2utf8.conv(buf);
			fprintf(fp2, "%s", sjis2utf8.get());
		}
	}
	fclose(fp);
	fclose(fp2);


	fp = fopen(filename, "w");
	if (!fp) return;

	fp2 = fopen(fname, "r");
	if (!fp2) return;

	while (fgets(buf, 1024, fp2) != NULL)
	{
		fprintf(fp, "%s", buf);
	}
	fclose(fp);
	fclose(fp2);
	_unlink(fname);
}

class ScatterWrk
{
public:
	float depth;
	int id=-1;
	ScatterWrk()
	{
		depth = 0;
	}
	bool operator<(const ScatterWrk& right) const {
		return depth < right.depth;
	}
};

class gnuPlot
{
	bool multiplot = false;
	std::string gnuplot_exe_path;
	int id;
	FILE* script = NULL;

	int plot_count = 0;
	bool replot = false;
	inline void script_reopen()
	{
		if (script == NULL)
		{
			replot = true;
			script = fopen(script_name.c_str(), "a");
		}
		char buff[512];
		sprintf(buff, "plot_%04d(%03d).dat", id, plot_count);
		data_name = buff;
	}

	bool capture_image = false;
	int capture_winsize[2] = { 640,480 };
public:
	bool histogram_gradation = true;
	std::string save_image_name = "image.png";
	std::string script_name = "plot.plt";
	std::string data_name = "plot.dat";


	std::string title = "";
	float linewidth = 2.0;
	float pointsize = 1.0;

	std::string linecolor = "";

	gnuPlot(std::string& gnuplot_exe_path_, const int script_id=-1)
	{
		gnuplot_exe_path = gnuplot_exe_path_;

		id = script_id;
		if (script_id < 0) id = rand();

		//save_image = save_image_;

		char buff[512];
		sprintf(buff, "plot_%04d.plt", id);
		script_name = buff;

		sprintf(buff, "plot_%04d.dat", id);
		data_name = buff;

		sprintf(buff, "image_%04d.png", id);
		data_name = buff;

		script = fopen(script_name.c_str(), "w");


		fprintf(script, "set border lc rgb \"black\"\n");
		fprintf(script, "set grid lc rgb \"#D8D8D8\" lt 2\n");
		//fprintf(script, "set colorsequence default\n");
		fprintf(script, "set colorsequence podo\n");

		//if (save_image)
		//{
		//	fprintf(script, "set terminal pngcairo\n");
		//}
	}

	void set_capture(int win[2], std::string& imagefile)
	{
		capture_image = true;
		capture_winsize[0] = win[0];
		capture_winsize[1] = win[1];
		save_image_name = imagefile;

		fprintf(script, "set term windows size %d,%d\n", capture_winsize[0], capture_winsize[1]);
		fprintf(script, "set term pngcairo size %d,%d\n", capture_winsize[0], capture_winsize[1]);
		fprintf(script, "set output \"%s\"\n", imagefile.c_str());
	}

	FILE* fp()
	{
		return script;
	}
	void set_label(float pos_x, float pos_y, int number, char* text)
	{
		script_reopen();
		if (script == NULL) return;

		std::string txt = text;
		if (txt.c_str()[0] != '\"')
		{
			txt = "\"" + txt + "\"";
		}
		fprintf(script, "set label %d at graph %.2f,%.2f %s\n", number, pos_x, pos_y, txt.c_str());
		//fprintf(script, "set label %d font %s\n", number, "'Arial,30'");
	}

	void set_palette(char* palette)
	{
		script_reopen();
		if (script == NULL) return;
		if (palette && *palette != '\0')
		{
			fprintf(script, "set palette %s\n", palette);
		}
	}
	void set_title(char* title)
	{
		if (script == NULL) return;
		fprintf(script, "set title \"%s\"\n", title);
	}
	void set_range_x(float x_min, float x_max)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set xrange[%f:%f]\n", x_min, x_max);
	}
	void set_range_y(float y_min, float y_max)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set yrange[%f:%f]\n", y_min, y_max);
	}
	void set_label_x(char* lable)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set xlabel \"%s\"\n", lable);
	}
	void set_label_y(char* lable)
	{
		fprintf(script, "set ylabel \"%s\"\n", lable);
	}
	void autoscale(int axis='y')
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set autoscale %c\n", axis);
	}

	void plot_lines(Matrix<dnn_double>&X, std::vector<std::string>& headers, int maxpoint=-1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		std::string title = "";
		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}

		if (headers.size())
		{
			title = headers[0];
			if (title.c_str()[0] != '\"')
			{
				title = (std::string("t \"") + headers[0] + std::string("\""));
			}
			else
			{
				title = "t " + headers[0];
			}
		}

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script,"%s '%s' %s using %d %s with lines linewidth %.1f %s\n",
			plot, data_name.c_str(), every.c_str(), 1, title.c_str(), linewidth, linecolor.c_str());

		for ( int i = 1; i < X.n; i++ )
		{
			title = "";
			if (headers.size() > i)
			{
				title = headers[i];
				if (title.c_str()[0] != '\"')
				{
					title = (std::string("t \"") + headers[i] + std::string("\""));
				}
				else
				{
					title = "t " + headers[i];
				}
			}
			fprintf(script,"replot '%s' %s using %d %s with lines linewidth %.1f %s\n",
				data_name.c_str(), every.c_str(), i+1, title.c_str(), linewidth, linecolor.c_str());
		}
		linecolor = "";
		plot_count++;
	}

	void plot_lines2(Matrix<dnn_double>&X, std::vector<std::string>& headers, int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		std::string title = "";
		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}
		printf("%s\n", every.c_str());

		if (headers.size())
		{
			title = headers[0];
			if (title.c_str()[0] != '\"')
			{
				title = (std::string("t \"") + headers[0] + std::string("\""));
			}
			else
			{
				title = "t " + headers[0];
			}
		}

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' %s using 1:%d %s with lines linewidth %.1f %s\n",
			plot, data_name.c_str(), every.c_str(), 2, title.c_str(), linewidth, linecolor.c_str());

		for (int i = 1; i < X.n-1; i++)
		{
			title = "";
			if (headers.size() > i)
			{
				title = headers[i];
				if (title.c_str()[0] != '\"')
				{
					title = (std::string("t \"") + headers[i] + std::string("\""));
				}
				else
				{
					title = "t " + headers[i];
				}
			}
			fprintf(script, "replot '%s' %s using 1:%d %s with lines linewidth %.1f %s\n",
				data_name.c_str(), every.c_str(), i + 2, title.c_str(), linewidth, linecolor.c_str());
		}
		linecolor = "";
		plot_count++;
	}

	void plot_lines2d(Matrix<dnn_double>&X, std::string& name, int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		std::string title = "";
		std::string every = "";

		if (maxpoint < 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}

		title = name;
		if (title.c_str()[0] != '\"')
		{
			title = (std::string("t \"") + name + std::string("\""));
		}
		else
		{
			title = "t " + name;
		}

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' %s using 1:%d %s with lines linewidth %.1f %s\n",
			plot, data_name.c_str(), every.c_str(), 2, title.c_str(), linewidth, linecolor.c_str());

		linecolor = "";
		plot_count++;
	}

	void plot_points(Matrix<dnn_double>&X, std::vector<std::string>& headers, int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		std::string title = "";
		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			every = "every " + std::to_string(every_num);
		}
		if (headers.size())
		{
			title = headers[0];
			if (title.c_str()[0] != '\"')
			{
				title = (std::string("t \"") + headers[0] + std::string("\"")).c_str();
			}
			else
			{
				title = "t " + headers[0];
			}
		}

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}

		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' %s using %d %s with points pointsize %.1f %s\n",
			plot, data_name.c_str(), every.c_str(), 1, title, pointsize, linecolor.c_str());

		for (int i = 1; i < X.n; i++)
		{
			title = "";
			if (headers.size() > i)
			{
				title = headers[i];
				if (title.c_str()[0] != '\"')
				{
					title = (std::string("t \"") + headers[i] + std::string("\"")).c_str();
				}
				else
				{
					title = "t " + headers[i];
				}
			}
			fprintf(script, "replot '%s' %s using %d %s with points pointsize %.1f %s\n",
				data_name.c_str(), every.c_str(), i+1, title, pointsize, linecolor.c_str());
		}
		linecolor = "";
		plot_count++;
	}

	void multi_scatter(Matrix<dnn_double>&X, int x_num, std::vector<std::string>& headers, int grid, int pointtype = 6, char* palette = "rgbformulae 34,35,36", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		/*
		set term png size 3024,3024
		set output "figure.png"
		fprintf(script, "set term png size 3024,3024\n");
		fprintf(script, "set output \"multi_scatter.png\"\n");
		*/

		//if (scatter_back_color_dark_mode)
		//{
		//	back_color_dark();
		//}

		multiplot = true;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "set multiplot layout %d,%d\n", X.n, X.n);
		fprintf(script, "set nokey\n");
		fprintf(script, "unset xtics\n");
		fprintf(script, "unset ytics\n");
		if (palette)
		{
			set_palette(palette);
			fprintf(script, "unset colorbox\n");
		}


		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}

		printf("x_num:%d X.n:%d\n", x_num, X.n);

		X.print_csv((char*)data_name.c_str());
		for (int i = 0; i < x_num; i++)
		{
			for (int j = x_num; j < X.n; j++)
			{
				std::string x, y;

				if (headers.size())
				{
					y = headers[j];
					if (y.c_str()[0] != '\"')
					{
						y = (std::string("\"") + headers[j] + std::string("\""));
					}

					x = headers[j];
					if (x.c_str()[0] != '\"')
					{
						x = (std::string("\"") + headers[i] + std::string("\""));
					}
				}

				fprintf(script, "unset xlabel\n");
				fprintf(script, "unset ylabel\n");
#if 0
				if (j > i)
				{
					fprintf(script, "unset border\n");
					fprintf(script, "plot - 1 notitle lc rgb \"white\"\n");
					fprintf(script, "set border\n");
					continue;
				}
#else
				if (j == i)
				{
					Matrix<dnn_double> &h = Histogram(X.Col(j), 10);
					char histogram_name[256];
					sprintf(histogram_name, "plot_(%d)%d_hist.dat", plot_count, i);
					h.print_csv(histogram_name);

					fprintf(script, "set xrange[*:*]\n");
					fprintf(script, "set yrange[*:*]\n");
					fprintf(script, "set style fill solid border  lc rgb \"dark-gray\"\n");

					if (palette)
					{
						//fprintf(script, "plot '%s' using 1:2:2 %s with boxes linewidth %.1f pal\n",
						//	histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/);
						fprintf(script, "plot '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
							histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/, linecolor.c_str());
					}
					else
					{
						fprintf(script, "plot '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
							histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/, linecolor.c_str());
					}
					continue;
				}
#endif

#if 10
				//fprintf(script, "set style fill  transparent solid 0.85 noborder\n");
				scatter(X, i, j, pointsize, grid, headers, 5, palette);
#else

				fprintf(script, "set xlabel %s\n", x.c_str());
				fprintf(script, "set ylabel %s\n", y.c_str());
				
				if (scatter_xyrange_setting)
				{
					double max_x = X.Col(j).Max();
					double min_x = X.Col(j).Min();
					double max_y = X.Col(i).Max();
					double min_y = X.Col(i).Min();
					double rate_x = (max_x - min_x) * 0.1 + 0.001;
					double rate_y = (max_y - min_y) * 0.1 + 0.001;
					fprintf(script, "set xrange[%.3f:%.3f]\n", min_x - rate_x, max_x + rate_x);
					fprintf(script, "set yrange[%.3f:%.3f]\n", min_y - rate_x, max_y + rate_x);
				}
				//{
				//	multiple_regression mreg;

				//	mreg.set(1);
				//	mreg.fit(X.Col(j), X.Col(i));


				//	double max_x = X.Col(j).Max();
				//	double min_x = X.Col(j).Min();
				//	double step = (max_x - min_x) / 3.0;
				//	Matrix<dnn_double> x(4, 2);
				//	Matrix<dnn_double> v(1, 1);
				//	for (int i = 0; i < 4; i++)
				//	{
				//		v(0, 0) = min_x + i*step;
				//		x(i, 0) = v(0, 0);
				//		x(i, 1) = mreg.predict(v);
				//	}
				//	std::string line_header_names;
				//	line_header_names = "t \"linear regression\"";

				//	char buf[256];
				//	sprintf(buf, "%s_%d_%d.dat", data_name.c_str(), i, j);
				//	x.print_csv(buf);

				//	fprintf(script, "plot '%s' %s using 1:%d %s with lines linewidth %.1f %s\n",
				//		buf, every.c_str(), 2, line_header_names.c_str(), linewidth, linecolor.c_str());
				//}
				fprintf(script, "set style fill  transparent solid 0.85 noborder\n");
				if (palette)
				{
					//fprintf(script, "plot '%s' %s using %d:%d:%d %s with points pointsize %.1f pt %d lc palette\n",
					//	data_name.c_str(), every.c_str(), i + 1, j + 1, j+1, title.c_str(), pointsize, pointtype);
					fprintf(script, "plot '%s' %s using %d:%d:%d %s with circles pal\n",
						data_name.c_str(), every.c_str(), j + 1, i + 1, j + 1, title.c_str());
				}
				else
				{
					//fprintf(script, "plot '%s' %s using %d:%d %s with points pointsize %.1f pt %d\n",
					//	data_name.c_str(), every.c_str(), i + 1, j + 1, title.c_str(), pointsize, pointtype);
					fprintf(script, "plot '%s' %s using %d:%d:(-1) %s with circles lc rgb \"sea-green\"\n",
						data_name.c_str(), every.c_str(), j + 1, i + 1, title.c_str());
				}
#endif
				fflush(script);
			}
		}
		fprintf(script, "unset multiplot\n");
		plot_count++;
	}

	void multi_scatter(Matrix<dnn_double>&X, std::vector<std::string>& headers, int grid, int pointtype = 6, char* palette = "rgbformulae 34,35,36", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		/*
		set term png size 3024,3024
		set output "figure.png"
		fprintf(script, "set term png size 3024,3024\n");
		fprintf(script, "set output \"multi_scatter.png\"\n");
		*/

		//if (scatter_back_color_dark_mode)
		//{
		//	back_color_dark();
		//}

		multiplot = true;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "set multiplot layout %d,%d\n", X.n, X.n);
		fprintf(script, "set nokey\n");
		fprintf(script, "unset xtics\n");
		fprintf(script, "unset ytics\n");
		if (palette)
		{
			set_palette(palette);
			fprintf(script, "unset colorbox\n");
		}


		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}

		bool scatter_use_label_ = scatter_use_label;
		X.print_csv((char*)data_name.c_str());
		for (int i = 0; i < X.n; i++)
		{
			for (int j = 0; j < X.n; j++)
			{
				std::string x, y;

				if (headers.size())
				{
					y = headers[i];
					if (y.c_str()[0] != '\"')
					{
						y = (std::string("\"") + headers[i] + std::string("\""));
					}

					x = headers[j];
					if (x.c_str()[0] != '\"')
					{
						x = (std::string("\"") + headers[j] + std::string("\""));
					}
				}

				fprintf(script, "unset xlabel\n");
				fprintf(script, "unset ylabel\n");
				if (j > i)
				{
					fprintf(script, "unset border\n");
					fprintf(script, "plot - 1 notitle lc rgb \"white\"\n");
					fprintf(script, "set border\n");
					continue;
				}
				if (j == i)
				{
					fprintf(script, "set style fill  transparent solid 1.0 noborder\n");

					Matrix<dnn_double> &h = Histogram(X.Col(j), 10);
					char histogram_name[256];
					sprintf(histogram_name, "plot_(%d)%d_hist.dat", plot_count, i);
					h.print_csv(histogram_name);

					fprintf(script, "set xrange[*:*]\n");
					fprintf(script, "set yrange[*:*]\n");
					fprintf(script, "set style fill solid border  lc rgb \"dark-gray\"\n");
					
					if (palette)
					{
						//fprintf(script, "plot '%s' using 1:2:2 %s with boxes linewidth %.1f pal\n",
						//	histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/);
						fprintf(script, "plot '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
							histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/, linecolor.c_str());
					}
					else
					{
						fprintf(script, "plot '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
							histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/, linecolor.c_str());
					}
					continue;
				}

#if 10
				scatter_use_label = scatter_use_label_;
				//fprintf(script, "set style fill  transparent solid 0.85 noborder\n");
				scatter(X, j, i, pointsize, grid, headers, 5, palette);
#else
				fprintf(script, "set xlabel %s\n", x.c_str());
				fprintf(script, "set ylabel %s\n", y.c_str());
				double max_x = X.Col(j).Max();
				double min_x = X.Col(j).Min();
				double max_y = X.Col(i).Max();
				double min_y = X.Col(i).Min();
				double rate_x = (max_x - min_x) * 0.1 + 0.001;
				double rate_y = (max_y - min_y) * 0.1 + 0.001;
				fprintf(script, "set xrange[%.3f:%.3f]\n", min_x - rate_x, max_x + rate_x);
				fprintf(script, "set yrange[%.3f:%.3f]\n", min_y - rate_x, max_y + rate_x);

				//{
				//	multiple_regression mreg;

				//	mreg.set(1);
				//	mreg.fit(X.Col(j), X.Col(i));


				//	double max_x = X.Col(j).Max();
				//	double min_x = X.Col(j).Min();
				//	double step = (max_x - min_x) / 3.0;
				//	Matrix<dnn_double> x(4, 2);
				//	Matrix<dnn_double> v(1, 1);
				//	for (int i = 0; i < 4; i++)
				//	{
				//		v(0, 0) = min_x + i*step;
				//		x(i, 0) = v(0, 0);
				//		x(i, 1) = mreg.predict(v);
				//	}
				//	std::string line_header_names;
				//	line_header_names = "t \"linear regression\"";

				//	char buf[256];
				//	sprintf(buf, "%s_%d_%d.dat", data_name.c_str(), i, j);
				//	x.print_csv(buf);

				//	fprintf(script, "plot '%s' %s using 1:%d %s with lines linewidth %.1f %s\n",
				//		buf, every.c_str(), 2, line_header_names.c_str(), linewidth, linecolor.c_str());
				//}
				fprintf(script, "set style fill  transparent solid 0.85 noborder\n");
				if (palette)
				{
					//fprintf(script, "plot '%s' %s using %d:%d:%d %s with points pointsize %.1f pt %d lc palette\n",
					//	data_name.c_str(), every.c_str(), i + 1, j + 1, j+1, title.c_str(), pointsize, pointtype);
					fprintf(script, "plot '%s' %s using %d:%d:%d %s with circles pal\n",
						data_name.c_str(), every.c_str(), j + 1, i + 1, j+1, title.c_str());
				}
				else
				{
					//fprintf(script, "plot '%s' %s using %d:%d %s with points pointsize %.1f pt %d\n",
					//	data_name.c_str(), every.c_str(), i + 1, j + 1, title.c_str(), pointsize, pointtype);
					fprintf(script, "plot '%s' %s using %d:%d:(-1) %s with circles lc rgb \"sea-green\"\n",
						data_name.c_str(), every.c_str(), j + 1, i + 1, title.c_str());
				}
#endif
				fflush(script);
			}
		}
		fprintf(script, "unset multiplot\n");
		plot_count++;
	}

	void multi_scatter_for_list(std::vector<int> indexs, Matrix<dnn_double>&X, std::vector<std::string>& headers, int grid, int pointtype = 6, char* palette = "rgbformulae 34,35,36", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");

		fprintf(script, "set multiplot layout %d,%d\n", indexs.size(), 1);
		fprintf(script, "set nokey\n");
		fprintf(script, "unset xtics\n");
		fprintf(script, "unset ytics\n");
		if (palette)
		{
			set_palette(palette);
		}


		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}


		X.print_csv((char*)data_name.c_str());
		for (int i = 0; i < indexs.size(); i++)
		{
			std::string x, y;

			if (headers.size())
			{
				y = headers[0];
				if (y.c_str()[0] != '\"')
				{
					y = (std::string("\"") + headers[0] + std::string("\""));
				}

				x = headers[indexs[i]];
				if (x.c_str()[0] != '\"')
				{
					x = (std::string("\"") + headers[indexs[i]] + std::string("\""));
				}
			}

			fprintf(script, "set xlabel %s\n", x.c_str());
			fprintf(script, "set ylabel %s\n", y.c_str());

			if (palette)
			{
				fprintf(script, "plot '%s' %s using %d:%d:%d %s with points pointsize %.1f pt %d lc palette\n",
					data_name.c_str(), every.c_str(), indexs[i], 1, 1, title.c_str(), pointsize, pointtype);
			}
			else
			{
				fprintf(script, "plot '%s' %s using %d:%d %s with points pointsize %.1f pt %d\n",
					data_name.c_str(), every.c_str(), indexs[i], 1, title.c_str(), pointsize, pointtype);
			}
			fflush(script);
		}
		fprintf(script, "unset multiplot\n");
		plot_count++;
	}

	bool scatter_back_color_dark_mode = false;
	bool scatter_density_mode = true;
	void back_color_dark()
	{
		fprintf(script,
			"set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb \"#dcdcdc\" fillstyle solid 1.0\n"
			"set grid lc rgb \"white\" lt 2\n"
			"set border lc rgb \"white\"\n"
		);
	}

	bool scatter_use_label = true;
	bool scatter_xyrange_setting = true;
	double scatter_circle_radius_screen = 0.01;
	void scatter(Matrix<dnn_double>&X, int col1, int col2, float point_size, int grid, std::vector<std::string>& headers, int pointtype = 6, char* palette = "rgbformulae 22, 13, -31", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		

		//fprintf(script, "set style circle radius screen %f\n", scatter_circle_radius_screen);
		fprintf(script, "set style circle radius graph %f\n", scatter_circle_radius_screen);
		if (scatter_density_mode)
		{
			fprintf(script, "set style fill  transparent solid 0.35 noborder\n");
		}
		else
		{
			fprintf(script, "set style fill  transparent solid 1.0 noborder\n");
		}
		fprintf(script, "set datafile separator \",\"\n");
		
		//ñ}ó·OFF
		fprintf(script, "set nokey\n");
		
		if (scatter_back_color_dark_mode)
		{
			back_color_dark();
		}
		Matrix<dnn_double> x = X.Col(col1);
		Matrix<dnn_double> y = X.Col(col2);

		double x_min = x.Min();
		double x_max = x.Max();
		double y_min = y.Min();
		double y_max = y.Max();

		double N = grid;
		if (grid <= 0) N = 10;
		double dx = (x_max - x_min) / (N - 1);
		double dy = (y_max - y_min) / (N - 1);

		x = x.appendCol(y);

		if (scatter_density_mode)
		{
			Matrix<dnn_double>& z = Matrix<dnn_double>().ones(N, N);
			for (int i = 0; i < x.m; i++)
			{
				int idx = (x(i, 0) - x_min) / dx;
				int idy = (x(i, 1) - y_min) / dy;
				if (idx < 0) idx = 0;
				if (idy < 0) idy = 0;
				if (idx >= N - 1) idx = N - 1;
				if (idy >= N - 1) idy = N - 1;
				z(idy, idx) += 1;
			}
			std::vector<ScatterWrk> wrk;
			for (int i = 0; i < x.m; i++)
			{
				int idx = (x(i, 0) - x_min) / dx;
				int idy = (x(i, 1) - y_min) / dy;
				if (idx < 0) idx = 0;
				if (idy < 0) idy = 0;
				if (idx >= N - 1) idx = N - 1;
				if (idy >= N - 1) idy = N - 1;
				ScatterWrk d;
				d.id = i;
				d.depth = z(idy, idx);
				wrk.push_back(d);
			}
			std::sort(wrk.begin(), wrk.end());
			Matrix<dnn_double>& z_tmp = Matrix<dnn_double>(x.m, 1);
			Matrix<dnn_double> x_tmp = x;
			for (int i = 0; i < x.m; i++)
			{
				z_tmp(i, 0) = wrk[i].depth;
				x_tmp(i, 0) = x(wrk[i].id, 0);
				x_tmp(i, 1) = x(wrk[i].id, 1);
			}
			x = x_tmp;
			z = z_tmp;

			if (palette)
			{
				x = x.appendCol(z);
				set_palette(palette);
			}
			else
			{
				command(std::string("rgb(r,g,b)=int(r)*65536+int(g)*256+int(b)"));
				x = x.appendCol(z);
			}
		}

		//y = y / (y.Max() - y.Min());
		//x = x.appendCol(y);
		x.print_csv((char*)data_name.c_str());

		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			every = "every " + std::to_string(every_num);
		}

		std::string label = "";
		if (headers.size())
		{
			label = "t " + headers[col2];
			if (headers[col2].c_str()[0] != '\"')
			{
				label = "t \"" + headers[col2] + "\"";
			}
		}
		std::string xx, yy;

		if (headers.size())
		{
			yy = headers[col2];
			if (yy.c_str()[0] != '\"')
			{
				yy = (std::string("\"") + headers[col2] + std::string("\""));
			}

			xx = headers[col1];
			if (xx.c_str()[0] != '\"')
			{
				xx = (std::string("\"") + headers[col1] + std::string("\""));
			}
		}

		if (scatter_use_label)
		{
			fprintf(script, "set xlabel %s\n", xx.c_str());
			fprintf(script, "set ylabel %s\n", yy.c_str());
		}
		scatter_use_label = true;
		if(scatter_xyrange_setting)
		{
			double max_x = x.Col(0).Max();
			double min_x = x.Col(0).Min();
			double max_y = x.Col(1).Max();
			double min_y = x.Col(1).Min();
			double rate_x = (max_x - min_x) * 0.1 + 0.001;
			double rate_y = (max_y - min_y) * 0.1 + 0.001;
			fprintf(script, "set xrange[%.3f:%.3f]\n", min_x - rate_x, max_x + rate_x);
			fprintf(script, "set yrange[%.3f:%.3f]\n", min_y - rate_y, max_y + rate_y);
		}
		scatter_xyrange_setting = true;
		const char* plot = (plot_count) ? "replot" : "plot";
		if (palette)
		{
			//fprintf(script, "set style circle radius graph 0.005\n");
			//fprintf(script, "%s '%s' %s using 1:2:3 %s with circles fs transparent solid 0.85 lw 0.1 pal\n",
			//	plot, data_name.c_str(), every.c_str(), label.c_str());

			fprintf(script, "#set style circle radius graph 0.005\n");

			if (scatter_density_mode)
			{
				fprintf(script, "%s '%s' %s using 1:2:3 %s with circles lw 0.1 pal\n",
					plot, data_name.c_str(), every.c_str(), label.c_str());
			}
			else
			{
				fprintf(script, "%s '%s' %s using 1:2:2 %s with circles lw 0.1 pal\n",
					plot, data_name.c_str(), every.c_str(), label.c_str());
			}
		}
		else
		{
			//fprintf(script, "%s '%s' %s using 1:2:(rgb(255,$3*255,$3*255)) %s with points pointsize %.1f pt %d rgb variable\n",
			//	plot, data_name.c_str(), every.c_str(), label.c_str(), point_size, pointtype);
			//fprintf(script, "%s '%s' %s using 1:2 %s with points pointsize %.1f pt %d\n",
			//	plot, data_name.c_str(), every.c_str(), label.c_str(), point_size, pointtype);
			if (scatter_density_mode)
			{
				fprintf(script, "%s '%s' %s using 1:2:(-1) %s with circles lc  rgb \"dark-orange\"\n",
					plot, data_name.c_str(), every.c_str(), label.c_str());
			}
			else
			{
				fprintf(script, "%s '%s' %s using 1:2:(-1) %s with circles lc  rgb \"dark-orange\"\n",
					plot, data_name.c_str(), every.c_str(), label.c_str());
			}
		}
		linecolor = "";
		plot_count++;
		scatter_back_color_dark_mode = false;
	}


	void plot_histogram(Matrix<dnn_double>&X, char* label_text =NULL, int shapiro_wilk_test=0)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		fprintf(script, "set style fill solid\n");
		fprintf(script, "set boxwidth 0.8 relative\n");

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}
		std::string label = "";
		if (label_text)
		{
			if (label_text[0] != '\"')
			{
				label = "t \"" + std::string(label_text) + "\"";
			}
			else
			{
				label = "t " + std::string(label_text);
			}
		}
		const char* plot = (plot_count) ? "replot" : "plot";

		if (histogram_gradation)
		{
			if (shapiro_wilk_test != 1)
			{
				fprintf(script, "%s for [i=51:51:-1] '%s' using 1:($2*i/51):($2*i/51) %s with boxes lc palette\n",
					plot, data_name.c_str(), label.c_str());

				fprintf(script, "%s for [i=50:1:-1] '%s' using 1:($2*i/51):($2*i/51) with boxes lc palette notitle\n",
					"replot", data_name.c_str());
			}
			else
			{
				fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"light-red\" linewidth %.1f %s\n",
					plot, data_name.c_str(), label.c_str(), linewidth, linecolor.c_str());
			}
		}
		else
		{
			if (shapiro_wilk_test != 1)
			{
				fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"chartreuse\" linewidth %.1f %s\n",
					plot, data_name.c_str(), label.c_str(), linewidth, linecolor.c_str());
			}
			else
			{
				fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"light-red\" linewidth %.1f %s\n",
					plot, data_name.c_str(), label.c_str(), linewidth, linecolor.c_str());
			}
		}
		linecolor = "";
		plot_count++;
	}

	void multi_histgram(std::string& imagefile, Matrix<dnn_double>&X, std::vector<std::string>& headers, std::vector<int>& residual_flag, int pointtype = 6, int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		/*
		set term png size 3024,3024
		set output "figure.png"
		fprintf(script, "set term png size 3024,3024\n");
		fprintf(script, "set output \"multi_scatter.png\"\n");
		*/
		multiplot = true;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "set multiplot layout %d,%d\n", (int)sqrt(X.n)+1, (int)sqrt(X.n) + 1);
		fprintf(script, "set nokey\n");
		fprintf(script, "unset xtics\n");
		fprintf(script, "unset ytics\n");

		fprintf(script, "unset colorbox\n");


		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			if (every_num != 0)
			{
				every = "every " + std::to_string(every_num);
			}
		}

		const char* plot = (plot_count) ? "replot" : "plot";

		X.print_csv((char*)data_name.c_str());
		for (int i = 0; i < X.n; i++)
		{
			//for (int j = 0; j < X.n; j++)
			{
				std::string x, y;

				if (headers.size())
				{
					y = headers[i];
					if (y.c_str()[0] != '\"')
					{
						y = (std::string("\"") + headers[i] + std::string("\""));
					}

					//x = headers[j];
					//if (x.c_str()[0] != '\"')
					//{
					//	x = (std::string("\"") + headers[j] + std::string("\""));
					//}
				}

				fprintf(script, "unset xlabel\n");
				fprintf(script, "unset ylabel\n");
				//if (j > i)
				//{
				//	fprintf(script, "unset border\n");
				//	fprintf(script, "plot - 1 notitle lc rgb \"white\"\n");
				//	fprintf(script, "set border\n");
				//	continue;
				//}
				//if (j == i)
				{
					fprintf(script, "set title %s\n", y.c_str());
					Matrix<dnn_double> &h = Histogram(X.Col(i), 30);
					char histogram_name[256];
					sprintf(histogram_name, "plot_(%d)%d_hist.dat", plot_count, i);
					h.print_csv(histogram_name);

					//fprintf(script, "set grid xtics mxtics ytics mytics\n");
					fprintf(script, "set style fill solid\n");
					fprintf(script, "set boxwidth 0.8 relative\n");

					if (histogram_gradation)
					{
						if (residual_flag[i])
						{
							fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"light-red\" linewidth %.1f %s\n",
								plot, histogram_name, (std::string("t ") + y).c_str(), 1/*linewidth*/, linecolor.c_str());
						}
						else
						{
							fprintf(script, "%s for [i=50:1:-1] '%s' using 1:($2*i/51):($2*i/51) with boxes lc palette notitle\n",
								plot, histogram_name);
						}
					}
					else
					{
						if (residual_flag[i])
						{
							fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"light-red\" linewidth %.1f %s\n",
								plot, histogram_name, (std::string("t ") + y).c_str(), 1/*linewidth*/, linecolor.c_str());
						}
						else
						{
							fprintf(script, "%s '%s' using 1:2 %s with boxes lc rgb \"chartreuse\" linewidth %.1f %s\n",
								plot, histogram_name, (std::string("t ") + y).c_str(), 1/*linewidth*/, linecolor.c_str());
						}
					}
					continue;
				}
				fprintf(script, "set xlabel %s\n", x.c_str());
				fprintf(script, "set ylabel %s\n", y.c_str());
				fflush(script);
			}
		}
		fprintf(script, "unset multiplot\n");
		plot_count++;
	}

	void command(std::string& command)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "%s\n", command.c_str());
	}

	void close()
	{
		if (script == NULL) return;
		fclose(script);
		script = NULL;
	}
	void draw(int pause=1000)
	{
		if (script == NULL) return;
		//if (save_image)
		//{
		//	fprintf(script, "set out \"%s\"\n", save_image_name.c_str());
		//	fprintf(script, "replot\n");
		//}
		//else
		{
			if (capture_image)
			{
				if (multiplot)
				{
					fprintf(script, "set multiplot\n");
				}
				pause = 0;
				fprintf(script, "set term windows size %d,%d\n", capture_winsize[0], capture_winsize[1]);
				fprintf(script, "set term pngcairo size %d,%d\n", capture_winsize[0], capture_winsize[1]);
				fprintf(script, "set output \"%s\"\n", save_image_name.c_str());
				fprintf(script, "replot\n");
				fprintf(script, "unset multiplot\n");
				multiplot = false;
			}

			fprintf(script, "pause %d\n", pause);
			//fprintf(script, "pause -1\n");
			//fprintf(script, "mouse keypress\n");
		}
		close();
		convf(script_name.c_str());
		system((gnuplot_exe_path + " " + script_name).c_str());

		capture_image = false;
	}

	void newplot()
	{
		if (script ) fclose(script);
		close();
		convf(script_name.c_str());
		system((gnuplot_exe_path + " " + script_name).c_str());
		capture_image = false;
	}

	void Heatmap(Matrix<dnn_double>&X, std::vector<std::string>& headers, std::vector<std::string>& rows, char* palette = "rgbformulae 21,22,23", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "#unset autoscale x\n");
		fprintf(script, "#unset autoscale y\n");
		//fprintf(script, "set yrange[0:%d]\n", X.m);
		//fprintf(script, "set xrange[0:%d]\n", X.n);

		FILE* fp = fopen((char*)data_name.c_str(), "w");
		if (fp == NULL)
		{
			return;
		}
		fprintf(fp, "%s,", "MAP");
		for (int j = 0; j < X.n-1; j++)
		{
			fprintf(fp, "%s,", headers[j].c_str());
		}
		fprintf(fp, "%s\n", headers[X.n - 1].c_str());

		for (int i = X.m-1; i >= 0; i--)
		{
			fprintf(fp, "%s,", rows[i].c_str());
			for (int j = 0; j < X.n-1; j++)
			{
				fprintf(fp, "%.3f,", X(i, j));
			}
			fprintf(fp, "%.3f\n", X(i, X.n-1));
		}
		fclose(fp);

		set_palette(palette);
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' matrix rowheaders columnheaders using 1:2:3 with image\n",
			plot, data_name.c_str());
		plot_count++;
	}

	void error_map(Matrix<dnn_double>&X, float point_size = 2, int pointtype = 5, char* palette = "rgbformulae 22, 13, -31", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "#unset autoscale x\n");
		fprintf(script, "#unset autoscale y\n");
		fprintf(script, "set yrange [] reverse\n");
		//fprintf(script, "set yrange[0:%d]\n", X.m);
		//fprintf(script, "set xrange[0:%d]\n", X.n);

		FILE* fp = fopen((char*)data_name.c_str(), "w");
		if (fp == NULL)
		{
			return;
		}

		for (int i = 0; i < X.m; i++)
		{
			for (int j = 0; j < X.n; j++)
			{
				if (X(i, j) == 1)
				{
					fprintf(fp, "%d,%d,1\n", j, i);
				}
				if (X(i, j) == -1)
				{
					fprintf(fp, "%d,%d,-1\n", j, i);
				}
			}
		}
		fclose(fp);

		set_palette(palette);
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' using 1:2:3 t \"error\" with points pointsize %.1f pt %d lc palette\n",
			plot, data_name.c_str(), point_size, pointtype);
		//fprintf(script, "%s '%s' using 1:2:3 t \"error\" with points pointsize %.1f pt %d\n",
		//	plot, data_name.c_str(), point_size, pointtype);

		plot_count++;
	}

	void probability_ellipse(Matrix<dnn_double>& T, int col1, int col2, dnn_double p=0.05)
	{
		Matrix<dnn_double> a = T.Col(col1);
		a = a.appendCol(T.Col(col2));
		//a.print();

		Matrix<dnn_double>& cor = a.Cor();
		cor.print();
		char text[32];
		sprintf(text, "r=%.3f", cor(0, 1));


		int NN = 100;
		Matrix<dnn_double> XYeli(NN + 1, 2);
		{
			Matrix<dnn_double>& x = T.Col(col1);
			x = x.appendCol(T.Col(col2));

			Matrix<dnn_double>& xmean = x.Mean();
			Matrix<dnn_double> xCovMtx = x.Cov(xmean);

			xmean.print("xmean");
			xCovMtx.print("xCovMtx");

			eigenvalues eig;
			eig.set(xCovMtx);
			eig.calc(true);

			Matrix<dnn_double> xy(NN + 1, 2);
			double s = 2.0*M_PI / NN;
			Matrix<dnn_double>& lambda = eig.getRealValue();

			lambda.print("lambda");
			for (int i = 0; i <= NN; i++)
			{
				xy(i, 0) = sqrt(lambda.v[0])*cos(i*s);
				xy(i, 1) = sqrt(lambda.v[1])*sin(i*s);
			}
			std::vector<Matrix<dnn_double>>&vecs0 = eig.getRightVector(0);
			std::vector<Matrix<dnn_double>>&vecs1 = eig.getRightVector(1);

			vecs0[0].print("Re vecs0[0]");
			vecs0[1].print("Im vecs0[1]");
			vecs1[0].print("Re vecs1[0]");
			vecs1[1].print("Im vecs1[1]");
			Matrix<dnn_double> stdElc(NN + 1, 2);

			for (int i = 0; i <= NN; i++)
			{
				stdElc(i, 0) = vecs0[0](0, 0)*xy(i, 0) + vecs1[0](0, 0)*xy(i, 1);
				stdElc(i, 1) = vecs0[0](1, 0)*xy(i, 0) + vecs1[0](1, 0)*xy(i, 1);
			}
			stdElc.print("stdElc");

			int N = x.m;
			int c1 = 2 * (N - 1) / N * (N + 1) / (N - 2);

			F_distribution f_distribution(2, N - 2);
			double F = sqrt(c1*f_distribution.p_value(p));

			printf("F:%f\n", F);
			for (int i = 0; i <= NN; i++)
			{
				XYeli(i, 0) = F*stdElc(i, 0) + xmean.v[0];
				XYeli(i, 1) = F*stdElc(i, 1) + xmean.v[1];
			}
		}
		char s[32];
		sprintf(s, "%.2f", 100.0*(1.0 - p));
		plot_lines2d(XYeli, std::string(s)+std::string("% confidence ellipse / probability ellipse"));
	}
};

#endif
