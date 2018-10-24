#ifndef _PLOT_H__
#define _PLOT_H__

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include <string>

class gnuPlot
{
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

public:
	std::string save_image_name = "image.png";
	std::string script_name = "plot.plt";
	std::string data_name = "plot.dat";

	bool save_image = false;
	std::string title = "";
	float linewidth = 2.0;
	float pointsize = 1.0;

	std::string linecolor = "";

	gnuPlot(std::string& gnuplot_exe_path_, const int script_id=-1, bool save_image_=false)
	{
		gnuplot_exe_path = gnuplot_exe_path_;

		id = script_id;
		if (script_id < 0) id = rand();

		save_image = save_image_;

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
		
		if (save_image)
		{
			fprintf(script, "set terminal png\n");
		}
	}
	
	
	void set_palette(char* palette)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set palette %s\n", palette);
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
			every = "every " + std::to_string(every_num);
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

	void multi_scatter(Matrix<dnn_double>&X, std::vector<std::string>& headers, int pointtype = 2, char* palette = "rgbformulae 34,35,36", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		fprintf(script, "set multiplot layout %d,%d\n", X.n, X.n);
		fprintf(script, "set nokey\n");
		fprintf(script, "unset xtics\n");
		fprintf(script, "unset ytics\n");

		
		std::string every = "";

		if (maxpoint > 0)
		{
			int every_num = X.m / maxpoint;
			every = "every " + std::to_string(every_num);
		}


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
					Matrix<dnn_double> &h = Histogram(X.Col(j), 10);
					char histogram_name[256];
					sprintf(histogram_name, "plot_(%d)%d_hist.dat", plot_count, i);
					h.print_csv(histogram_name);

					fprintf(script, "#set style fill solid border  lc rgb \"black\"\n");
					fprintf(script, "plot '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
						histogram_name, (std::string("t ") + x).c_str(), 1/*linewidth*/, linecolor.c_str());

					continue;
				}
				fprintf(script, "set xlabel %s\n", x.c_str());
				fprintf(script, "set ylabel %s\n", y.c_str());
				fprintf(script, "plot '%s' %s using %d:%d %s with points pointsize %.1f pt %d\n",
					data_name.c_str(), every.c_str(), i+1, j+1, title.c_str(), pointsize, pointtype);
				fflush(script);
			}
		}
		fprintf(script, "unset multiplot\n");
		plot_count++;
	}

	void scatter(Matrix<dnn_double>&X, int col1, int col2, std::vector<std::string>& headers, int pointtype=6, char* palette ="rgbformulae 34,35,36", int maxpoint = -1)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");

		Matrix<dnn_double> x = X.Col(col1);
		Matrix<dnn_double> y = X.Col(col2);

		x = x.appendCol(y);
		if (palette)
		{
			x = x.appendCol(y);
			set_palette(palette);
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

		fprintf(script, "set xlabel %s\n", xx.c_str());
		fprintf(script, "set ylabel %s\n", yy.c_str());
		const char* plot = (plot_count) ? "replot" : "plot";
		if (palette)
		{
			fprintf(script, "%s '%s' %s using 1:2:3 %s with points pointsize %.1f pt %d lc palette\n",
				plot, data_name.c_str(), every.c_str(), label.c_str(), pointsize, pointtype);
		}
		else
		{
			fprintf(script, "%s '%s' %s using 1:2 %s with points pointsize %.1f pt %d\n",
				plot, data_name.c_str(), every.c_str(), label.c_str(), pointsize, pointtype);
		}
		linecolor = "";
		plot_count++;
	}

	void plot_histogram(Matrix<dnn_double>&X, char* label_text =NULL)
	{
		script_reopen();
		if (script == NULL) return;
		fprintf(script, "set datafile separator \",\"\n");
		X.print_csv((char*)data_name.c_str());

		fprintf(script, "set style fill solid border  lc rgb \"black\"\n");

		if (linecolor != "")
		{
			linecolor = "lc " + linecolor;
		}
		std::string label = "";
		if (label_text)
		{
			label = "t \"" + std::string(label_text) + "\"";
		}
		const char* plot = (plot_count) ? "replot" : "plot";
		fprintf(script, "%s '%s' using 1:2 %s with boxes linewidth %.1f %s\n",
			plot, data_name.c_str(), label.c_str(), linewidth, linecolor.c_str());

		linecolor = "";
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
	void draw()
	{
		if (script == NULL) return;
		if (save_image)
		{
			fprintf(script, "set out \"%s\"\n", save_image_name.c_str());
			fprintf(script, "replot\n");
		}
		else
		{
		fprintf(script, "pause 3\n");
		}
		close();
		system((gnuplot_exe_path + " " + script_name).c_str());
	}
	
	void newplot()
	{
		if (script ) fclose(script);
		close();
		system((gnuplot_exe_path + " " + script_name).c_str());
	}
};

#endif
