#include <stdio.h>
#include <stdlib.h>

#define _cublas_Init_def
//#define USE_FLOAT
#include "../../include/Matrix.hpp"
#include "../../include/util/csvreader.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"
#endif

#include <map>

int write_csv(std::string& filename, Matrix<dnn_double>&TT, std::vector<std::string>&header_names_wrk, std::vector<std::string>& category_str, int max_elm)
{
	FILE* fp = fopen(filename.c_str(), "w");
	if (fp == NULL)
	{
		fprintf(stderr, "file open error[write][%s]\n", "bbb3.csv");
		return -1;
	}
	for (int j = 0; j < TT.n - 1; j++)
	{
		fprintf(fp, "%s,", header_names_wrk[j].c_str());
	}
	fprintf(fp, "%s\n", header_names_wrk[TT.n - 1].c_str());

	for (int i = 0; i < TT.m; i++)
	{
		for (int j = 0; j < TT.n; j++)
		{
			if (category_str.size() > 0 && TT.v[TT.n*i + j] >= max_elm)
			{
				int id = TT.v[TT.n*i + j] - max_elm;
				fprintf(fp, "%s", category_str[id].c_str());
			}
			else
			{
				fprintf(fp, "%f", TT.v[TT.n*i + j]);
			}
			if (j < TT.n - 1) fprintf(fp, ",");
			else fprintf(fp, "\n");
		}
	}
	fclose(fp);
}

int main(int argc, char** argv)
{
	std::string csvfile;
	int start_col = 0;
	bool header = false;


	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
			continue;
		}else
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
			continue;
		}
		else {
			std::cerr << "Invalid parameter specified - \"" << argname << "\""
				<< std::endl;
			return -1;
		}
	}

	std::map<std::string, int> category_var;

	CSVReader csv1(csvfile, ',', header);
	std::vector<std::string> header_names;
	Matrix<dnn_double> T = csv1.toMat();

	int max_elm = (int)(T.Max()+0.5);

	header_names.resize(T.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < T.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < T.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}
	if (csv1.nan_cell.size() == 0)
	{
		write_csv(std::string("embedding.csv"), T, header_names, std::vector<std::string>(), 0);
		return 0;
	}

	std::vector<std::string> category_str;
	if (csv1.nan_cell.size())
	{
		int count = 0;
		for (int i = 0; i < csv1.nan_cell.size(); i++)
		{
			int ii = csv1.nan_cell[i] / T.n;
			int jj = csv1.nan_cell[i] % T.n;
			//printf("%s\n", csv1.ItemCol(jj)[ii].c_str());
			if (category_var.count(csv1.ItemCol(jj)[ii]))
			{
				T(ii, jj) = max_elm + category_var[csv1.ItemCol(jj)[ii]];
			}
			else
			{
				category_var[csv1.ItemCol(jj)[ii]] = count;
				category_str.push_back(csv1.ItemCol(jj)[ii]);
				T(ii, jj) = max_elm + count;
				count++;
			}
		}
	}


	for (auto iter = category_var.begin(); iter != category_var.end(); iter = std::next(iter,1))
	{
		std::cout << iter->first << " => " << iter->second << "\n";
	}	
	printf("category:%d\n", category_var.size());

	std::vector<int> category_col;
	for (int j = 0; j < T.n; j++)
	{
		int count = 0;
		for (int i = 0; i < T.m; i++)
		{
			if (category_var.count(csv1.ItemCol(j)[i]))
			{
				count++;
			}
		}
		if (count == T.m)
		{
			category_col.push_back(j);
		}
	}
	printf("category_col:%d\n", category_col.size());

	std::vector<std::string> header_names_wrk = header_names;
	Matrix<dnn_double> TT = T;
	for (int k = 0; k < category_col.size(); k++)
	{
		printf("%d\n", category_col[k]);
		TT = TT.removeCol(category_col[k] - k);
		header_names_wrk.erase(header_names_wrk.begin() + (category_col[k] - k));

		Matrix<dnn_double> one_hot = Matrix<dnn_double>::zeros(T.m, category_var.size());

#pragma omp parallel for
		for (int i = 0; i < T.m; i++)
		{
			if (category_var.count(csv1.ItemCol(category_col[k])[i]))
			{
				int s = category_var[csv1.ItemCol(category_col[k])[i]];
				//printf("[%d,%d]%s %d\n", i, k, csv1.ItemCol(category_col[k])[i].c_str(), s);
				one_hot(i, s) = 1;
			}
		}
		for (int kk = 0; kk < category_str.size(); kk++)
		{
			char buf[32];
			sprintf(buf, "%d", k);
			char* p = new char[category_str[kk].length() + 1];
			strcpy(p, category_str[kk].c_str());
			if (*p == '\"')
			{
				char* q = p + strlen(p);
				q--;
				if (*q == '\"')*q = '\0';
				strcat(p, buf);
				strcat(p, "\"");
			}
			header_names_wrk.push_back(p);
			delete[] p;
		}
		TT = TT.appendCol(one_hot);
	}
	TT.print();
	TT.print_csv("bbb.csv", header_names_wrk);


	std::vector<int> zero_cols;
	for (int k = 0; k < category_str.size()*category_col.size(); k++)
	{
		double t = 0.0;
		for (int i = 0; i < T.m; i++)
		{
			t += TT(i, TT.n - category_str.size()*category_col.size() + k);
		}
		if (t == 0) zero_cols.push_back(TT.n - category_str.size()*category_col.size() + k);
	}

	std::vector<std::string> no_use_category_str;
	printf("zero_cols:%d\n", zero_cols.size());
	for (int k = 0; k < zero_cols.size(); k++)
	{
		printf("%d\n", zero_cols[k]);
		TT = TT.removeCol(zero_cols[k] - k);
		no_use_category_str.push_back(*(header_names_wrk.begin() + (zero_cols[k] - k)));
		header_names_wrk.erase(header_names_wrk.begin() + (zero_cols[k] - k));
	}
	TT.print();
	TT.print_csv("bbb2.csv", header_names_wrk);

	for (int k = 0; k < no_use_category_str.size(); k++)
	{
		printf("%s\n", no_use_category_str[k].c_str());
	}

	write_csv(std::string("embedding.csv"), TT, header_names_wrk, category_str, max_elm);


	csvfile = "embedding.csv";
	CSVReader csv2(csvfile, ',', header);
	Matrix<dnn_double> tmp = csv2.toMat();

	return 0;
}