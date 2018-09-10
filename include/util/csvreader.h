#ifndef _CSVREADER_H
#define _CSVREADER_H

#include "../../third_party/CSVparser/CSVparser_single.hpp"

class CSVReader
{
	csv::Parser* csvfile_;
public:
	CSVReader(char* filename, char separator = ',', bool use_header = true)
	{
		csvfile_ = new csv::Parser(std::string(filename), csv::DataType::eFILE, separator, use_header);
	}
	~CSVReader()
	{
		delete csvfile_;
		csvfile_ = NULL;
	}

	std::vector<std::string> getHeader()
	{
		return csvfile_->getHeader();
	}
	std::string getHeader(const int index)
	{
		return csvfile_->getHeaderElement(index);
	}

	Matrix<dnn_double> toMat(int rowMax=-1)
	{
		std::vector<int> empty;
		std::vector<int> nan;

		return toMat(rowMax, empty, nan);
	}
	Matrix<dnn_double> toMat_removeEmptyRow()
	{
		std::vector<int> empty;
		std::vector<int> nan;

		Matrix<dnn_double> mat = toMat(-1, empty, nan);
		if (empty.size() == 0 || mat.m == 1)
		{
			return mat;
		}
		int m = mat.m;

		do
		{
			m--;
			empty.clear();
			nan.clear();
			mat = toMat(m, empty, nan);
		} while (empty.size() && m >= 1);

		return mat;
	}

	Matrix<dnn_double> toMat(
		int rowMax,
		std::vector<int>& empty, std::vector<int>& nan)
	{
		csv::Parser& csvfile = *csvfile_;

		int m = csvfile.rowCount();
		int n = csvfile.columnCount();

		printf("m,n:%d, %d\n", m, n);
		fflush(stdout);
		if (rowMax > 0)
		{
			m = rowMax;
		}

		Matrix<dnn_double> mat(m, n);

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				csv::Row& row = csvfile.getRow(i);

				if (csvfile[i][j] == "")
				{
					empty.push_back(i*n + j);
					mat(i, j) = 0.0;
					continue;
				}

				std::string cell = row[j];
				const char* value = cell.c_str();

				if (*value == '+' || *value == '-' || *value == '.' || isdigit(*value))
				{
				}
				else
				{
					nan.push_back(i*n + j);
				}
				double v;
				sscanf(value, "%lf", &v);
				mat(i, j) = v;
			}
		}
		printf("empty cell:%d\n", empty.size());
		printf("nan cell:%d\n", nan.size());
		return mat;
	}
};

#endif
