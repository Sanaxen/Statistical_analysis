#ifndef _CSVREADER_H
#define _CSVREADER_H

#include "../../third_party/CSVparser/CSVparser_single.hpp"

class CSVReader
{
	csv::Parser* csvfile_;
public:
	CSVReader(char* filename, char separator = ',', bool use_header = true)
	{
		try
		{
			csvfile_ = new csv::Parser(std::string(filename), csv::DataType::eFILE, separator, use_header);
		}
		catch (csv::Error& e)
		{
			printf("CSVReader error:%s\n", e.what());
		}
	}
	CSVReader(std::string filename, char separator = ',', bool use_header = true)
	{
		try
		{
			csvfile_ = new csv::Parser(filename, csv::DataType::eFILE, separator, use_header);
		}
		catch (csv::Error& e)
		{
			printf("CSVReader error:%s\n", e.what());
		}
	}
	~CSVReader()
	{
		if (csvfile_ ) delete csvfile_;
		csvfile_ = NULL;
	}

	void clear()
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

	std::vector<int> empty_cell;
	std::vector<int> nan_cell;
	Matrix<dnn_double> toMat(int rowMax=-1)
	{
		std::vector<int> empty;
		std::vector<int> nan;

		Matrix<dnn_double>& M =  toMat(rowMax, empty, nan);
		empty_cell = empty;
		nan_cell = nan;
		return M;
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

	std::vector<std::string> ItemCol(int col)
	{
		csv::Parser& csvfile = *csvfile_;

		int m = csvfile.rowCount();
		int n = csvfile.columnCount();

		std::vector<std::string> items;
		for (int i = 0; i < m; i++)
		{
			items.push_back(csvfile[i][col]);
		}
		return items;
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

				double v = 0.0;
				if (*value == '+' || *value == '-' || *value == '.' || isdigit(*value))
				{
					bool no_number = false;
					int dot = 0;
					char* p = (char*)cell.c_str();
					while (isspace(*p)) p++;
					if (*p == '+' || *p == '-') p++;
					if (*p == '.')
					{
						p++;
						dot++;
					}
					if (!isdigit(*p)) no_number = true;
					if (!no_number)
					{
						while (isdigit(*p)) p++;
						if (*p == '.' && dot == 0)
						{
							p++;
							dot++;
						}
						if ( dot == 2) no_number = true;
						if (!no_number)
						{
							if (isdigit(*p))
							{
								while (isdigit(*p)) p++;
							}
							if (*p == 'E' || *p == 'e')
							{
								p++;
								if (*p == '+' || *p == '-')
								{
									p++;
								}
							}
							if (isdigit(*p))
							{
								while (isdigit(*p)) p++;
							}
						}
						while (isspace(*p)) p++;
						if (*p != '\0') no_number = true;
					}
					if (!no_number)
					{
						sscanf(value, "%lf", &v);
					}
					else
					{
						nan.push_back(i*n + j);
						v = 0;
					}
				}
				else
				{
					nan.push_back(i*n + j);
					v = 0;
				}
				mat(i, j) = v;
			}
		}
		printf("empty cell:%d\n", empty.size());
		printf("nan cell:%d\n", nan.size());
		return mat;
	}
};

#endif
