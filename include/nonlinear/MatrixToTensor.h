#ifndef _MatrixToTensor_H

#define _MatrixToTensor_H

inline void MatrixToTensor(Matrix<dnn_double>& X, tiny_dnn::tensor_t& T, int read_max = -1)
{
	size_t rd_max = read_max < 0 ? X.m : std::min(read_max, X.m);
	for (int i = 0; i < rd_max; i++)
	{
		tiny_dnn::vec_t x;
		for (int j = 0; j < X.n; j++)
		{
			x.push_back(X(i, j));
		}
		T.push_back(x);
	}
}

inline void TensorToMatrix(tiny_dnn::tensor_t& T, Matrix<dnn_double>& X)
{
	X = Matrix<dnn_double>(T.size(), T[0].size());
	for (int i = 0; i < T.size(); i++)
	{
		for (int j = 0; j < T[i].size(); j++)
		{
			X(i, j)= T[i][j];
		}
	}
}

inline tiny_dnn::vec_t label2tensor(size_t lable, int class_max_num)
{
	tiny_dnn::vec_t tmp(class_max_num, 0);
	if (lable < 0 || lable >= class_max_num)
	{
		return tmp;
	}
	tmp[lable] = 1;
	//printf("%d %d:", class_max_num, tmp.size());
	//for (int i = 0; i < class_max_num; i++)
	//{
	//	printf(" %f", tmp[i]);
	//}
	//printf("\n");
	return tmp;
}

tiny_dnn::tensor_t diff_vec(tiny_dnn::tensor_t& X, std::vector<int>& idx, int lag = 1)
{
	tiny_dnn::tensor_t diff;
	const bool isidx = idx.size() > 0;

	diff.resize(X.size() - lag);
	for (int i = 0; i < X.size() - lag; i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			if (!isidx || isidx && !idx[k])
			{
				float_t z = X[i + lag][k] - X[i][k];
				diff[i].push_back(z);
			}
			else
			{
				diff[i].push_back(X[i + lag][k]);
			}
		}
	}
	return diff;
}

tiny_dnn::tensor_t diffinv_vec(tiny_dnn::tensor_t& base, tiny_dnn::tensor_t& X, std::vector<int>& idx, int lag = 1, bool logfnc = false)
{
	tiny_dnn::tensor_t diffinv;
	diffinv.resize(X.size());

	const bool isidx = idx.size() > 0;

	for (int i = 0; i < X.size(); i++)
	{
		diffinv[i].resize(X[0].size(), 0.0);
	}
	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			if (!isidx || isidx && !idx[k])
			{
				if (i <= lag - 1)
				{
					if (logfnc)
					{
						diffinv[i][k] = log(base[i][k]);
					}
					else
					{
						diffinv[i][k] = base[i][k];
					}
				}
				else
				{
					diffinv[i][k] = diffinv[i - lag][k] + X[i - lag][k];
				}
			}
			else
			{
				if (i <= lag - 1)
				{
					diffinv[i][k] = base[i][k];
				}
				else
				{
					diffinv[i][k] = X[i - lag][k];
				}
			}
		}
	}
	return diffinv;
}

tiny_dnn::tensor_t log(tiny_dnn::tensor_t& X, std::vector<int>& idx)
{
	tiny_dnn::tensor_t r = X;

	const bool isidx = idx.size() > 0;

#pragma omp parallel for
	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			if (X[i][k] < 0)
			{
				printf("ERROR:-------- log ( 0 < x ) --------\n");
			}
			if (!isidx || isidx && !idx[k]) r[i][k] = log(X[i][k]);
			else r[i][k] = X[i][k];
		}
	}
	return r;
}
tiny_dnn::tensor_t exp(tiny_dnn::tensor_t& X, std::vector<int>& idx)
{
	tiny_dnn::tensor_t r = X;
	const bool isidx = idx.size() > 0;

#pragma omp parallel for
	for (int i = 0; i < X.size(); i++)
	{
		for (int k = 0; k < X[0].size(); k++)
		{
			if (!isidx || isidx && !idx[k]) r[i][k] = exp(X[i][k]);
			else r[i][k] = X[i][k];
		}
	}
	return r;
}
#endif
