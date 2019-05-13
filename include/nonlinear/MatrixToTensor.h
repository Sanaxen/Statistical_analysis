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
	tiny_dnn::vec_t tmp;
	for (int i = 0; i < lable - 1; i++)
	{
		tmp.push_back(0);
	}
	tmp.push_back(1);
	for (int i = lable; i < class_max_num; i++)
	{
		tmp.push_back(0);
	}
	return tmp;
}

#endif
