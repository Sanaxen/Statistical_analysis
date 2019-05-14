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

#endif
