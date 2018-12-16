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


#endif
