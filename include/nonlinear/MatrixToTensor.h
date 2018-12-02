#ifndef _MatrixToTensor_H

#define _MatrixToTensor_H

inline void MatrixToTensor(Matrix<dnn_double>& X, tiny_dnn::tensor_t& T)
{
	for (int i = 0; i < X.m; i++)
	{
		tiny_dnn::vec_t x;
		for (int j = 0; j < X.n; j++)
		{
			x.push_back(X(i, j));
		}
		T.push_back(x);
	}
}

inline void MatrixToTensorSeq(Matrix<dnn_double>& X, int seq, tiny_dnn::tensor_t& T)
{
	for (int i = 0; i < X.m-seq; i++)
	{
		tiny_dnn::vec_t x;
		for (int ii = i; ii < i + seq; ii++)
		{
			for (int j = 0; j < X.n; j++)
			{
				x.push_back(X(ii, j));
			}
		}
		T.push_back(x);
	}
}

#endif
