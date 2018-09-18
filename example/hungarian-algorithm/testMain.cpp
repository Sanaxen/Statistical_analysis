//#define _CRT_SECURE_NO_WARNINGS
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#define _cublas_Init_def extern
#define _cublas_Init_def
#include <iostream>
#include "../../include/hungarian-algorithm/Hungarian.h"

/*
0,0     1,2     2,3     3,4
cost: 31
*/

template<class T>
class VectorIndex
{
public:
	T dat;
	T abs_dat;
	int id;
	bool zero_changed;
	
	VectorIndex()
	{
		zero_changed = false;
	}
	bool operator<(const VectorIndex& right) const {
		return abs_dat < right.abs_dat;
	}
};

int main(void)
{
    // please use "-std=c++11" for this initialization of vector.
	//double x[] = {  10, 19, 8, 15, 0,
	//				10, 18, 7, 17, 0,
	//				13, 16, 9, 14, 0, 
	//				12, 19, 8, 18, 0 };
	//Matrix<dnn_double> costMatrix(x, 4, 5);

	/*
	[[  1.32550801e+02   1.79197108e+00   8.63452588e+00   1.49997664e+03]
	[  9.07141471e+00   3.79031870e+02   8.58641502e+01   5.88733191e+02]
	[  4.48629928e+00   9.41502609e+01   1.33061378e+00   9.28545664e+00]
	[  3.09862471e+00   8.49656359e+00   7.02029863e+02   2.14242378e+03]]
	>>> from copy import deepcopy
	>>> from munkres import Munkres
	>>> m = Munkres()
	>>> ixs = np.vstack(m.compute(deepcopy(W_ica_)))
	>>> print(ixs)
	[[0 2]
	[1 0]
	[2 3]
	[3 1]]
	*/
	dnn_double xx[] =
	{
		7.54427732e-03, -5.58044720e-01,   1.15814118e-01, -6.66677048e-04,
		1.10236389e-01,   2.63830057e-03,   1.16463040e-02, -1.69856229e-03,
		-2.22900867e-01, -1.06213195e-02, -7.51532878e-01,   1.07695296e-01,
		3.22723819e-01, -1.17694641e-01, -1.42444083e-03,   4.66761062e-04
	};
	Matrix<dnn_double> W_ica(xx, 4, 4);
	W_ica.print_e();

	Matrix<dnn_double> W_ica_ = Abs(W_ica).Reciprocal();
	W_ica_.print_e();
	/*
	#[[1.32550801e+02 1.79197108e+00 8.63452588e+00 1.49997664e+03]
	# [9.07141471e+00 3.79031870e+02 8.58641502e+01 5.88733191e+02]
	# [4.48629928e+00 9.41502609e+01 1.33061378e+00 9.28545664e+00]
	# [3.09862471e+00 8.49656359e+00 7.02029863e+02 2.14242378e+03]]
	*/

	HungarianAlgorithm HungAlgo;
	vector<int> replacement;

	double cost = HungAlgo.Solve(W_ica_, replacement);

	for ( int x = 0; x < W_ica_.m; x++)
		std::cout << x << "," << replacement[x] << "\t";

	std::cout << "\ncost: " << cost << std::endl;


	Matrix<dnn_double>& ixs = toMatrix(replacement);
	ixs.print();
	Substitution(replacement).print();
	/*
	#[[0 2]
	# [1 0]
	# [2 3]
	# [3 1]]
	*/

	Matrix<dnn_double>& W_ica_perm = (Substitution(replacement).inv()*W_ica);
	W_ica_perm.print_e();
	/*
	#[[ 1.10236389e-01  2.63830057e-03  1.16463040e-02 -1.69856229e-03]
	# [ 3.22723819e-01 -1.17694641e-01 -1.42444083e-03  4.66761062e-04]
	# [ 7.54427732e-03 -5.58044720e-01  1.15814118e-01 -6.66677048e-04]
	# [-2.22900867e-01 -1.06213195e-02 -7.51532878e-01  1.07695296e-01]]
	*/

	Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
	Matrix<dnn_double> D2(diag_vector(D));
	(D2.Reciprocal()).print_e();

	Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal()));

	W_ica_perm_D.print_e();
	/*
	#[[ 1.00000000e+00  2.39331186e-02  1.05648454e-01 -1.54083630e-02]
	# [-2.74204345e+00  1.00000000e+00  1.21028521e-02 -3.96586504e-03]
	# [ 6.51412578e-02 -4.81845158e+00  1.00000000e+00 -5.75644023e-03]
	# [-2.06973634e+00 -9.86238015e-02 -6.97832595e+00  1.00000000e+00]]
	*/

	Matrix<dnn_double>& b_est = Matrix<dnn_double>().unit(W_ica_perm_D.m, W_ica_perm_D.n) - W_ica_perm_D;
	b_est.print_e();
	/*
	#[[ 0.00000000e+00 -2.39331186e-02 -1.05648454e-01  1.54083630e-02]
	# [ 2.74204345e+00  0.00000000e+00 -1.21028521e-02  3.96586504e-03]
	# [-6.51412578e-02  4.81845158e+00  0.00000000e+00  5.75644023e-03]
	# [ 2.06973634e+00  9.86238015e-02  6.97832595e+00  0.00000000e+00]]
	*/

	std::vector<std::vector<int>> replacement_list;
	std::vector<int> v(W_ica_perm_D.m);
	std::iota(v.begin(), v.end(), 0);       // v に 0, 1, 2, ... N-1 を設定
	do {
		std::vector<int> replacement_case;

		for (auto x : v) replacement_case.push_back(x);
		replacement_list.push_back(replacement_case);
		for (auto x : v) cout << x << " "; cout << "\n";    // v の要素を表示
	} while (next_permutation(v.begin(), v.end()));     // 次の順列を生成

	const int n = W_ica_perm_D.m;
	const int N = int(n * (n + 1) / 2) - 1;

	//for (int i = 0; i < b_est.m*b_est.n; i++)
	//{
	//	if (fabs(b_est.v[i]) < 1.0e-8) b_est.v[i] = 0.0;
	//}

	Matrix<dnn_double> b_est_tmp = b_est;

	dnn_double min_val = 0.0;
	std::vector<VectorIndex<dnn_double>> tmp;
	for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
	{
		VectorIndex<dnn_double> d;
		d.dat = b_est_tmp.v[i];
		d.abs_dat = fabs(b_est_tmp.v[i]);
		d.id = i;
		tmp.push_back(d);
	}
	//絶対値が小さい順にソート
	std::sort(tmp.begin(), tmp.end());

	int N_ = N;
	bool tri_ng = false;
	do
	{
		//b_est_tmp 成分の絶対値が小さい順に n(n+1)/2 個を 0 と置き換える
		int nn = 0;
		for (int i = 0; i < tmp.size(); i++)
		{
			if (nn >= N_) break;
			if (tmp[i].zero_changed) continue;
			printf("[%d]%f ", i, tmp[i].dat);
			tmp[i].dat = 0.0;
			tmp[i].abs_dat = 0.0;
			tmp[i].zero_changed = true;
			nn++;
		}
		printf("\n");
		N_ = 1;	//次は次に絶対値が小さい成分を 0 と置いて再び確認

		//b_est_tmp 成分に戻す
		for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
		{
			b_est_tmp.v[tmp[i].id] = tmp[i].dat;
		}
		b_est_tmp.print_e();

		tri_ng = false;
		//行を並び替えて下三角行列にできるか調べる。
		for (int k = 0; k < replacement_list.size(); k++)
		{
			//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v の要素を表示

			Matrix<dnn_double> tmp = Substitution(replacement_list[k])*b_est_tmp;

			for (int i = 0; i < tmp.m; i++)
			{
				for (int j = i; j < tmp.n; j++)
				{
					if (tmp(i, j) != 0.0)
					{
						tri_ng = true;
						break;
					}
				}
				if (tri_ng) break;
			}
			//if (!tri_ng)
			//{
			//	b_est = tmp;
			//	for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v の要素を表示
			//	break;
			//}
			if (!tri_ng)
			{
				b_est = Substitution(replacement_list[k])*b_est;
				for (int i = 0; i < b_est.m; i++)
				{
					for (int j = i; j < b_est.n; j++)
					{
						b_est(i, j) = 0.0;
					}
				}
				replacement = replacement_list[k];
				//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v の要素を表示
				break;
			}
		}
	} while (tri_ng);

	b_est.print();

	printf("reference\n");
	printf(
		"/*\n"
		"#[[ 0.          0.          0.          0.        ]\n"
		"# [ 2.74204345  0.          0.          0.        ]\n"
		"# [-0.06514126  4.81845158  0.          0.        ]\n"
		"# [ 2.06973634  0.0986238   6.97832595  0.        ]]\n"
		"*/\n");

	return 0;
}
