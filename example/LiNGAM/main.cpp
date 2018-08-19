#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"

void read_csv(int n, char* filename, Matrix<dnn_double>& x)
{
	FILE* fp = fopen(filename, "r");
	if (fp == NULL)
	{
		return;
	}

	x = Matrix<dnn_double>(n, 1);
	for (int i = 0; i < n; i++)
	{
		dnn_double s = 0;
		if (i < n - 1)
		{
			int k = fscanf(fp, "%lf,", &s);
			if (k != 1)
			{
				printf("%d error\n", i);
			}
		}
		else
		{
			int k = fscanf(fp, "%lf", &s);
			if (k != 1)
			{
				printf("%d error\n", i);
			}
		}
		x(i, 0) = s;
	}
	fclose(fp);
}

//https://qiita.com/m__k/items/bd87c063a7496897ba7c
//LiNGAM���f���̐�����@�ɂ���
int main()
{
	Matrix<dnn_double> x, y, z, w;

	int len = 1000;
	read_csv(len, "x.csv", x);
	read_csv(len, "y.csv", y);
	read_csv(len, "z.csv", z);
	read_csv(len, "w.csv", w);

	Matrix<dnn_double>xs(len, 4);

	for (int i = 0; i < len; i++)
	{
		xs(i, 0) = x(i, 0);
		xs(i, 1) = y(i, 0);
		xs(i, 2) = z(i, 0);
		xs(i, 3) = w(i, 0);

	}
	xs.print_e();

	Lingam LiNGAM;

	LiNGAM.set(4);
	LiNGAM.fit(xs);

	LiNGAM.B.print_e();


//	ICA ica;
//	ica.set(4);
//	ica.fit(xs, 4000);
//	(ica.A.transpose()).inv().print_e();
//
//	Matrix<dnn_double>& W_ica = (ica.A.transpose()).inv();
//	Matrix<dnn_double>& W_ica_ = Abs(W_ica).Reciprocal();
//
//	HungarianAlgorithm HungAlgo;
//	vector<int> replacement;
//
//	double cost = HungAlgo.Solve(W_ica_, replacement);
//
//	for (unsigned int x = 0; x < W_ica_.m; x++)
//		std::cout << x << "," << replacement[x] << "\t";
//	printf("\n");
//
//	Matrix<dnn_double>& ixs = toMatrix(replacement);
//	ixs.print();
//	Substitution(replacement).print();
//
//	Matrix<dnn_double>& W_ica_perm = (Substitution(replacement).inv()*W_ica);
//	W_ica_perm.print_e();
//
//	Matrix<dnn_double>& D = Matrix<dnn_double>().diag(W_ica_perm);
//	Matrix<dnn_double> D2(diag_vector(D));
//	(D2.Reciprocal()).print_e();
//
//	Matrix<dnn_double>& W_ica_perm_D = W_ica_perm.hadamard(to_vector(D2.Reciprocal()));
//
//	W_ica_perm_D.print_e();
//
//	Matrix<dnn_double>& b_est = Matrix<dnn_double>().unit(W_ica_perm_D.m, W_ica_perm_D.n) - W_ica_perm_D;
//	b_est.print_e();
//
//	std::vector<std::vector<int>> replacement_list;
//	std::vector<int> v(W_ica_perm_D.m);
//	std::iota(v.begin(), v.end(), 0);       // v �� 0, 1, 2, ... N-1 ��ݒ�
//	do {
//		std::vector<int> replacement_case;
//
//		for (auto x : v) replacement_case.push_back(x);
//		replacement_list.push_back(replacement_case);
//		//for (auto x : v) cout << x << " "; cout << "\n";    // v �̗v�f��\��
//	} while (next_permutation(v.begin(), v.end()));     // ���̏���𐶐�
//
//	const int n = W_ica_perm_D.m;
//	const int N = int(n * (n + 1) / 2) - 1;
//
//	//for (int i = 0; i < b_est.m*b_est.n; i++)
//	//{
//	//	if (fabs(b_est.v[i]) < 1.0e-8) b_est.v[i] = 0.0;
//	//}
//
//	Matrix<dnn_double> b_est_tmp = b_est;
//
//	dnn_double min_val = 0.0;
//	std::vector<VectorIndex<dnn_double>> tmp;
//	for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
//	{
//		VectorIndex<dnn_double> d;
//		d.dat = b_est_tmp.v[i];
//		d.abs_dat = fabs(b_est_tmp.v[i]);
//		d.id = i;
//		tmp.push_back(d);
//	}
//	//��Βl�����������Ƀ\�[�g
//	std::sort(tmp.begin(), tmp.end());
//
//	int N_ = N;
//	bool tri_ng = false;
//	do
//	{
//		//b_est_tmp �����̐�Βl������������ n(n+1)/2 �� 0 �ƒu��������
//		int nn = 0;
//		for (int i = 0; i < tmp.size(); i++)
//		{
//			if (nn >= N_) break;
//			if (tmp[i].zero_changed) continue;
//			printf("[%d]%f ", i, tmp[i].dat);
//			tmp[i].dat = 0.0;
//			tmp[i].abs_dat = 0.0;
//			tmp[i].zero_changed = true;
//			nn++;
//		}
//		printf("\n");
//		N_ = 1;	//���͎��ɐ�Βl�������������� 0 �ƒu���čĂъm�F
//
//				//b_est_tmp �����ɖ߂�
//		for (int i = 0; i < b_est_tmp.m*b_est_tmp.n; i++)
//		{
//			b_est_tmp.v[tmp[i].id] = tmp[i].dat;
//		}
//		b_est_tmp.print_e();
//		if (b_est_tmp.isZero(1.0e-8))
//		{
//			break;
//		}
//
//		tri_ng = false;
//		//�s����ёւ��ĉ��O�p�s��ɂł��邩���ׂ�B
//		for (int k = 0; k < replacement_list.size(); k++)
//		{
//			//for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v �̗v�f��\��
//
//			Matrix<dnn_double> tmp = Substitution(replacement_list[k])*b_est_tmp;
//
//#if 10
//			//�Ίp�Ƀ[�������Ȃ��悤�Ȓu���s���T���΂����B
//			for (int i = 0; i < tmp.m*tmp.n; i++)
//			{
//				//�Ίp�Ƀ[�������Ă��܂���
//				if (tmp(i, i) == 0.0)
//				{
//					tri_ng = true;
//					break;
//				}
//			}
//#else
//			//�����s�񂩃`�F�b�N
//			for (int i = 0; i < tmp.m; i++)
//			{
//				for (int j = i; j < tmp.n; j++)
//				{
//					if (tmp(i, j) != 0.0)
//					{
//						tri_ng = true;
//						break;
//					}
//				}
//				if (tri_ng) break;
//			}
//#endif
//
//			if (!tri_ng)
//			{
//				b_est = tmp;
//				for (auto x : v) cout << replacement_list[k][x] << " "; cout << "\n";    // v �̗v�f��\��
//				break;
//			}
//		}
//	} while (tri_ng);
//
//	b_est.print_e();

}