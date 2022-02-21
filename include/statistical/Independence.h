#ifndef _Independence_H

#define _Independence_H
#include <algorithm>

Matrix<dnn_double> squareform_pdist(Matrix<dnn_double>& x)
{
    if (x.n > 1)
    {
        printf("n > 1 error.\n");
        return Matrix<dnn_double>().unit(x.m, x.m);
    }
    Matrix<dnn_double>mat = Matrix<dnn_double>().zeros(x.m, x.m);

    for (int j = 0; j < x.m; j++)
    {
#pragma omp parallel for
        for (int jj = j + 1; jj < x.m; jj++)
        {
            mat(j, jj) = (x(j, 0) - x(jj, 0)) * (x(j, 0) - x(jj, 0));
            mat(jj, j) = mat(j, jj);
        }
    }

    return mat;
}

Matrix<dnn_double> Gram_Matrix(Matrix<dnn_double>& x)
{
    Matrix<dnn_double>& pairwise_dist = squareform_pdist(x);
    auto& t = Pow(pairwise_dist, 2.0);
    return Exp(-0.5 * t);
}

class HSIC
{
public:

    double value(Matrix<dnn_double>& x, Matrix<dnn_double>& y, int nrperm = 0)
    {
        std::mt19937 engine(1234);

        Matrix<dnn_double> K_x;
        Matrix<dnn_double> K_y;

        int m;

        if (nrperm > x.m) nrperm = 0;
        if (nrperm > 0)
        {

            vector<size_t> permutation(x.m, 0);
            for (size_t i = 0; i < x.m; i++)
                permutation[i] = i;

            std::shuffle(permutation.begin(), permutation.end(), engine);

            Matrix<dnn_double> xx(nrperm, 1);
            Matrix<dnn_double> yy(nrperm, 1);

            for (int i = 0; i < nrperm; i++)
            {
                xx(i, 0) = x(permutation[i], 0);
                yy(i, 0) = y(permutation[i], 0);
            }
            K_x = Gram_Matrix(xx);
            K_y = Gram_Matrix(yy);
            m = nrperm;
        }
        else
        {
            K_x = Gram_Matrix(x);
            K_y = Gram_Matrix(y);
            m = x.m;
        }

        const Matrix<dnn_double>& H = Matrix<dnn_double>().unit(m, m) - 1.0 / (double)m;
        const Matrix<dnn_double>& K_yH = K_y * H;
        const Matrix<dnn_double>& HK_yH = H * K_yH;
        Matrix<dnn_double>& K_xHK_yH = K_x * HK_yH;

        double hsic = pow(m - 1, -2) * K_xHK_yH.Tr();

        return hsic;
    }

    double value_(Matrix<dnn_double>& x, Matrix<dnn_double>& y, int nrperm = 0, int sample = 3)
    {
        std::vector<double> hsic(sample, 0);
#pragma omp parallel for
        for (int i = 0; i < sample; i++)
        {
            hsic[i] = value(x, y, nrperm) / (double)sample;
        }

        double h = 0.0;
        for (int i = 0; i < sample; i++)
        {
            h += hsic[i];
        }
        return h;
    }

};

//
class HSIC_ref
{
public:
    double hsic;         // empirical HSIC estimate
    double p_value;      // p-value for independence test (small p-value means independence is unlikely)

private:
    void norm(Matrix<dnn_double>& mat, std::vector<double>& result)
    {
        size_t dims = mat.n;
        size_t N = mat.m;

        result.resize(N * N);
        if (dims == 1) {  // optimized version of dims > 1 case
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < i; j++) {
                    double x = (mat.v[i] - mat.v[j]) * (mat.v[i] - mat.v[j]);
                    result[i * N + j] = x;
                    result[j * N + i] = x;
                }
                result[i * N + i] = 0.0;
            }
        }
        else {
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < i; j++) {
                    double x = 0.0;
                    for (size_t k = 0; k < N * dims; k += N) {
                        double x_k = mat.v[i + k] - mat.v[j + k];
                        x += x_k * x_k;
                    }
                    result[i * N + j] = x;
                    result[j * N + i] = x;
                }
                result[i * N + i] = 0.0;
            }
        }
    }

    double sigma(const std::vector<double>& norm, size_t N) {
        double med = 0.0;
        std::vector<double> x;
        x.reserve(N * (N - 1) / 2);
        for (size_t i = 0; i < N; i++)
            for (size_t j = i + 1; j < N; j++)
                if (norm[i * N + j] != 0.0)
                    x.push_back(norm[i * N + j]);
        nth_element(x.begin(), x.begin() + x.size() / 2, x.end());
        double x1 = *(x.begin() + x.size() / 2);
        if (x.size() % 2) {
            med = x1;
        }
        else {
            nth_element(x.begin(), x.begin() + x.size() / 2 - 1, x.end());
            double x2 = *(x.begin() + x.size() / 2 - 1);
            med = (x1 + x2) / 2.0;
        }

        return sqrt(0.5 * med);
    }


    void calcHSIC(Matrix<dnn_double>& x, Matrix<dnn_double>& y, double sigma_x = 0.0, double sigma_y = 0.0, int nrperm = 1)
    {
        // build matrices x_norm, y_norm (containing the squared l2-distances between data points)

        size_t dims_x = x.n;
        size_t dims_y = y.n;
        size_t N = x.m;

        std::vector<double> x_norm;
        std::vector<double> y_norm;

        norm(x, x_norm);
        norm(y, y_norm);

        // if necessary, choose kernel bandwidth using heuristic
        if (sigma_x == 0.0)
            sigma_x = sigma(x_norm, N);
        if (sigma_y == 0.0)
            sigma_y = sigma(y_norm, N);

        // build RBF kernel matrix Kx
        double c = -0.5 / (sigma_x * sigma_x);
        std::vector<double> Kx;
        Kx.reserve(N * N);
        double Kx_sum = 0.0;
        std::vector<double> Kx_sums(N, 0.0);
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                double x = exp(x_norm[i * N + j] * c);
                Kx[i * N + j] = x;
                Kx_sums[i] += x;
            }
            Kx_sum += Kx_sums[i];
        }

        // build RBF kernel matrix Ky
        c = -0.5 / (sigma_y * sigma_y);
        std::vector<double> Ky;
        Ky.reserve(N * N);
        double Ky_sum = 0.0;
        std::vector<double> Ky_sums(N, 0.0);
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                double y = exp(y_norm[i * N + j] * c);
                Ky[i * N + j] = y;
                Ky_sums[i] += y;
            }
            Ky_sum += Ky_sums[i];
        }

        if (nrperm < 0) { // unbiased HSIC estimate with permutation test (see [2])
            double KxKysum = 0.0;
            double tr_KxKy = 0.0;
            for (size_t i = 0; i < N; i++) {
                KxKysum += Kx_sums[i] * Ky_sums[i];
                for (size_t j = 0; j < N; j++)
                    tr_KxKy += Kx[i * N + j] * Ky[i * N + j];
            }
            hsic = (tr_KxKy + (Kx_sum * Ky_sum) / ((N - 1) * (N - 2)) - 2.0 / (N - 2) * KxKysum) / (N * (N - 3));

            if (abs(nrperm) > 0)
            {
                std::mt19937 engine(1234);

                // construct random permutation
                vector<size_t> permutation(N, 0);
                for (size_t i = 0; i < N; i++)
                    permutation[i] = i;

                size_t count = 0;  // counts how often permuted HSIC is larger than HSIC
                for (size_t perm = 0; perm < abs(nrperm); perm++) {
                    std::shuffle(permutation.begin(), permutation.end(), engine);

                    double ptr_KxKy = 0.0;
                    double pKxKysum = 0.0;
                    for (size_t i = 0; i < N; i++) {
                        size_t pi = permutation[i];

                        pKxKysum += Kx_sums[i] * Ky_sums[pi];
                        for (size_t j = 0; j < N; j++)
                            ptr_KxKy += Kx[i * N + j] * Ky[pi * N + permutation[j]];
                    }
                    double phsic = (ptr_KxKy + (Kx_sum * Ky_sum) / ((N - 1) * (N - 2)) - 2.0 / (N - 2) * pKxKysum) / (N * (N - 3));

                    if (phsic > hsic)
                        count++;
                }

                p_value = (count + 1.0) / (abs(nrperm) + 2.0);  // Incorporate a little prior to prevent p_values of 0 or 1
            }
        }
        else { // biased HSIC estimate with permutation test (see [1])
         // calculate HSIC and precalculate KxH, HKy
            double tr_KxHKyH = 0.0;
            vector<double> KxH, HKy;
            KxH.reserve(N * N);
            HKy.reserve(N * N);
            for (size_t i = 0; i < N; i++)
                for (size_t j = 0; j < N; j++) {
                    double KxH_ij = Kx[i * N + j] - Kx_sums[i] / N;
                    double KyH_ji = Ky[i * N + j] - Ky_sums[j] / N;
                    KxH.push_back(KxH_ij);
                    HKy.push_back(KyH_ji);
                    tr_KxHKyH += KxH_ij * KyH_ji;
                }
            hsic = tr_KxHKyH / (N * N);

            if (nrperm > 0)
            {
                std::mt19937 engine(1234);

                // construct random permutation
                vector<size_t> permutation(N, 0);
                for (size_t i = 0; i < N; i++)
                    permutation[i] = i;

                size_t count = 0;  // counts how often permuted HSIC is larger than HSIC
                for (size_t perm = 0; perm < nrperm; perm++) {
                    std::shuffle(permutation.begin(), permutation.end(), engine);

                    double ptr_KxHKyH = 0.0;
                    for (size_t i = 0; i < N; i++) {
                        size_t pi = permutation[i];
                        for (size_t j = 0; j < N; j++)
                            ptr_KxHKyH += KxH[i * N + j] * HKy[pi * N + permutation[j]];
                    }

                    if (ptr_KxHKyH > tr_KxHKyH)
                        count++;
                }

                p_value = (count + 1.0) / (nrperm + 2.0);  // Incorporate a little prior to prevent p_values of 0 or 1
            }
        }
    }

public:
    double HSIC_pvalue(Matrix<dnn_double>& x, Matrix<dnn_double>& y)
    {
        calcHSIC(x, y, 0.0, 0.0, 15);
        return p_value;
    }
    double HSIC_value(Matrix<dnn_double>& x, Matrix<dnn_double>& y)
    {
        calcHSIC(x, y, 0.0, 0.0, 0);
        return hsic;
    }
}; 

/* Mutual information*/
//https://qiita.com/hyt-sasaki/items/ffaab049e46f800f7cbf
class MutualInformation
{
    void gridtabel(Matrix<dnn_double>& M1, Matrix<dnn_double>& M2)
    {
        double max1, min1;
        double max2, min2;

#pragma omp parallel
        {
#pragma omp sections
            {
                M1.MaxMin(max1, min1);
            }
#pragma omp sections
            {
                M2.MaxMin(max2, min2);
            }
        }

#if 0
        double dx = (max1 - min1) / grid;
        double dy = (max2 - min2) / grid;
#else
        max1 = (max1 > max2) ? max1 : max2;
        min1 = (min1 < min2) ? min1 : min2;
        max2 = max1;
        min2 = min1;
        double dx = (max1 - min1) / grid;
        double dy = (max2 - min2) / grid;
#endif

        //printf("dx:%f dy:%f\n", dx, dy);
        std::vector<dnn_double> table1(grid, 0);
        std::vector<dnn_double> table2 = table1;
        Matrix<dnn_double>& table12 = Matrix<dnn_double>().zeros(grid, grid);

        const int thread_num = omp_get_max_threads() + 1;
        std::vector < std::vector<dnn_double>> tmp_table1(thread_num, std::vector<dnn_double>(grid, 0));
        std::vector < std::vector<dnn_double>> tmp_table2 = tmp_table1;

        const int sz1 = M1.m * M1.n;
#pragma omp parallel for
        for (int k = 0; k < sz1; k++)
        {
            const double m1_k = M1.v[k];
            const double m2_k = M2.v[k];
            const int thread_id = omp_get_thread_num();
            for (int i = 0; i < grid; i++)
            {
                double c = 0;
                if (i == grid - 1) c = 0.000001;
                if (m1_k >= min1 + dx * i && m1_k < min1 + dx * (i + 1))
                {
                    tmp_table1[thread_id][i] += 1;
                }
            }

            for (int i = 0; i < grid; i++)
            {
                double c = 0;
                if (i == grid - 1) c = 0.000001;
                if (m2_k >= min2 + dy * i && m2_k < min2 + dy * (i + 1) + c)
                {
                    tmp_table2[thread_id][i] += 1;
                }
            }
            for (int i = 0; i < grid; i++)
            {
                double c = 0;
                if (i == grid - 1) c = 0.000001;
                bool range1 = (m2_k >= min2 + dy * i && m2_k < min2 + dy * (i + 1) + c);

                if (!range1)
                {
                    continue;
                }
                for (int j = 0; j < grid; j++)
                {
                    double d = 0;
                    if (j == grid - 1) d = 0.000001;

                    bool range2 = (m1_k >= min1 + dx * j && m1_k < min1 + dx * (j + 1) + d);

                    if (range2)
                    {
#pragma omp critical
                        {
                            table12(i, j) += 1;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < grid; i++)
        {
            for (int j = 0; j < thread_num; j++)
            {
                table1[i] += tmp_table1[j][i];
                table2[i] += tmp_table2[j][i];
            }
        }
        //for (int i = 0; i < table1.size(); i++)
        //{
        //	printf("%d, ", (int)table1[i]);
        //}
        //printf("\n");
        //printf("dx:%f\n", dx);

        probability1.resize(grid, 0);
        probability2.resize(grid, 0);
        probability12 = probability12.zeros(grid, grid);
        probability1_2 = probability12;

        double s1 = 0;
        double s2 = 0;
        for (int i = 0; i < grid; i++)
        {
            probability1[i] = table1[i];
            probability2[i] = table2[i];
            s1 += probability1[i];
            s2 += probability2[i];
        }
        double s12 = 0;
        for (int i = 0; i < grid; i++)
        {
            for (int j = 0; j < grid; j++)
            {
                probability12(i, j) = table12(i, j);
                s12 += probability12(i, j);
            }
        }

        if (s1 == 0.0) s1 = 1;
        if (s2 == 0.0) s2 = 1;
        if (s12 == 0.0) s12 = 1;
        for (int i = 0; i < grid; i++)
        {
            probability1[i] /= s1;
            probability2[i] /= s2;
        }
        probability12 = probability12 / s12;

        s12 = 0;
        for (int i = 0; i < grid; i++)
        {
            for (int j = 0; j < grid; j++)
            {
                probability1_2(i, j) = probability1[i] * probability2[j];
                s12 += probability1_2(i, j);
            }
        }
        //probability1_2 = probability1_2 / s12;

        //dump
        //Matrix<dnn_double> tmp1(probability1);
        //Matrix<dnn_double> tmp2(probability2);

        //tmp1.print_csv("p1.csv");
        //tmp2.print_csv("p2.csv");
        //probability12.print_csv("p12.csv");		//p(x,y)
        //probability1_2.print_csv("p1_2.csv");	//p(x)*p(y)
    }

public:
    int grid;
    std::vector<dnn_double> probability1;	//p(x)
    std::vector<dnn_double> probability2;	//p(y)
    Matrix<dnn_double> probability12;		//p(x,y)
    Matrix<dnn_double> probability1_2;		//p(x)*p(y)


    MutualInformation(Matrix<dnn_double>& Col1, Matrix<dnn_double>& Col2, int grid_ = 30)
    {
        grid = grid_;
        //ダイバジェンスの公式
        //grid = (int)(log2((double)Col1.m) + 1);
        //grid = (int)(sqrt((double)Col1.m) + 1);

        gridtabel(Col1, Col2);
    }

    double Information()
    {
        double I = 0.0;

        Matrix<dnn_double>& zz = Matrix<dnn_double>().zeros(grid, grid);

#pragma omp parallel for
        for (int i = 0; i < grid; i++)
        {
            for (int j = 0; j < grid; j++)
            {
                if (probability1_2(i, j) < 1.0e-32 || probability12(i, j) < 1.0e-32)
                {
                    //continue;
                }
                else
                {
                    double z = probability12(i, j) / probability1_2(i, j);
                    if (z > 0)
                    {
                        zz(i, j) = probability12(i, j) * log(z);
                    }
                }
            }
        }
        I = zz.Sum();
        return I;
    }
};

// MIC ? Maximal Information Coefficient
// https://github.com/dspinellis/OpenMIC
///


inline double independ_test(const Matrix<double>& x, const Matrix<double>& y)
{
    auto& xx = Matrix<double>(x.v, x.m * x.n, 1);
    auto& yy = Matrix<double>(y.v, x.m * x.n, 1);

    return fabs(xx.Cor(Tanh(yy))) + fabs(yy.Cor(Tanh(xx)));
}


inline Matrix<dnn_double> _normalize(Matrix<dnn_double>& x)
{
    auto& x_mean = x.Mean();

    return (x - x_mean.v[0]) / (x.Std(x_mean).v[0] + 1.0e-10);
}

//Fast and Robust Fixed-Point Algorithms for Independent Component Analysis
//New approximations of differential entropy for independent co ¨ mponent analysis and projection pursuit
inline double _entropy(Matrix<dnn_double>& u)
{
    auto Hv = (1 + log(2 * M_PI)) / 2.0;
    const double k1 = 79.047;
    const double k2 = 7.4129;		//	36/(8*sqrt(3)-9)
    const double gamma = 0.37457;

    //Hv - k1*(np.mean(np.log(np.cosh(u))) - gamma)**2 
    // - k2 * (np.mean(u * np.exp(-1 * u**2 /2)))**2
    double G2 = Log(Cosh(u)).mean() - gamma;

    double G1 = u.hadamard(Exp(-0.5 * u.hadamard(u))).mean();

    return Hv - k1 * G2 * G2 - k2 * G1 * G1;
}

inline double _MutualInformation(Matrix<dnn_double>& x, Matrix<dnn_double>& y, bool normalize = false)
{
    auto xx = _normalize(x);
    auto yy = _normalize(y);
    auto z = x.appendRow(y);
    auto zz = _normalize(z);

    double mi = _entropy(xx) + _entropy(yy) - _entropy(zz);

    if (normalize)
    {
        return mi / (sqrt(_entropy(xx) * _entropy(yy) + .0e-10));
    }
    return mi;
}

#endif
