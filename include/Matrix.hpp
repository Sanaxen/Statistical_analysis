#ifndef MATRIX_HPP
#define MATRIX_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#include <iostream>
//#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <assert.h>
#include <numeric>
#define _USE_MATH_DEFINES
#include <math.h>

#define MKL_INT long int

#include "matrix_config.h"
#ifdef USE_FLOAT
#define dnn_double	float
#else
#define dnn_double	double
#endif

#include "matrix/SVD.h"

// -> matrix_config.h
#define USE_LAPACK
//#define USE_BLAS
//#define USE_MKL
//#define USE_CUBLAS


#ifdef USE_CUBLAS
#include "../include/util/cuda_util.h"
_cublas_Init_def cublas_init _cublas_Init;
#endif

#ifdef USE_BLAS
#ifndef USE_LAPACK
#define USE_LAPACK
#endif
#endif

#ifdef USE_MKL
#include "util/mkl_util.h"
#ifdef USE_BLAS
#undef USE_BLAS
#endif
#ifdef USE_LAPACK
#undef USE_LAPACK
#endif
#endif


#ifdef USE_LAPACK
#include "util/clapack_util.h"
#endif

//#ifdef USE_LAPACK
//#define USE_BLAS
//#endif

#ifdef USE_CUBLAS
const int use_cublas = 1;
#else
const int use_cublas = 0;
#endif

#ifdef USE_MKL
const int use_mkl = 1;
#else
const int use_mkl = 0;
#endif

#ifdef USE_GPU
const int use_gpu = 1;
#else
const int use_gpu = 0;
#endif

#define IDX2C(i,j,ld) (((j)*(ld)+(i)))

template<class T>
struct Matrix;

#if USE_GPU
#include "../include/util/cpp_amp.hpp"
#endif

#include "matcalc.hpp"

#include <omp.h>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb/stb_image_write.h"
extern "C" int stbi_write_bmp(char const *filename, int w, int h, int comp, const void  *data);
#endif


template<class T>
struct Matrix
{
	int m, n;
	T* v;
	
	inline Matrix(): m(0), n(0), v(NULL) { }
	inline Matrix( const int& m, const int& n ) :m(m), n(n)
	{
		v = new T[m*n];
	}

	inline Matrix(const std::vector<int>& vv) : m(vv.size()), n(1)
	{
		this->v = new T[m*n];
		for (int i = 0; i < v.size(); i++) this->v[i] = vv[i];
	}

	inline Matrix( const std::vector<T>& v ) :m(v.size()), n(1)
	{
		const int mn = m*n;
		this->v = new T[mn];

//#ifdef USE_CUBLAS
//		cublasXcopy(_cublas_Init.handle, m*n, (dnn_double*)v, (dnn_double*)this->v);
//		return;
//#endif

//#ifdef USE_BLAS
//		int one = 1;
//#ifndef USE_FLOAT
//		f2c_dcopy(&(int)mn, (dnn_double*)v, &one, (dnn_double*)this->v, &one);
//#else
//		f2c_scopy(&(int)mn, (dnn_double*)v, &one, (dnn_double*)this->v, &one);
//
//#endif
//		return;
//#endif

#ifdef USE_MKL
#ifndef USE_FLOAT
		cblas_dcopy(mn, (dnn_double*)&v[0], 1, (dnn_double*)this->v, 1);
#else
		cblas_scopy(mn, (dnn_double*)&v[0], 1, (dnn_double*)this->v, 1);
#endif
		return;
#endif

		for( int i = 0; i < m; ++i ) this->v[i] = v[i];
	}

	inline Matrix(const int* mat, int m_, int n_) :m(m_), n(n_)
	{
		this->v = new T[m*n];
		for (int i = 0; i < m*n; i++) this->v[i] = mat[i];
	}

	inline Matrix(const T* mat, int m_, int n_) :m(m_), n(n_)
	{
		const int mn = m*n;
		this->v = new T[mn];

		//#ifdef USE_CUBLAS
		//		cublasXcopy(_cublas_Init.handle, m*n, (dnn_double*)mat, (dnn_double*)this->v);
		//		return;
		//#endif

		//#ifdef USE_BLAS
		//		int one = 1;
		//#ifndef USE_FLOAT
		//		f2c_dcopy(&(int)mn, (dnn_double*)mat, &one, (dnn_double*)this->v, &one);
		//#else
		//		f2c_scopy(&(int)mn, (dnn_double*)mat, &one, (dnn_double*)this->v, &one);
		//
		//#endif
		//		return;
		//#endif

#ifdef USE_MKL
#ifndef USE_FLOAT
		cblas_dcopy(mn, (dnn_double*)mat, 1, (dnn_double*)this->v, 1);
#else
		cblas_scopy(mn, (dnn_double*)mat, 1, (dnn_double*)this->v, 1);
#endif
		return;
#endif

		for (int i = 0; i < mn; ++i) this->v[i] = mat[i];
	}

	inline Matrix( const Matrix<T>& mat )
	{
		m = mat.m; n = mat.n;
		if( m == 0 || n == 0 ){
			v = NULL;
		}
		else{
			v = new T[m*n];
			const int mn = m*n;

//#ifdef USE_CUBLAS
//			cublasXcopy(mn, (dnn_double*)mat.v, (dnn_double*)this->v);
//			return;
//#endif

//#ifdef USE_BLAS
//			int one = 1;
//#ifndef USE_FLOAT
//			f2c_dcopy(&(int)mn, (dnn_double*)mat.v, &one, (dnn_double*)this->v, &one);
//#else
//			f2c_scopy(&(int)mn, (dnn_double*)mat.v, &one, (dnn_double*)this->v, &one);
//
//#endif
//			return;
//#endif

#ifdef USE_MKL
#ifndef USE_FLOAT
			cblas_dcopy(mn, (dnn_double*)mat.v, 1, (dnn_double*)this->v, 1);
#else
			cblas_scopy(mn, (dnn_double*)mat.v, 1, (dnn_double*)this->v, 1);
#endif
			return;
#endif

#pragma omp parallel for
			for( int i = 0; i < mn; ++i )  v[i] = mat.v[i];
		}
	}

	inline bool isZero(const T eps)
	{
		const int mn = m*n;

		dnn_double sum = 0.0;
		for (int i = 0; i < mn; ++i)  sum += fabs(v[i]);

		if (sum*sum < eps) return true;
		return false;
	}

	void print_e(char* title = NULL)
	{
		if (title)printf("=== %s ===\n", title);
		printf("%d x %d\n", m, n);
		for (int i = 0; i < m; i++)
		{
			if (i > 3 && i < m - 4)
			{
				if ( i == 4 ) printf(".....\n");
				continue;
			}
			for (int j = 0; j < n; j++)
			{
				printf("%10.6e ", v[n*i + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	void print(char* title = NULL, char* format=NULL)
	{
		char* format_ = "%10.6f ";
		if (format) format_ = format;

		if (title )printf("=== %s ===\n", title);
		printf("%d x %d\n", m, n);
		for (int i = 0; i < m; i++)
		{
			if (i > 3 && i < m - 4)
			{
				if (i == 4) printf(".....\n");
				continue;
			}
			for (int j = 0; j < n; j++)
			{
				printf(format_, v[n*i + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	void print_f(char *text, ...)
	{
		char*	buf = new char[4096];

		va_list ptr;

		va_start(ptr, text);
		vsprintf(buf, text, ptr);
		va_end(ptr);

		print(buf);
		delete[] buf;
	}

	void print_csv(char* filename)
	{
		FILE* fp = fopen(filename, "w");
		if (fp == NULL)
		{
			fprintf(stderr, "file open error[write][%s]\n", filename);
			return;
		}
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				fprintf(fp, "%f", v[n*i + j]);
				if (j < n - 1) fprintf(fp, ",");
				else fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}


	Matrix<T>& operator = ( const Matrix& mat )
	{
		if( v != NULL ) delete [] v;

		m = mat.m; n = mat.n;
		if( m == 0 || n == 0 ){
			v = NULL;
			return *this;
		}
			
		v = new T[m*n];
		const int mn = m*n;

//#ifdef USE_CUBLAS
//		cublasXcopy(mn, (dnn_double*)mat.v, (dnn_double*)this->v);
//		return *this;
//#endif

//#ifdef USE_BLAS
//			int one = 1;
//#ifndef USE_FLOAT
//			f2c_dcopy(&(int)mn, (dnn_double*)mat.v, &one, (dnn_double*)this->v, &one);
//#else
//			f2c_scopy(&(int)mn, (dnn_double*)mat.v, &one, (dnn_double*)this->v, &one);
//
//#endif
//			return *this;
//#endif

#ifdef USE_MKL
#ifndef USE_FLOAT
			cblas_dcopy(mn, (dnn_double*)mat.v, 1, (dnn_double*)this->v, 1);
#else
			cblas_scopy(mn, (dnn_double*)mat.v, 1, (dnn_double*)this->v, 1);
#endif
			return *this;
#endif

#pragma omp parallel for
		for( int i = 0; i < mn; ++i )  v[i] = mat.v[i];

		return *this;
	}

	~Matrix ()
	{
		if( v != NULL ) delete [] v;
	}
	
	Matrix<T> chop(const T eps)
	{
		Matrix<T> ret(m, n);

#pragma omp parallel for
		for (int i = 0; i < m*n; ++i)
		{
			if (fabs(v[i]) < eps) ret.v[i] = 0.0;
			else ret.v[i] = v[i];
		}
		return ret;
	}

	static Matrix<T> unit ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		ret = ret.zeros(m, n);

#pragma omp parallel for
		for( int i = 0; i < std::min(m,n); ++i ) ret(i,i) = 1.0;
		return ret;
	}

	static Matrix<T> values(const int& m, const int& n, const double a)
	{
		Matrix<T> ret(m, n);
		const int mn = m*n;
#pragma omp parallel for
		for (int i = 0; i < mn; ++i) ret.v[i] = a;
		return ret;
	}

	static Matrix<T> ones ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		const int mn = m*n;

//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) ret.v[i] = 1.0;

		ones_(ret.v, m, n);
		return ret;
	}

	static Matrix<T> zeros ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		const int mn = m*n;

//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) ret.v[i] = 0.0;

		zeros_(ret.v, m, n);
		return ret;
	}

	static Matrix<T> transpose( const Matrix<T>& mat )
	{
 		int m = mat.m, n = mat.n;
 		Matrix<T> ret(n, m);

		const int mn = m*n;
 //#pragma omp parallel for
	//	for (int j = 0; j < n; ++j) {
	//		for (int i = 0; i < m; ++i) {
	//			ret(j, i) = mat(i, j);
	//		}
	//	}
		transpose_(mat.v, m, n, ret.v);
		return ret;
	}
	
	inline Matrix<T> transpose()
	{
		Matrix<T> ret(n, m);
		transpose_(v, m, n, ret.v);
		return ret;
	}

	static Matrix<T> hadamard ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);

//		const int mn = m*n;
//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) ret.v[i] = m1.v[i]*m2.v[i];

		hadamard_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);

		return ret;
	}
	Matrix<T> hadamard(const Matrix<T>& mat)
	{
		Matrix<T> ret(m, n);

		//		const int mn = m*n;
		//#pragma omp parallel for
		//		for( int i = 0; i < mn; ++i ) ret.v[i] = v[i]*mat.v[i];

		hadamard_(v, m, n, mat.v, mat.m, mat.n, ret.v);

		return ret;
	}
	Matrix<T> hadamard(const std::vector<T>& vec)
	{
		Matrix<T> ret(m, n);

		const int mn = m*n;
#pragma omp parallel for
		for (int i = 0; i < m; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				ret(i,j) = (*this)(i,j) * vec[i];
			}
		}
		return ret;
	}

	T norm_fro ( const Matrix<T>& mat )
	{
		int m = mat.m, n = mat.n;
		T ret = 0.0;

		const int mn = m*n;
#pragma omp parallel for reduction(+:ret)
		for( int i = 0; i < mn; ++i ) ret += mat.v[i]*mat.v[i];

		return sqrt(ret);
	}

	T norm()
	{
		T ret = 0.0;

		const int mn = m*n;
#pragma omp parallel for reduction(+:ret)
		for (int i = 0; i < mn; ++i) ret += v[i] * v[i];

		return sqrt(ret);
	}

	T Sum()
	{
		T d = 0.0;

		for (int i = 0; i < n*m; i++) d += v[i];

		return d;
	}


	void apply ( const std::function<double(const double&)>& func )
	{
		const int mn = m*n;
#if 0
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) this->v[i] = func(this->v[i]);
#else

#pragma omp parallel
		{
			const int N = 4;
			const int NN = mn / N;
#pragma omp for nowait
			for (int i = 0; i < NN; i++)
			{
				const int j = N * i;
				v[j] = func(v[j]);
				v[j + 1] = func(v[j + 1]);
				v[j + 2] = func(v[j + 2]);
				v[j + 3] = func(v[j + 3]);
				v[j + 4] = func(v[j + 4]);
			}
#pragma omp for
			for (int i = (NN==0)?0:mn - mn%N; i < mn; ++i)
			{
				v[i] = func(this->v[i]);
			}
		}
#endif
	}
	inline const T& operator () ( const int i, const int j ) const
	{
		return v[i*n + j];
	}

	inline T& operator () ( const int i, const int j )
	{
		return v[i*n + j];
	}

	Matrix<T>& operator += ( const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		const int mn = m*n;
		
		//Matrix<T> ret2 = *this;

// #pragma omp parallel for
// 		for( int i = 0; i < m; ++i )
// 			for( int j = 0; j < n; ++j )
// 				(*this)(i,j) += m1(i,j);

//		const int mn = m*n;
//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) this->v[i] += m1.v[i];

#ifdef USE_BLAS
		int one = 1;
		dnn_double one_f = dnn_double(1.0);
#ifndef USE_FLOAT
		f2c_daxpy(&(int)mn, &one_f, m1.v, &one, v, &one);
#else
		f2c_saxpy(&(int)mn, &one_f, m1.v, &one, v, &one);

		//if (1)
		//{
		//	for( int i = 0; i < mn; ++i ) ret2.v[i] += m1.v[i];
		//	double eps = 0.0;
		//	for (int i = 0; i < m*n; i++)
		//	{
		//		eps += fabs(v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}

#endif
		return *this;
#endif

		if (use_mkl && !use_gpu)
		{
#ifdef USE_MKL
#ifndef USE_FLOAT
			cblas_daxpy(mn, dnn_double(1.0), m1.v, 1, v, 1);
#else
			cblas_saxpy(mn, dnn_double(1.0), m1.v, 1, v, 1);
#endif
			return *this;
#endif
		}
		plus_eqd_(v, m, n, m1.v, m1.m, m1.n);

		return *this;
	}

	Matrix<T>& operator -= ( const Matrix<T>& m1 )
	{
		//Matrix<T> ret2 = *this;

		int m = m1.m, n = m1.n;
		const int mn = m*n;

#ifdef USE_BLAS
		int one = 1;
		dnn_double neg_one_f = dnn_double(-1.0);
#ifndef USE_FLOAT
		f2c_daxpy(&(int)mn, m1.v, &neg_one_f, v, &one);
#else
		f2c_saxpy(&(int)mn, m1.v, &neg_one_f, v, &one);

		//どこからも呼ばれていないので未検証
		//if (1)
		//{
		//	for( int i = 0; i < mn; ++i ) ret2.v[i] -= m1.v[i];
		//	double eps = 0.0;
		//	for (int i = 0; i < m*n; i++)
		//	{
		//		eps += fabs(v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}
#endif
		return *this;
#endif

#ifdef USE_MKL
#ifndef USE_FLOAT
		cblas_daxpy(mn, dnn_double(-1.0), m1.v,  1, v, 1);
#else
		cblas_saxpy(mn, dnn_double(-1.0), m1.v, 1, v, 1);
#endif
		return *this;
#endif

#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) this->v[i] -= m1.v[i];

		return *this;
	}

	Matrix<T>& operator *= ( const Matrix<T>& m1 )
	{
		*this = *this*m1;
		return *this;
	}


	Matrix<T>& operator *= ( const T& c )
	{

		//Matrix<dnn_double> ret2 = *this;

		const int mn = m*n;
//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) this->v[i] *= c;

#ifdef USE_BLAS
		int one = 1;
#ifndef USE_FLOAT
		f2c_dscal(&(int)mn, &(dnn_double)c, v, &one);
#else
		f2c_sscal(&(int)mn, &(dnn_double)c, v, &one);
		//if (1)
		//{
		//	for( int i = 0; i < mn; ++i ) ret2.v[i] *= c;
		//	double eps = 0.0;
		//	for (int i = 0; i < mn; i++)
		//	{
		//		eps += fabs(v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}

#endif
		return *this;
#endif

		if (use_mkl && !use_gpu)
		{
#ifdef USE_MKL
#ifndef USE_FLOAT
			cblas_dscal(mn, c, v, 1);
#else
			cblas_sscal(mn, c, v, 1);
#endif
			return *this;
#endif
		}
		scara_prod_(&v[0], m, n, c);

		return *this;
	}
	
	Matrix<T>& operator /= ( const T& c )
	{
		const int mn = m*n;
#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) this->v[i] /= c;

		return *this;
	}

	friend Matrix<T> operator + ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);

		const int mn = m*n;
//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) ret.v[i] = m1.v[i] + m2.v[i];

#ifdef USE_BLAS
		int one = 1;
		dnn_double one_f = dnn_double(1.0);
		ret = m2;
#ifndef USE_FLOAT
		f2c_daxpy(&(int)mn, &one_f, m1.v, &one, ret.v, &one);
#else
		f2c_saxpy(&(int)mn, &one_f, m1.v, &one, ret.v, &one);
		//どこからも呼ばれていないので未検証
		//if (1)
		//{
		//	for( int i = 0; i < mn; ++i ) ret2.v[i]  = m1.v[i] + m2.v[i];
		//	double eps = 0.0;
		//	for (int i = 0; i < mn; i++)
		//	{
		//		eps += fabs(ret.v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}

#endif
		return ret;
#endif

		if (use_mkl && !use_gpu)
		{
#ifdef USE_MKL
			ret = m2;
#ifndef USE_FLOAT
			cblas_daxpy(mn, dnn_double(1.0), m1.v, 1, ret.v, 1);
#else
			cblas_saxpy(mn, dnn_double(1.0), m1.v, 1, ret.v, 1);
#endif
			return ret;
#endif
		}
		plus_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);

		return ret;
	}
	
	friend Matrix<T> operator - ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);
		const int mn = m*n;

#ifdef USE_BLAS
		int one = 1;
		dnn_double neg_one_f = dnn_double(-1.0);
		ret = m1;
#ifndef USE_FLOAT
		f2c_daxpy(&(int)mn, &neg_one_f, m2.v, &one, ret.v, &one);
#else
		f2c_saxpy(&(int)mn, &neg_one_f, m2.v, &one, ret.v, &one);
		//どこからも呼ばれていないので未検証
		//if (1)
		//{
		//	Matrix<T> ret2(m, n);
		//	for( int i = 0; i < mn; ++i ) ret2.v[i]  = m1.v[i] - m2.v[i];
		//	double eps = 0.0;
		//	for (int i = 0; i < mn; i++)
		//	{
		//		eps += fabs(ret.v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}

#endif
		return ret;
#endif

#ifdef USE_MKL
		ret = m1;
#ifndef USE_FLOAT
		cblas_daxpy(mn, dnn_double(-1.0), m2.v, 1, ret.v, 1);
#else
		cblas_saxpy(mn, dnn_double(-1.0), m2.v, 1, ret.v, 1);
#endif
		return ret;
#endif

#pragma omp parallel for
		for( int i = 0; i < mn; ++i ) ret.v[i] = m1.v[i] - m2.v[i];

		return ret;
	}

	friend Matrix<T> operator * ( const T& c, const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);
		const int mn = m*n;
//#pragma omp parallel for
//		for( int i = 0; i < mn; ++i ) ret.v[i] = c*m1.v[i];
//

#ifdef USE_BLAS
		int one = 1;
		ret = m1;
#ifndef USE_FLOAT
		f2c_dscal(&(int)mn, &(dnn_double)c, ret.v, &one);
#else
		f2c_sscal(&(int)mn, &(dnn_double)c, ret.v, &one);
		//どこからも呼ばれていないので未検証
		//if (1)
		//{
		//	Matrix<T> ret2(m, n);
		//	for( int i = 0; i < mn; ++i ) ret2.v[i] = c*m1.v[i];
		//	double eps = 0.0;
		//	for (int i = 0; i < mn; i++)
		//	{
		//		eps += fabs(ret.v[i] - ret2.v[i]);
		//	}
		//	printf("\neps=%f\n", eps);
		//}
#endif
		return ret;
#endif

		if (use_mkl && !use_gpu)
		{
#ifdef USE_MKL
			ret = m1;
#ifndef USE_FLOAT
			cblas_dscal(mn, c, ret.v, 1);
#else
			cblas_sscal(mn, c, ret.v, 1);
#endif
			return ret;
#endif
		}

		prod_(m1.v, m1.m, m1.n, c, ret.v);

		return ret;
	}

	friend Matrix<T> operator * ( const Matrix<T>& m1, const T& c )
	{
		return c*m1;
	}
	 
	friend Matrix<T> operator / ( const Matrix<T>& m1, const T& c )
	{
		return (1.0/c)*m1;
	}

	friend std::ostream& operator << ( std::ostream& os, const Matrix<T>& A )
	{
		for( int i = 0; i < A.m; ++i ){
			for( int j = 0; j < A.n; ++j ){
				if( j != 0 ) os << " ";
				os << std::scientific << std::setprecision(3) << std::setw(10) << A(i,j);
			}
			std::cout << std::endl;
		}
		return os;
	}

	Matrix<T> sub ( int y, int x, int h, int w ) const
	{
		Matrix<T> ret(h, w);
#pragma omp parallel for
		for( int i = 0; i < w; ++i )
			for( int j = 0; j < h; ++j )
				ret(j, i) = (*this)(y + j, x + i);

		return ret;
	}
	void sub(int y, int x, int h, int w, const Matrix<T>& mat)
	{
		Matrix<T> ret(h, w);
#pragma omp parallel for
		for (int i = 0; i < w; ++i)
			for (int j = 0; j < h; ++j)
				(*this)(y + j, x + i) = mat(j, i);
	}

	inline dnn_double Tr()
	{
		dnn_double tr = 0.0;
		for (int i = 0; i < n; ++i)
			tr += (*this)(i, i);
		return tr;
	}

	Matrix<T> inv()
	{
		SVDcmp<T> svd(*this);

		return svd.inv();
	}

	Matrix<T> diag(Matrix<T>& X)
	{
		Matrix<T> d = Matrix<T>::zeros(X.n, X.n);

		for (int i = 0; i < X.n; i++) d(i, i) = X(i, i);

		return d;
	}
	void diag_vec(Matrix<T>& t)
	{
		for (int i = 0; i < n; i++) v[i] = t(i, i);
	}
	Matrix<T> diag(T* t)
	{
		Matrix<T> d = Matrix<T>::zeros(n, n);

		for (int i = 0; i < n; i++) d(i, i) = t[i];

		return d;
	}
	Matrix<T> diag(T x)
	{
		Matrix<T> d = Matrix<T>::zeros(this->n, this->n);

		for (int i = 0; i < X.n; i++) d(i, i) = x;

		return d;
	}

	Matrix<T> Centers(Matrix<T>& means)
	{
		Matrix<T> ret = *this;
		means = Matrix<T>().zeros(1, n);
		
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				means.v[j] += ret(i, j);

		for (int i = 0; i<n; i++)
			means.v[i] /= T(m);

		for (int i = 0; i<m; i++)
			for (int j = 0; j<n; j++)
				ret(i, j) -= means.v[j];
		return ret;
	}

	Matrix<T> Mean()
	{
		Matrix<T>& x = *this;
		Matrix<T>& means = Matrix<T>().zeros(1, n);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				means.v[j] += x(i, j);

		for (int i = 0; i<n; i++)
			means.v[i] /= T(m);

		return means;
	}

	Matrix<T> Std(Matrix<T>& means)
	{
		Matrix<T>& x = *this;
		Matrix<T>& sigma = Matrix<T>().zeros(1, n);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				sigma.v[j] += (x(i, j) - means.v[j])*(x(i, j) - means.v[j]);

		for (int i = 0; i<n; i++)
			sigma.v[i] /= T(m-1);

		return Sqrt(sigma);
	}

	Matrix<T> DeCenters(Matrix<T>& means)
	{
		Matrix<T> ret = *this;

		for (int i = 0; i<m; i++)
			for (int j = 0; j<n; j++)
				ret(i, j) += means.v[j];
		return ret;
	}

	T MaxDiag()
	{
		T max = (*this)(0, 0);
		for (int i = 1; i<m; i++)
			if ((*this)(i,i) > max)
				max = (*this)(i, i);
		return max;
	}

	Matrix<T> Reciprocal()
	{
		Matrix<T> d = *this;
		const int mn = d.m*d.n;

		for (int i = 0; i < mn; i++) d.v[i] = 1.0 / d.v[i];

		return d;
	}
	inline Matrix<T> Rand()
	{
		Matrix<T> d = *this;
		const int mn = d.m*d.n;

		for (int i = 0; i < mn; i++) d.v[i] = dnn_double(rand()) / RAND_MAX;
		return d;
	}

	inline Matrix<T> RandMT()
	{
		Matrix<T> d = *this;
		const int mn = d.m*d.n;

		std::mt19937 mt(1234);
		std::uniform_real_distribution<> rand01(0.0, 1.0);

		for (int i = 0; i < mn; i++) d.v[i] = rand01(mt);
		return d;
	}

	inline Matrix<T> one_sub_sqr(T alp)
	{
		Matrix<T> d = *this;
		const int mn = d.m*d.n;

		for (int i = 0; i < mn; i++) d.v[i] = alp*(1.0 - d.v[i]* d.v[i]);

		return d;
	}
	inline Matrix<T> mean_rows()
	{
		Matrix<T> vec(1, m);
		T sum;

		for (int i = 0; i< m; i++) {
			T sum = 0;
			for (int j = 0; j < n; j++)
				sum += (*this)(i, j);
			vec.v[i] = sum / T(n);
		}
		return vec;
	}


	double Max() const
	{
		const int mn = m*n;

		double mx = -1.0E32;
		for (int i = 0; i < mn; i++)
		{
			if (mx < v[i]) mx = v[i];
		}
		return mx;
	}
	double Min() const
	{
		const int mn = m*n;

		double mx = 1.0E32;
		for (int i = 0; i < mn; i++)
		{
			if (mx > v[i]) mx = v[i];
		}
		return mx;
	}
	double Max(int& idx) const
	{
		idx = -1;
		const int mn = m*n;

		double mx = -1.0E32;
		for (int i = 0; i < mn; i++)
		{
			if (mx < v[i])
			{
				mx = v[i];
				id = i;
			}
		}
		return mx;
	}
	double Min(int& idx) const
	{
		idx = -1;
		const int mn = m*n;

		double mx = 1.0E32;
		for (int i = 0; i < mn; i++)
		{
			if (mx > v[i])
			{
				mx = v[i];
				idx = i;
			}
		}
		return mx;
	}

	Matrix<dnn_double> removeRow(int row)
	{
		Matrix<dnn_double> ret(m - 1, n);
		for (int i = 0; i < m; i++)
		{
			if (i == row) continue;
			for (int j = 0; j < n; j++)
			{
				if (i > row)
				{
					ret(i - 1, j) = (*this)(i, j);
				}
				else
				{
					ret(i, j) = (*this)(i, j);
				}
			}
		}
		return ret;
	}

	Matrix<dnn_double> removeCol(int col_s, int col_e)
	{
		int N = col_e - col_s + 1;
		if (col_e < 0) N = n - col_s + 1;

		Matrix<dnn_double> ret = *this;
		for (int i = 0; i < N; i++)
		{
			if (ret.n == col_s) break;
			ret = ret.removeCol(col_s);
		}
		return ret;
	}



	Matrix<dnn_double> removeCol(int col)
	{
		Matrix<dnn_double> ret(m, n - 1);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (j == col) continue;
				if (j > col)
				{
					ret(i, j - 1) = (*this)(i, j);
				}
				else
				{
					ret(i, j) = (*this)(i, j);
				}
			}
		}
		return ret;
	}

	Matrix<dnn_double> Col(int col)
	{
		Matrix<dnn_double> ret(m, 1);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (j == col) 
				{
					ret(i, 0) = (*this)(i, j);
				}
			}
		}
		return ret;
	}
	Matrix<dnn_double> Row(int row)
	{
		Matrix<dnn_double> ret(1, n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				if (i == row)
				{
					ret(0, j) = (*this)(i, j);
				}
			}
		}
		return ret;
	}

	Matrix<dnn_double> appendRow(Matrix<dnn_double>& A)
	{
		Matrix<dnn_double> ret(m + A.m, n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				ret(i, j) = (*this)(i, j);
			}
		}
		for (int i = 0; i < A.m; i++)
		{
			for (int j = 0; j < A.n; j++)
			{
				ret(m+i, j) = A(i, j);
			}
		}
		return ret;
	}

	Matrix<dnn_double> appendCol(Matrix<dnn_double>& A)
	{
		Matrix<dnn_double> ret(m, n+A.n);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				ret(i, j) = (*this)(i, j);
			}
		}
		for (int i = 0; i < A.m; i++)
		{
			for (int j = 0; j < A.n; j++)
			{
				ret(i, n+j) = A(i, j);
			}
		}
		return ret;
	}

	void saveImage(const char* filename, int channel=3)
	{
		const int mn = m*n;
		double maxvalue = 0.0;
		for (int i = 0; i < channel * mn; i++)
		{
			if (fabs(v[i]) > maxvalue) maxvalue = fabs(v[i]);
		}
		unsigned char *data = new unsigned char[channel * mn];

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				int pos = (i*n + j);

				for (int k = 0; k < channel; k++)
				{
					data[channel * pos + k] = v[channel * pos + k] / maxvalue * 127. + 128.;
				}
			}
		}
		stbi_write_bmp(filename, n, m, channel, (void*)data);
	}

	inline operator T() const {
		if ( m == 1 && n == 1 )return (*this)(0, 0); 

		printf("mxn != 1x1\n");
		return 0.0;
	}
};

template<class T>
inline bool readImage(const char *filename, Matrix<T>&R, Matrix<T>&G, Matrix<T>&B, Matrix<T>&A)
{
	int x, y;
	unsigned char *data = 0;
	int nbit;
	data = stbi_load(filename, &x, &y, &nbit, 0);
	if (data == NULL)
	{
		printf("image file[%s] read error.\n", filename);
		return false;
	}
	//printf("height %d   width %d \n", y, x);

	R = Matrix<T>(x, y);
	G = Matrix<T>(x, y);
	B = Matrix<T>(x, y);
	A = Matrix<T>(x, y);

	//#pragma omp parallel for
	for (int i = 0; i<y; ++i) {
		for (int j = 0; j<x; ++j) {
			if (nbit == 1)	//8bit
			{
				int pos = (i*x + j);
				R.v[pos] = data[pos];
				G.v[pos] = data[pos];
				B.v[pos] = data[pos];
				A.v[pos] = 255.0;
			}
			if (nbit == 2)	//16bit
			{
				int pos = (i*x + j);
				R.v[pos] = data[pos * 2 + 0];
				G.v[pos] = data[pos * 2 + 1];
				B.v[pos] = data[pos * 2 + 2];
				A.v[pos] = 255.0;
			}
			if (nbit == 3)	//24
			{
				int pos = (i*x + j);
				R.v[pos] = data[pos * 3 + 0];
				G.v[pos] = data[pos * 3 + 1];
				B.v[pos] = data[pos * 3 + 2];
				A.v[pos] = 255.0;
			}
			if (nbit == 4)	//32
			{
				int pos = (i*x + j);
				R.v[pos] = data[pos * 4 + 0];
				G.v[pos] = data[pos * 4 + 1];
				B.v[pos] = data[pos * 4 + 2];
				A.v[pos] = data[pos * 4 + 3];
			}
		}
	}
	stbi_image_free(data);
	return true;
}


template<class T>
class column_major_Matrix
{
	inline void row_major_to_column_major( const Matrix<T>& mat, const int m, const int n)
	{
		M = Matrix<T>(m, n);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				M.v[IDX2C(i, j, m)] = mat.v[i*n + j];

	}
	Matrix<T> M;
public:
	column_major_Matrix()
	{}
	inline Matrix<T>& toColumn_major(Matrix<T>& mat)
	{
		row_major_to_column_major(mat, mat.m, mat.n);
		return M;
	}
	inline column_major_Matrix(const Matrix<T>& mat)
	{
		row_major_to_column_major(mat, mat.m, mat.n);
	}
	inline column_major_Matrix(Matrix<T>& mat)
	{
		row_major_to_column_major(mat, mat.m, mat.n);
	}
	inline Matrix<T>& get_column_major()
	{
		return M;
	}
	inline void set_column_major(Matrix<T>& mat)
	{
		M = mat;
	}

	inline void toRow_major(Matrix<T>& ret)
	{
		Matrix<T> M_ = M;
		const int m = M_.m;
		const int n = M_.n;
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				ret.v[i*n + j] = M_.v[IDX2C(i, j, m)];
	}
};

//calculation: ret = alpha * m1 * m2 + beta * ret
template<class T>
inline int cublasXgemm(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& ret, const T alpha, const T beta)
{
#ifdef USE_CUBLAS
	//printf("cuBLAS\n");
	int m = m1.m; // number of rows in A, C
	int n = m2.n; // number of columns in B, C
	int k = m1.n; // number of columns in A, rows in B
	T* d_A;
	T* d_B;
	T* d_C;

	column_major_Matrix<T> hA(m1);
	column_major_Matrix<T> hB(m2);
	column_major_Matrix<T> hC(ret);

	cudaMalloc((void**)&d_A, m * k * sizeof(T));
	cudaMalloc((void**)&d_B, k * n * sizeof(T));
	cudaMalloc((void**)&d_C, m * n * sizeof(T));
	cublasStatus_t status;

	cublasHandle_t handle;
	//// create the context
	//status = cublasCreate(&handle);
	//if (status != CUBLAS_STATUS_SUCCESS) {
	//	std::cerr << "***cublasCreate failed***\n";
	//	//return 2;
	//}
	handle = _cublas_Init.handle;

	// copy host matrix h_A to device matrix d_A [m, k]
	int ld_h_A = m;
	int ld_d_A = m;
	status = cublasSetMatrix(m, k, sizeof(T), hA.get_column_major().v, ld_h_A, d_A, ld_d_A);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasSetMatrix A failed***\n";
		return 2;
	}
	// copy host matrix h_B to device matrix d_B [k, n]
	int ld_h_B = k;
	int ld_d_B = k;
	status = cublasSetMatrix(k, n, sizeof(T), hB.get_column_major().v, ld_h_B, d_B, ld_d_B);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasSetMatrix B failed***\n";
		return 2;
	}
	// level 3 calculation: C = alpha * A * B + beta * C
	int ld_d_C = m;
#ifdef USE_FLOAT
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
		&alpha, d_A, ld_d_A, d_B, ld_d_B, &beta, d_C, ld_d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasSgemm failed***\n";
		return 2;
	}
#else
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
		&alpha, d_A, ld_d_A, d_B, ld_d_B, &beta, d_C, ld_d_C);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasDgemm failed***\n";
		return 2;
	}
#endif
	// copy device matrix d_C to host matrix h_C [m, n]
	status = cublasGetMatrix(m, n, sizeof(T), d_C, m, hC.get_column_major().v, m);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasGetMatrix C failed***\n";
		return 2;
	}
	hC.toRow_major(ret);

	//cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
#endif
	return -1;
}

template<class T>
inline int cublasXgemm0(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& ret, const T alpha, const T beta)
{
#ifdef USE_CUBLAS
	//printf("@@cuBLAS\n");
	return cublasXgemm(m1, m2, ret, alpha, beta);
#endif
	return -1;
}



template<class T>
inline int cublasXcopy(const int mn, const T* m1, T* ret)
{
#ifdef USE_CUBLAS
	//printf("cuBLAS\n");
	T* d_A;
	T* d_C;

	cudaMalloc((void**)&d_A, mn * sizeof(T));
	cudaMalloc((void**)&d_C, mn * sizeof(T));
	cublasStatus_t status;

	cublasHandle_t handle;
	//// create the context
	//status = cublasCreate(&handle);
	//if (status != CUBLAS_STATUS_SUCCESS) {
	//	std::cerr << "***cublasCreate failed***\n";
	//	//return 2;
	//}
	handle = _cublas_Init.handle;

	status = cublasSetVector(mn, sizeof(T), m1, 1, d_A, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasSetVector A failed***\n";
		return 2;
	}

#ifdef USE_FLOAT
	status = cublasScopy(handle, mn, d_A, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasScopy failed***\n";
		return 2;
	}
#else
	status = cublasDcopy(handle, mn, d_A, 1, d_C, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasDcopy failed***\n";
		return 2;
	}
#endif
	status = cublasGetVector(mn, sizeof(T), d_C, 1, ret, 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "***cublasGetMatrix C failed***\n";
		return 2;
	}

	//cublasDestroy(handle);
	cudaFree(d_A);
	cudaFree(d_C);

	return 0;
#endif
	return -1;
}

inline Matrix<dnn_double> operator * (const Matrix<dnn_double>& m1, const Matrix<dnn_double>& m2)
{
	int m = m1.m, n = m2.n, l = m1.n;

	Matrix<dnn_double> ret(m, n);

//	Matrix<dnn_double> ret2(m, n);
//#pragma omp parallel for
//	for( int i = 0; i < m; ++i )
//		for( int j = 0; j < n; ++j ){
//			dnn_double sum = 0.0;
//			for( int k = 0; k < l; ++k )
//				sum += m1(i,k)*m2(k,j);
//			ret2(i,j) = sum;
//		}
//
	if (use_cublas)
	{
#ifdef USE_CUBLAS
		if (m > MIN_SIZE_APPLYING_GPGPU && n > MIN_SIZE_APPLYING_GPGPU)
		{
			//printf("cuBLAS\n");
			ret.zeros(m, n);
			cublasXgemm(m1, m2, ret, dnn_double(1.0), dnn_double(0.0));

			//if (1)
			//{
			//	double eps = 0.0;
			//	for (int i = 0; i < m*n; i++)
			//	{
			//		eps += fabs(ret.v[i] - ret2.v[i]);
			//	}
			//	printf("\neps=%f\n", eps);
			//}
			return ret;
		}
#endif
	}

#ifdef USE_BLAS
	//Matrix<dnn_double> ret2(m, n);
	{
		//auto beg = std::chrono::system_clock::now();

		dnn_double ONE = 1.0, ZERO = 0.0;
		long int m = m1.m, n = m2.n, l = m1.n;
		if (m != 0 && n != 0 && l != 0)
		{
#ifndef USE_FLOAT
			f2c_dgemm("N", "N", &n, &m, &l, &ONE, &m2(0, 0), &n, &m1(0, 0), &l, &ZERO, &ret(0, 0), &n);
#else
			f2c_sgemm("N", "N", &n, &m, &l, &ONE, &m2(0, 0), &n, &m1(0, 0), &l, &ZERO, &ret(0, 0), &n);
#endif
		}
		//auto end = std::chrono::system_clock::now();
		//printf("  Matrix<dnn_double> operator *: %3lld\n",std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());

	}
#endif

#ifdef USE_MKL
	const dnn_double alpha = dnn_double(1.0), beta = dnn_double(0.0);
#ifndef USE_FLOAT
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m1.m, m2.n, m1.n, alpha, m1.v, m1.n, m2.v, m2.n, beta, ret.v, n);
#else
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, l, alpha, m1.v, l, m2.v, n, beta, ret.v, n);
#endif
	// A=m x k
	// B=k x n
	// C=m x n
	//m, n, k, alpha, A, k, B, n, beta, C, n);
#endif

	//auto beg = std::chrono::system_clock::now();
#ifndef USE_BLAS
#ifndef USE_MKL
	mull_(m1.v, m1.m, m1.n, m2.v, m2.m, m2.n, ret.v);
#endif
#endif

	//auto end = std::chrono::system_clock::now();
	//printf("  Matrix<dnn_double> operator *: %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());

	//if (1)
	//{
	//	double eps = 0.0;
	//	for (int i = 0; i < m*n; i++)
	//	{
	//		eps += fabs(ret.v[i] - ret2.v[i]);
	//	}
	//	printf("\neps=%f\n", eps);
	//}

	return ret;
}

//Matrix<float> operator * ( const Matrix<float>& m1, const Matrix<float>& m2 )
//{
//	int m = m1.m, n = m2.n, l = m1.n;
//		
//	Matrix<float> ret(m, n);
//#pragma omp parallel for
//	for( int i = 0; i < m; ++i )
//		for( int j = 0; j < n; ++j ){
//			float sum = 0.0;
//			for( int k = 0; k < l; ++k )
//				sum += m1(i,k)*m2(k,j);
//			ret(i,j) = sum;
//		}
//
//	return ret;
//}



template<class T>
std::vector<T> to_vector(Matrix<T>& X)
{
	std::vector<T> d(X.v, X.v + X.m*X.n);
	return d;
}

template<class T>
std::vector<T> diag_vector(Matrix<T>& X)
{
	std::vector<T> d(X.n);

	for (int i = 0; i < X.n; i++) d[i] = X(i, i);

	return d;
}

template<class T>
T SumAll(Matrix<T>& X)
{
	T d= 0.0;

	for (int i = 0; i < X.n*X.m; i++) d += X.v[i];

	return d;
}

template<class T>
inline Matrix<T> Tanh(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = tanh(mat.v[i]);

	return ret;
}
template<class T>
inline Matrix<T> InvSqrt(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = 1.0 / sqrt(mat.v[i]);
	return ret;
}
template<class T>
inline Matrix<T> Sqr(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = mat.v[i] * mat.v[i];
	return ret;
}
template<class T>
inline Matrix<T> Sqrt(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = sqrt(mat.v[i]);
	return ret;
}
template<class T>
inline Matrix<T> Abs(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = fabs(mat.v[i]);
	return ret;
}
template<class T>
inline Matrix<T> Pow(Matrix<T>& mat, T e)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = pow(mat.v[i], e);
	return ret;
}
template<class T>
inline Matrix<T> Sin(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = sin(mat.v[i]);
	return ret;
}
template<class T>
inline Matrix<T> Cos(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = sin(mat.v[i]);
	return ret;
}
template<class T>
inline Matrix<T> Tan(Matrix<T>& mat)
{
	Matrix<T>ret(mat.m, mat.n);
	const int mn = mat.m*mat.n;

	for (int i = 0; i < mn; i++) ret.v[i] = sin(mat.v[i]);
	return ret;
}

inline Matrix<dnn_double> toMatrix(std::vector<int>& change)
{
	Matrix<dnn_double> p(change.size(), 2);
	for (int x = 0; x < change.size(); x++)
	{
		p(x, 0) = x;
		p(x, 1) = change[x];
	}
	return p;
}

inline Matrix<dnn_double> Substitution(std::vector<int>& change)
{
	Matrix<dnn_double> s;

	s = Matrix<dnn_double>().zeros(change.size(), change.size());
	for (int x = 0; x < change.size(); x++)
	{
		s(x, change[x]) = 1.0;
	}

	return s;
}


//#ifndef USE_LAPACK
//#ifndef USE_MKL
template<class T>
class svd_decomposition_nr
{
	nrSVD<T> svd;
public:
	Matrix<T> U;
	Matrix<T> V;
	Matrix<T> Sigma;
	Matrix<T> *A;

	//void svdcmp(float **a, int m, int n, float w[], float **v)
	svd_decomposition_nr(Matrix<T>& mat)
	{
		A = &mat;

		const int m = mat.m;
		const int n = mat.n;

		T **u = svd.matrix(1, m, 1, n);
		T **v = svd.matrix(1, n, 1, n);
		T *w = svd.vector(1, n);

		U = Matrix<T>(m, n);
		V = Matrix<T>(n, n);
		Sigma = Matrix<T>(n, n);

		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < n + 1; j++)
				u[i][j] = mat(i - 1, j - 1);

		svd.svdcmp(u, m, n, w, v);
		
		for (int i = 1; i < m + 1; i++)
			for (int j = 1; j < n + 1; j++)
				U(i - 1, j - 1) = u[i][j];

		for (int i = 1; i < n + 1; i++)
			for (int j = 1; j < n + 1; j++)
				V(i - 1, j - 1) = v[i][j];
		
		//Sigma.zeros(n, n);
		for (int i = 1; i < n + 1; i++)
			for (int j = 1; j < n + 1; j++)
				Sigma(i - 1, j - 1) = 0.0;

		for (int i = 1; i < n + 1; i++)
			Sigma(i - 1, i - 1) = w[i];

#if 0
		printf("Product u*w*(v-transpose):\n");
		float **aa = svd.matrix(1, m, 1, n);
		for (int k = 1; k <= m; k++) {
			for (int l = 1; l <= n; l++) {
				aa[k][l] = 0.0;
				for (int j = 1; j <= n; j++)
					aa[k][l] += u[k][j] * w[j] * v[l][j];
			}
			for (int l = 1; l <= n; l++) printf("%.6f ", fabs(mat(k-1,l-1) - aa[k][l]));
			printf("\n");
		}
		svd.free_matrix(aa, 1, m, 1, n);
#endif

		svd.free_matrix(v, 1, n, 1, n);
		svd.free_matrix(u, 1, m, 1, n);
		svd.free_vector(w, 1, n);
	}

	Matrix<T> inv() const
	{
		//A^-1 = V * [diag(1/Wj)] *Vt;

		Matrix<T> Ws = Sigma;
		for (int i = 0; i < Sigma.n; i++)
		{
			Ws(i, i) = 1.0 / Sigma(i, i);
		}
		Matrix<T> Ut = U.transpose(U);
#if 0
		Matrix<T> t = *A*(V*Ws*Ut);
		std::cout << t << std::endl;
#endif
		return V*Ws*Ut;
	}

};
//#endif
//#endif


class _SingularValueDecomposition
{
	int error;
public:
	long int numberOfSingularValues;
	dnn_double *s;
	dnn_double *u;
	dnn_double *vt;

	inline int getStatus() const
	{
		return error;
	}

	_SingularValueDecomposition(int m_, int n_, dnn_double *a)
	{
		error = -999;
		char jobu = 'A', jobvt = 'S';
		long int m, n, lda, ldu, ldvt, lwork, info;
		m = lda = ldu = m_;
		n = ldvt = n_;
		lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));

		u = new dnn_double[m*m];
		vt = new dnn_double[n*n];
		numberOfSingularValues = std::min(m, n);
		s = new dnn_double[numberOfSingularValues];
		dnn_double* work = new dnn_double[lwork];

		dnn_double* B = new dnn_double[m*n];
		memcpy(B, a, sizeof(dnn_double)*m*n);

		long int lwork_tmp = -1;
#ifdef USE_FLOAT
#ifdef USE_MKL
		sgesvd_(&jobu, &jobvt, (const MKL_INT*)&m, (const MKL_INT*)&n, B, (const MKL_INT*)&lda, s, u, (const MKL_INT*)&ldu, vt, (const MKL_INT*)&ldvt, work, (const MKL_INT*)&lwork_tmp, (MKL_INT*)&info);
#else
		sgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork_tmp, &info);
#endif

#else
#ifdef USE_MKL
		dgesvd_(&jobu, &jobvt, (const MKL_INT*)&m, (const MKL_INT*)&n, B, (const MKL_INT*)&lda, s, u, (const MKL_INT*)&ldu, vt, (const MKL_INT*)&ldvt, work, (const MKL_INT*)&lwork_tmp, (MKL_INT*)&info);
#else
		dgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork_tmp, &info);
#endif
#endif
		if (info != 0)
		{
			printf("ERROR dgesvd_ %d\n", info);
			return;
		}
		if (info == 0)
		{
			lwork = static_cast<int>(work[0]);
			delete[] work;
			work = new dnn_double[lwork];
		}

		//memcpy(B, a, sizeof(double)*m*n);
		//memset(u, '\0', sizeof(double)*m*m);
		//memset(s, '\0', sizeof(double)*numberOfSingularValues);
		//memset(vt, '\0', sizeof(double)*n*n);
#ifdef USE_FLOAT
#ifdef USE_MKL
		sgesvd_(&jobu, &jobvt, (const MKL_INT*)&m, (const MKL_INT*)&n, B, (const MKL_INT*)&lda, s, u, (const MKL_INT*)&ldu, vt, (const MKL_INT*)&ldvt, work, (const MKL_INT*)&lwork, (MKL_INT*)&info);
#else
		sgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
#endif

#else
#ifdef USE_MKL
		dgesvd_(&jobu, &jobvt, (const MKL_INT*)&m, (const MKL_INT*)&n, B, (const MKL_INT*)&lda, s, u, ( MKL_INT*)&ldu, vt, ( MKL_INT*)&ldvt, work, ( MKL_INT*)&lwork, ( MKL_INT*)&info);
#else
		dgesvd_(&jobu, &jobvt, &m, &n, B, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
#endif
#endif

		if (info != 0)
		{
			printf("ERROR dgesvd_ %d\n", info);
		}
		error = info;

		delete[] B;
		delete[] work;
	}

	~_SingularValueDecomposition()
	{
		free(s);
		free(u);
		free(vt);
	}
};

template<class T>
class svd_decomposition
{
	int error;
	_SingularValueDecomposition* svd;
public:
	Matrix<T> U;
	Matrix<T> V;
	Matrix<T> Sigma;
	Matrix<T> *A;

	inline int getStatus() const
	{
		return error;
	}

	~svd_decomposition()
	{
		delete svd;
	}
	svd_decomposition() {}

	//void svdcmp(float **a, int m, int n, float w[], float **v)
	svd_decomposition(Matrix<T>& mat)
	{
		A = &mat;

		const int m = mat.m;
		const int n = mat.n;

		svd = new _SingularValueDecomposition(m, n, A->transpose().v);
		error = svd->getStatus();

		if (error != 0)
		{
			printf("_SingularValueDecomposition ERROR\n");
			return;
		}

		U = Matrix<T>(m, n);
		V = Matrix<T>(n, n);
		Sigma = Matrix<T>(svd->numberOfSingularValues, svd->numberOfSingularValues);


		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				U(i, j) = svd->u[j*m + i];

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				V(j, i) = svd->vt[j*n + i];

		//Sigma.zeros(n, n);
		for (int i = 0; i < svd->numberOfSingularValues; i++)
			for (int j = 0; j < svd->numberOfSingularValues; j++)
				Sigma(i, j) = 0.0;

		for (int i = 0; i < svd->numberOfSingularValues; i++)
			Sigma(i, i) = svd->s[i];

#if 0
		printf("u*w*(v-transpose):\n");
		{
			Matrix<T> x = U*Sigma*V;
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					fprintf(stdout, "%.4f ", fabs(x(i, j) - mat(i, j)));
				}
				printf("\n");
			}
		}
#endif
	}

	Matrix<T> inv() const
	{
		if (error != 0)
		{
			printf("_SingularValueDecomposition ERROR\n");
			return Matrix<dnn_double>();
		}
		//A^-1 = V * [diag(1/Wj)] *Vt;

		Matrix<T> Ws = Sigma;
		for (int i = 0; i < Sigma.n; i++)
		{
			Ws(i, i) = 1.0 / Sigma(i, i);
		}
		Matrix<T> Ut = U.transpose(U);
#if 0
		Matrix<T> t = *A*(V*Ws*Ut);
		std::cout << t << std::endl;
#endif
		return V*Ws*Ut;
	}

};

class eigenvalues_
{
public:
	dnn_double wr;
	dnn_double wi;
	std::vector<dnn_double> vl;
	std::vector<dnn_double> vr;

	bool operator<(const eigenvalues_& another) const
	{
		return wr < another.wr;
	}
};



class eigenvalues
{
	int error;
	/*
	The right eigenvector v(j) of A satisfies
	A * v(j) = lambda(j) * v(j)
	where lambda(j) is its eigenvalue.
	The left eigenvector u(j) of A satisfies
	u(j)**H * A = lambda(j) * u(j)**H
	where u(j)**H denotes the conjugate transpose of u(j).
	*/
	long int n;
	dnn_double *wr;
	dnn_double *wi;
	dnn_double *vl;
	dnn_double *vr;
	dnn_double *work;

	dnn_double* a;

	std::vector<Matrix<dnn_double>> leftR;
	std::vector<Matrix<dnn_double>> leftI;

	std::vector<Matrix<dnn_double>> rightR;
	std::vector<Matrix<dnn_double>> rightI;

	void getLeftVector()
	{
		//column_major_Matrix<dnn_double> cmmA;
		//Matrix<dnn_double> tmp(vl, n, n);
		//cmmA.set_column_major(tmp);
		//cmmA.toRow_major(tmp);

		//for (int i = 0; i < n; i++)
		//{
		//	Matrix<dnn_double> rv(1, n);
		//	Matrix<dnn_double> iv(1, n);
		//	int k = 0;
		//	int j = 0;
		//	while (j < n)
		//	{
		//		if (fabs(wi[j]) < 1.0E-6)
		//		{
		//			rv(0, k) = tmp.v[i + n*j];
		//			iv(0, k) = 0.0;
		//			j++;
		//		}
		//		else
		//		{
		//			rv(0, k) = tmp.v[i + n*j];
		//			iv(0, k) = tmp.v[i + n*(j + 1)];
		//			k++;

		//			rv(0, k) = tmp.v[i + n*j];
		//			iv(0, k) = -tmp.v[i + n*(j + 1)];
		//			j += 2;
		//		}
		//		k++;
		//	}
		//	leftR.push_back(rv);
		//	leftI.push_back(iv);
		//}


		for (int i = 0; i < n; i++)
		{
			Matrix<dnn_double> rv(1, n);
			Matrix<dnn_double> iv(1, n);
			int k = 0;
			int j = 0;
			while (j < n)
			{
				if (fabs(wi[j]) < 1.0E-6)
				{
					rv(0, k) = vl[i + n*j];
					iv(0, k) = 0.0;
					j++;
				}
				else
				{
					rv(0, k) = vl[i + n*j];
					iv(0, k) = vl[i + n*(j + 1)];
					k++;

					rv(0, k) = vl[i + n*j];
					iv(0, k) = -vl[i + n*(j + 1)];
					j += 2;
				}
				k++;
			}
			leftR.push_back(rv);
			leftI.push_back(iv);
		}
	}
	//ちなみに、高校課程で習うのは右固有ベクトル
	void getRightVector()
	{
		//column_major_Matrix<dnn_double> cmmA;
		//Matrix<dnn_double> tmp(vl, n, n);
		//cmmA.set_column_major(tmp);
		//cmmA.toRow_major(tmp);

		//tmp.print("tmp");
		//for (int i = 0; i < n; i++) printf("%.2f ", wi[i]);
		//printf("\n");

		//for (int i = 0; i < n; i++)
		//{
		//	Matrix<dnn_double> rv(1, n);
		//	Matrix<dnn_double> iv(1, n);

		//	int k = 0;
		//	int j = 0;
		//	while (j < n)
		//	{
		//		if (fabs(wi[j]) < 1.0E-6)
		//		{
		//			rv(0, k) = tmp(i, j);
		//			iv(0, k) = 0.0;
		//			j++;
		//		}
		//		else
		//		{
		//			rv(0, k) = tmp(i, j);
		//			iv(0, k) = tmp(i, j+1);

		//			k++;
		//			rv(0, k) = tmp(i, j);
		//			iv(0, k) = -tmp(i, j+1);
		//			j += 2;
		//		}
		//		k++;
		//	}
		//	rightR.push_back(rv);
		//	rightI.push_back(iv);
		//}

		for (int i = 0; i < n; i++)
		{
			Matrix<dnn_double> rv(1, n);
			Matrix<dnn_double> iv(1, n);

			int k = 0;
			int j = 0;
			while (j < n)
			{
				if (fabs(wi[j]) < 1.0E-6)
				{
					rv(0, k) = vr[i + n*j];
					iv(0, k) = 0.0;
					j++;
				}
				else
				{
					rv(0, k) = vr[i + n*j];
					iv(0, k) = vr[i + n*(j + 1)];

					k++;
					rv(0, k) = vr[i + n*j];
					iv(0, k) = -vr[i + n*(j + 1)];
					j += 2;
				}
				k++;
			}
			rightR.push_back(rv);
			rightI.push_back(iv);
		}
	}

	std::vector<int> index;
	struct __sort_w
	{
		dnn_double w;
		int index;

		bool operator<(const __sort_w& right) const {
			return right.w < w;
		}
	};
	void sort(std::vector<int>& index)
	{
		std::vector<struct __sort_w> wr_(n);
		for (int i = 0; i < n; i++)
		{
			wr_[i].index = i;
			wr_[i].w = wr[i];
		}
		std::sort(wr_.begin(), wr_.end());

		index.resize(n);
		for (int i = 0; i < n; i++)
		{
			index[i] = wr_[i].index;
		}
	}

public:
	long int info;
	column_major_Matrix<dnn_double> cmmA;

	inline int getStatus() const
	{
		return error;
	}
	eigenvalues() {}

	eigenvalues(Matrix<dnn_double>&A)
	{
		set(A);
	}
	
	void set(Matrix<dnn_double>&A)
	{
		leftR.clear();
		leftI.clear();

		rightR.clear();
		rightI.clear();

		error = -999;
		a = A.v;
		a = cmmA.toColumn_major(A).v;

		n = A.n;
		info = 0;

		wr = new dnn_double[n];
		wi = new dnn_double[n];
		vl = new dnn_double[n * n];
		vr = new dnn_double[n * n];
		work = NULL;

		index.resize(n);
		for (int i = 0; i < n; i++) index[i] = i;
	}

	std::vector<int> getIndex()
	{
		return index;
	}
	int calc(bool sort_value = false)
	{
		error = -999;

		char *jobvl = "V";
		char *jobvr = "V";
		long int lwork = -1;
		dnn_double pwork = 0;
#ifdef USE_FLOAT
		sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, &pwork, &lwork, &info);

		if (info != 0)
		{
			error = info;
			return info;
		}
		lwork = (long int)pwork;
		work = new dnn_double[lwork];
		//printf("info=%d lwork=%d\n", info, lwork);

		sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, work, &lwork, &info);
		//printf("info=%d\n", info);
		if (info != 0)
		{
			error = info;
			return info;
		}
#else
		dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, &pwork, &lwork, &info);
		if (info != 0)
		{
			error = info;
			return info;
		}
		lwork = (int)pwork;
		work = new dnn_double[lwork];
		//printf("info=%d lwork=%d\n", info, lwork);

		dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, vl, &n, vr, &n, work, &lwork, &info);
		//printf("info=%d\n", info);
		if (info != 0)
		{
			error = info;
			return info;
		}
#endif
		if (work) delete[] work;

		error = info;
		if (info == 0)
		{
			getLeftVector();
			getRightVector();
		}
		
		if (sort_value ) sort(index);

		return info;
	}


	Matrix<dnn_double> getRealValue()
	{
		Matrix<dnn_double> v(1, n);
		for (int i = 0; i < n; i++) v(0, i) = wr[index[i]];
		return v;
	}
	Matrix<dnn_double> getImageValue()
	{
		Matrix<dnn_double> v(1, n);
		for (int i = 0; i < n; i++) v(0, i) = wi[index[i]];
		return v;
	}

	std::vector<Matrix<dnn_double>> getLeftVector(const int i)
	{
		std::vector<Matrix<dnn_double>> v;

		Matrix<dnn_double> vr(n, 1);
		Matrix<dnn_double> vi(n, 1);

		for (int j = 0; j < n; j++)
		{
			vr(j, 0) = leftR[j](0, index[i]);
			vi(j, 0) = leftI[j](0, index[i]);
		}
		v.push_back(vr);
		v.push_back(vi);
		return v;
	}

	std::vector<Matrix<dnn_double>> getRightVector(const int i)
	{
		std::vector<Matrix<dnn_double>> v;

		Matrix<dnn_double> vr(n, 1);
		Matrix<dnn_double> vi(n, 1);
		for (int j = 0; j < n; j++)
		{
			vr(j, 0) = rightR[j](0, index[i]);
			vi(j, 0) = rightI[j](0, index[i]);
		}
		v.push_back(vr);
		v.push_back(vi);
		return v;
	}


	~eigenvalues()
	{
		delete[] wr;
		delete[] wi;
		delete[] vl;
		delete[] vr;
	}
};

class linear_equation
{
	int error;
public:
	Matrix<dnn_double> x;

	inline int getStatus() const
	{
		return error;
	}
	linear_equation()
	{
		error = -999;
	}
	int solv(Matrix<dnn_double>&A, Matrix<dnn_double>&B)
	{
		if (A.m != A.n || A.n != B.m)
		{
			error = -99;
			return error;
		}
		column_major_Matrix<dnn_double> cmmA(A);
		column_major_Matrix<dnn_double> cmmB(B);
		x = cmmB.get_column_major();
		Matrix<dnn_double> a = cmmA.get_column_major();

		long int n = a.m;
		long int nrhs = 1;
		long int lda = a.m;
		long int* ipiv = new long int[n];
		long int ldb = x.m;
		long int info = 0;

#ifdef USE_FLOAT
		sgesv_(&n, &nrhs, &a.v[0], &lda, ipiv, &x.v[0], &ldb, &info);
#else
		dgesv_(&n, &nrhs, &a.v[0], &lda, ipiv, &x.v[0], &ldb, &info);
#endif
		delete[] ipiv;

		Matrix<dnn_double> xx = x;
		column_major_Matrix<dnn_double> cmmX(xx);
		cmmX.toRow_major(x);
		error = info;
		return info;
	}

	Matrix<dnn_double> inv(Matrix<dnn_double>&A)
	{
		column_major_Matrix<dnn_double> cmmA(A);
		Matrix<dnn_double>& a = cmmA.get_column_major();
		Matrix<dnn_double> b(A.m, A.m);
		b = b.zeros(a.m, a.m);
		b = b.unit(a.m, a.m);

		column_major_Matrix<dnn_double> cmmB(b);
		b = cmmA.get_column_major();

		long int n = a.m;
		long int nrhs = a.m;
		long int lda = a.m;
		long int* ipiv = new long int[n];
		long int ldb = b.m;
		long int info = 0;

#ifdef USE_FLOAT
		sgesv_(&n, &nrhs, &a.v[0], &lda, ipiv, &b.v[0], &ldb, &info);
#else
		dgesv_(&n, &nrhs, &a.v[0], &lda, ipiv, &b.v[0], &ldb, &info);
#endif
		column_major_Matrix<dnn_double> cmm(b);
		Matrix<dnn_double>& B = b;
		cmm.toRow_major(B);
		printf("info:%d\n", info);
		//B.print();

		delete[] ipiv;
		error = info;
		return B;
	}
};

//過剰または過小定義の連立一次方程式 [A]{X}={B}の最小2乗または最小ノルム解をQRまたはLQ分解を用いて求める
class linear_east_square
{
	int error;
public:
	Matrix<dnn_double> x;

	inline int getStatus() const
	{
		return error;
	}
	linear_east_square()
	{
		error = -999;
	}

	//行数の方が大きいまたは等しい場合:過剰定義連立方程式の最小2乗解
	//列数の方大きい場合:最小ノルム解
	//It is assumed that A has full rank.
	int fit(Matrix<dnn_double>&A, Matrix<dnn_double>&B)
	{
		column_major_Matrix<dnn_double> cmmB(B);
		column_major_Matrix<dnn_double> cmmA(A);
		Matrix<dnn_double>& b = cmmB.get_column_major();
		Matrix<dnn_double>& a = cmmA.get_column_major();

		//Matrix<dnn_double> b = B;
		//Matrix<dnn_double> a = A;

		long int m = a.m;
		long int n = a.n;
		long int nrhs = b.n;
		long int lda = a.m;
		long int ldb = std::max(m, n);
		long int info = 0;

		long int lwork = -1;
		dnn_double wkopt;
		dnn_double* work;

		dnn_double* b_ = NULL;
		bool b_alloc = false;

		if (ldb*nrhs >= b.m*b.n)
		{
			b_alloc = true;
			b_ = new dnn_double[ldb*nrhs];
			memcpy(b_, &b.v[0], sizeof(dnn_double)*b.m*b.n);
		}
		else
		{
			b_ = &b.v[0];
		}

#ifdef USE_FLOAT
		sgels_("N", &m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &wkopt, &lwork, &info);

		printf("info:%d\n", info);
		error = info;
		if (info != 0)
		{
			return error;
		}

		lwork = (int)wkopt;
		work = new dnn_double[lwork];
		sgels_("N", &m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, work, &lwork, &info);
		error = info;
		printf("info:%d\n", info);
		delete[] work;
		if (info != 0)
		{
			return error;
		}
		{
			int i, j;
			printf("\n %s\n", "Least squares solution");
			for (i = 0; i < n; i++) {
				for (j = 0; j < nrhs; j++) printf(" %6.2f", x.v[i + j*lda]);
				printf("\n");
			}
		}
#else
		dgels_("N", &m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &wkopt, &lwork, &info);

		printf("info:%d\n", info);
		error = info;
		if (info != 0)
		{
			return error;
		}

		lwork = (int)wkopt;
		work = new dnn_double[lwork];
		dgels_("N", &m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, work, &lwork, &info);
		error = info;
		printf("info:%d\n", info);
		delete[] work;
		if (info != 0)
		{
			return error;
		}

		x = Matrix<dnn_double>(ldb, nrhs);
		for (int i = 0; i < x.m; i++)
			for (int j = 0; j < x.n; j++)
				x(i, j) = b_[i*x.n + j];

		if (b_alloc ) delete[] b_;

		column_major_Matrix<dnn_double> cmm;
		cmm.set_column_major(x);
		cmm.toRow_major(x);

#endif
		error = info;
		return info;
	}

	//行数の方が大きいまたは等しい場合:過剰定義連立方程式の最小2乗解
	//列数の方大きい場合:最小ノルム解
	//using the singular value decomposition (SVD) of A. A is an M-by-N
	//matrix which may be rank - deficient.
	int fit2(Matrix<dnn_double>&A, Matrix<dnn_double>&B)
	{
		column_major_Matrix<dnn_double> cmmB(B);
		column_major_Matrix<dnn_double> cmmA(A);
		Matrix<dnn_double>& b = cmmB.get_column_major();
		Matrix<dnn_double>& a = cmmA.get_column_major();

		//Matrix<dnn_double> b = B;
		//Matrix<dnn_double> a = A;

		long int m = a.m;
		long int n = a.n;
		long int nrhs = b.n;
		long int lda = a.m;
		long int ldb = std::max(m, n);
		long int info = 0;

		dnn_double* b_ = NULL;
		bool b_alloc = false;

		if (ldb*nrhs >= b.m*b.n)
		{
			b_alloc = true;
			b_ = new dnn_double[ldb*nrhs];
			memcpy(b_, &b.v[0], sizeof(dnn_double)*b.m*b.n);
		}
		else
		{
			b_ = &b.v[0];
		}

		std::vector<dnn_double> S(std::min(m, n), 1);
		dnn_double RCOND = 0.01;
		long int rank = 0;

		long int lwork = -1;
		dnn_double wkopt;
		dnn_double* work;
#ifdef USE_FLOAT
		sgelss_(&m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &S[0], &RCOND, &rank, &wkopt, &lwork, &info);

		printf("info:%d\n", info);
		error = info;
		if (info != 0)
		{
			return error;
		}

		lwork = (int)wkopt;
		work = new dnn_double[lwork];
		sgelss_(&m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &S[0], &RCOND, &rank, work, &lwork, &info);
		error = info;
		printf("info:%d\n", info);
		delete[] work;
		if (info != 0)
		{
			return error;
		}
		{
			int i, j;
			printf("\n %s\n", "Least squares solution");
			for (i = 0; i < n; i++) {
				for (j = 0; j < nrhs; j++) printf(" %6.2f", x.v[i + j*lda]);
				printf("\n");
			}
		}
#else
		dgelss_(&m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &S[0], &RCOND, &rank, &wkopt, &lwork, &info);

		printf("info:%d\n", info);
		error = info;
		if (info != 0)
		{
			return error;
		}

		lwork = (int)wkopt;
		work = new dnn_double[lwork];
		dgelss_(&m, &n, &nrhs, &a.v[0], &lda, b_, &ldb, &S[0], &RCOND, &rank, work, &lwork, &info);
		error = info;
		printf("info:%d\n", info);
		delete[] work;
		if (info != 0)
		{
			return error;
		}

		x = Matrix<dnn_double>(ldb, nrhs);
		for (int i = 0; i < x.m; i++)
			for (int j = 0; j < x.n; j++)
				x(i, j) = b_[i*x.n + j];

		if (b_alloc) delete[] b_;

		column_major_Matrix<dnn_double> cmm;
		cmm.set_column_major(x);
		cmm.toRow_major(x);

#endif
		error = info;
		return info;
	}

};

template<class T>
inline void n_mat_multiplication_cpu(const int num, const std::vector<Matrix<T>>& A, const std::vector<Matrix<T>>& B, std::vector<Matrix<T>>& C)
{
	if (num > A.size())
	{
		fprintf(stderr, "n_mat_multiplication_cpu call ERROR.\n");
		return;
	}
	if (A.size() == B.size() && A.size() <= C.size())
	{
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			C[i] = A[i] * B[i];
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			C[i] = A[i] * B[0];
		}
	}
}

template<class T>
inline void n_mat_multiplication(const int num, const std::vector<Matrix<T>>& A, const std::vector<Matrix<T>>& B, std::vector<Matrix<T>>& C)
{
#ifdef USE_GPU
	std::vector<T*> a(A.size());
	std::vector<T*> b(B.size());
	std::vector<T*> c(C.size());

	//printf("%d %d %d %d\n", num, (int)a.size(), (int)b.size(), (int)c.size());

	if (a.size() == b.size() && a.size() <= c.size() || a.size() <= c.size() && b.size() == 1)
	{
		a.resize(num);
		c.resize(num);
		//OK!!
	}
	else
	{
		fprintf(stderr, "n_mat_multiplication call ERROR.\n");
		return;
	}

	if (A[0].m*A[0].n < 1024 * 1024 && num < 10)
	{
		n_mat_multiplication_cpu(num, A, B, C);
		return;
	}
	//printf("gpu\n");

	if (a.size() == b.size())
	{
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			a[i] = A[i].v;
			b[i] = B[i].v;
			c[i] = C[i].v;
		}
	}
	else
	{
		for (int i = 0; i < num; ++i)
		{
			a[i] = A[i].v;
			c[i] = C[i].v;
		}
		b[0] = B[0].v;
	}
	const int am = A[0].m;
	const int an = A[0].n;
	const int bm = B[0].m;
	const int bn = B[0].n;

	mull_gpu2(a, am, an, b, bm, bn, c);

	//Check!!
#if 0
	std::vector<Matrix<T>> X(num);

	if (a.size() == b.size() && a.size() <= c.size())
	{
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			X[i] = A[i] * B[i];
		}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i < num; ++i)
		{
			X[i] = A[i] * B[0];
		}
	}
	T _d = T(0.0);
	for (int i = 0; i < num; ++i)
	{
		for (int k = 0; k < am*bn; ++k)
		{
			_d += fabs(X[i].v[k] - C[i].v[k]);
		}
	}
	printf("_d = %f\n", _d);
#endif

#else
	n_mat_multiplication_cpu(num, A, B, C);
#endif
}


template<class T>
inline Matrix<dnn_double> zca_whitening_matrix(Matrix<T>& X, const T eps = 1.0e-5, bool bias = false)
{

	//const double eps = 1.0E-5;	//Whitening constant: prevents division by zero

	const int sz = X.m*X.n;
	std::vector<T> sum(X.m, 0.0);

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sum[i] += X(i, j);
		}
	}

	std::vector<T> mu(X.m, 0.0);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		mu[i] = sum[i] / X.n;
		//printf("%f\n", mu[i]);
	}


	//不偏共分散行列	Sigma = (X-mu) * (X-mu)' / N
	Matrix<T> sigma = Matrix<T>::zeros(X.m, X.n);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sigma(i, j) = (X(i, j) - mu[i]);
		}
	}
	//std::cout << sigma << "\n";
	if (bias)
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / X.n;
	}
	else
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / (X.n - 1.0);
	}

	//std::cout << sigma << "\n";

	//Singular Value Decomposition. X = U * np.diag(W) * V
	svd_decomposition<T> svd(sigma);
	//std::cout << "U" << svd.U << "\n";
	//std::cout << "W" << svd.W << "\n";
	//std::cout << "V" << svd.V << "\n";

	for (int i = 0; i < svd.W.n; i++) svd.W(i, i) = 1.0 / sqrt(svd.W(i, i) + eps);

	//std::cout << "W" << svd.W << "\n";

	//ZCA Whitening matrix: U * diag(1/sqrt(W+eps) * U'
	return svd.U*(svd.W * Matrix<T>::transpose(svd.U));
}

#if 10
template<class T>
inline Matrix<T> zca_whitening_matrix2(Matrix<T>& X, const double eps = 1.0e-5, bool bias = false)
{

	//const double eps = 1.0E-8;	//Whitening constant: prevents division by zero

	const int sz = X.m*X.n;
	double sum = 0.0;

	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sum += X(i, j);
		}
	}

	double mu = 0.0;
	mu = sum / sz;


	//不偏共分散行列	Sigma = (X-mu) * (X-mu)' / N
	Matrix<T> sigma = Matrix<T>::zeros(X.m, X.n);
#pragma omp parallel for
	for (int i = 0; i < X.m; i++)
	{
		for (int j = 0; j < X.n; j++)
		{
			sigma(i, j) = (X(i, j) - mu);
		}
	}
	//std::cout << sigma << "\n";
	if (bias)
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / X.n;
	}
	else
	{
		sigma = sigma*Matrix<T>::transpose(sigma) / (X.n - 1.0);
	}

	//std::cout << sigma << "\n";

	//Singular Value Decomposition. X = U * np.diag(W) * V
	svd_decomposition<T> svd(sigma);
	//std::cout << "U" << svd.U << "\n";
	//std::cout << "W" << svd.W << "\n";
	//std::cout << "V" << svd.V << "\n";

	for (int i = 0; i < svd.W.n; i++) svd.W(i, i) = 1.0 / sqrt(svd.W(i, i) + eps);

	//std::cout << "W" << svd.W << "\n";

	//ZCA Whitening matrix: U * diag(1/sqrt(W+eps) * U'
	return svd.U*(svd.W * svd.U.transpose());
}
#endif


#endif
