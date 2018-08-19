#ifndef _CLAPCK_UTIL_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define _CLAPCK_UTIL_HPP

#if 0
#pragma comment(lib, "../../third_party/clapack/lib/cLAPACK.lib")
#pragma comment(lib, "../../third_party/clapack/lib/BLAS.lib")
#pragma comment(lib, "../../third_party/clapack/lib/libI77.lib")
#pragma comment(lib, "../../third_party/clapack/lib/libF77.lib")
#define __BLASWRAP_H
#else
#pragma comment(lib, "../../third_party/clapack-3.2.1/lib/LAPACK.lib")
#pragma comment(lib, "../../third_party/clapack-3.2.1/lib/BLAS.lib")
#pragma comment(lib, "../../third_party/clapack-3.2.1/lib/libf2c.lib")
#endif

extern "C"
{
	//FILE _iob[] = { *stdin, *stdout, *stderr };
	//extern "C" FILE * __cdecl __iob_func(void)
	//{
	//	return _iob;
	//}
	FILE *Snap = 0;
	
	extern "C"  void clapack_dummy()
	{
		char c[8];
		sprintf(c, "%s", "dummy");
		printf("%s", c);
	}
	
#ifndef USE_MKL
	extern "C" int dgesvd_(char*jobu, char*jobvt, long int*m, long int*n, double*A, long int*lda, double*s, double*u, long int*ldu, double*vt, long int*ldvt, double*work, long int*lwork_tmp, long int*info);
	extern "C" int sgesvd_(char*jobu, char*jobvt, long int*m, long int*n, float*A, long int*lda, float*s, float*u, long int*ldu, float*vt, long int*ldvt, float*work, long int*lwork_tmp, long int*info);
	
	extern "C" int dgesv_(long int *n, long int *nrhs, double *a, long int *lda, long int *ipiv, double *b, long int *ldb, long int *info);
	extern "C" int sgesv_(long int *n, long int *nrhs, float *a, long int *lda, long int *ipiv, float *b, long int *ldb, long int *info);

	extern "C" int dgeev_(char *jobvl, char *jobvr, long int *n, double *a,
		long int *lda, double *wr, double *wi, double *vl,
		long int *ldvl, double *vr, long int *ldvr, double *work,
		long int *lwork, long int *info);
	extern "C" int sgeev_(char *jobvl, char *jobvr, long int *n, float *a,
		long int *lda, float *wr, float *wi, float *vl,
		long int *ldvl, float *vr, long int *ldvr, float *work,
		long int *lwork, long int *info);

	extern "C" void dgels_(char* trans, long int* m, long int* n, long int* nrhs, double* a, long int* lda,
		double* b, long int* ldb, double* work, long int* lwork, long int* info);
	extern "C" void sgels_(char* trans, long int* m, long int* n, long int* nrhs, float* a, long int* lda,
		float* b, long int* ldb, float* work, long int* lwork, long int* info);

	extern "C" void sgelss_(const long int* m, const long int* n, const long int* nrhs,
		float* a, const long int* lda, float* b, const long int* ldb,
		float* s, const float* rcond, long int* rank, float* work,
		const long int* lwork, long int* info );

	extern "C" void dgelss_(const long int* m, const long int* n, const long int* nrhs,
		double* a, const long int* lda, double* b, const long int* ldb,
		double* s, const double* rcond, long int* rank, double* work,
		const long int* lwork, long int* info);
#endif

	//BLAS
#ifndef USE_BLAS_WRAP
	extern "C" int f2c_dgemm(char* transa, char* transb, long int* m, long int* n, long int* k, double* alpha, const double* A, long int* lda, const double* B, long int* ldb, double* beta, double* C, long int* ldc);
	extern "C" int f2c_sgemm(char* transa, char* transb, long int* m, long int* n, long int* k, float* alpha, const float* A, long int* lda, const float* B, long int* ldb, float* beta, float* C, long int* ldc);
	
	extern "C" void f2c_daxpy(int *, double *, double *, int *, double *, int *);
	extern "C" void f2c_saxpy( int *, float *, float *,  int *, float *, int *);

	extern "C" void f2c_sscal(int*, float*, float*, int*);
	extern "C" void f2c_dscal(int*, double*, double*, int*);

	extern "C" void f2c_scopy(int*, float*, int*, float*, int*);
	extern "C" void f2c_dcopy(int*, double*, int*, double*, int*);
#else
	extern "C" int dgemm_(char* transa, char* transb, long int* m, long int* n, long int* k, double* alpha, const double* A, long int* lda, const double* B, long int* ldb, double* beta, double* C, long int* ldc);
	extern "C" int sgemm_(char* transa, char* transb, long int* m, long int* n, long int* k, float* alpha, const float* A, long int* lda, const float* B, long int* ldb, float* beta, float* C, long int* ldc);

	extern "C" void daxpy_(int *, double *, double *, int *, double *, int *);
	extern "C" void saxpy_(int *, float *, float *, int *, float *, int *);

	extern "C" void sscal_(int*, float*, float*, int*);
	extern "C" void dscal_(int*, double*, double*, int*);

	extern "C" void scopy_(int*, float*, int*, float*, int*);
	extern "C" void dcopy_(int*, double*, int*, double*, int*);

#define f2c_dgemm dgemm_
#define f2c_sgemm sgemm_

#define f2c_daxpy daxpy_
#define f2c_saxpy saxpy_

#define f2c_sscal sscal_
#define f2c_dscal dscal_

#define f2c_scopy scopy_
#define f2c_dcopy dcopy_

#define f2c_dgemm dgemm_
#define f2c_sgemm sgemm_
#endif


};

#endif