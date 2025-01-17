#ifndef __MATRIX_CONFIG_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.
//Engineering and Scientific Subroutine Library

#define __MATRIX_CONFIG_H

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

//#define USE_FLOAT

#define USE_LAPACK
#define USE_BLAS
//#define USE_MKL
//#define USE_CUBLAS
//#define USE_EIGEN

//#define USE_GPU	1	//use C++AMP
#define USE_CDFLIB
#define USE_GNUPLOT
#define USE_GRAPHVIZ_DOT

#define SVDcmp	svd_decomposition
//#define SVDcmp	svd_decomposition_nr
#pragma warning( disable : 4819 ) 
#pragma warning( disable : 4996 ) 
#pragma warning( disable : 4267 ) 
#pragma warning( disable : 4101 ) 
#pragma warning( disable : 4477 ) 

#endif
