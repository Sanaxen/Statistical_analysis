x64
Row-major method


Project setting
Additional Include Files

#define For USE_CUBLAS
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1\common\inc

Duplicate definition prevention
* #define STB_IMAGE_IMPLEMENTATION
* #define STB_IMAGE_WRITE_IMPLEMENTATION
* #define _cublas_Init_def extern

** matrix_config.h  

|options|description|default|additional requirements to use|
|-----|-----|----|----|
|USE_LAPACK|Use [CLAPACK](http://www.netlib.org/clapack/) |ON|[CLAPACK, version 3.2.1 CMAKE package](http://www.netlib.org/clapack/clapack-3.2.1-CMAKE.tgz)|
|USE_BLAS|Use [CLAPACK](http://www.netlib.org/clapack/) |ON|[CLAPACK, version 3.2.1 CMAKE package](http://www.netlib.org/clapack/clapack-3.2.1-CMAKE.tgz)|
|USE_MKL|Use [Intel@ Math Kernel Library (Intel@ MKL)](https://software.intel.com/en-us/mkl)|OFF|https://software.intel.com/en-us/mkl|
|USE_CUBLAS|Use [NVIDIA cuBLAS library](https://developer.nvidia.com/cublas) |OFF|https://developer.nvidia.com/cublas|

