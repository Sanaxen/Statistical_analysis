#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

//Please add include directory according to environment.
//In the case of CUDA 9.1 it was the following directory

//C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include
#include "cublas.h"
#include <cublas_v2.h>

//C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1\common\inc
#include "helper_cuda.h"

#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\lib\\x64\\cuda.lib")
#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\lib\\x64\\cudadevrt.lib")
#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\lib\\x64\\cudart.lib")

#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\lib\\x64\\cublas.lib")
#pragma comment(lib, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.1\\lib\\x64\\cublas_device.lib")

//#define IDX2C(i,j,ld) (((j)*(ld)+(i)))

class cublas_init
{
public:
	cublasHandle_t handle;

	cublas_init()
	{
		int deviceCount;

		cudaGetDeviceCount(&deviceCount);

		int device;

		for (device = 0; device < deviceCount; ++device) {

			cudaDeviceProp deviceProp;

			cudaGetDeviceProperties(&deviceProp, device);

			printf("Device %d has compute capability %d.%d.(%s)\n", device, deviceProp.major, deviceProp.minor, deviceProp.name);

		}
		cudaSetDevice(0);
		cublasInit();
		// create the context
		int status  = cublasCreate(&handle);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cerr << "***cublasCreate failed***\n";
			throw;
		}
		std::cerr << "***cublasCreate OK!!***\n";

	}
	~cublas_init()
	{
		cublasDestroy(handle);
		cublasShutdown();
	}
};
#endif
