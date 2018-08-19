#ifndef _MPI_UTIL_HPP
#define _MPI_UTIL_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#pragma comment(lib, "C:/Program Files/MPICH2/lib/mpi.lib")

#ifdef USE_MPI
inline MPI_Datatype mpi_util_size()
{
	if (sizeof(dnn_double) == sizeof(double)) return MPI_DOUBLE;
	if (sizeof(dnn_double) == sizeof(float)) return MPI_FLOAT;
	return MPI_BYTE;
}
#endif

#endif
