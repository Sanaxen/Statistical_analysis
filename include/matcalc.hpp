#ifndef _MATMUL_HPP
#define _MATMUL_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.

//Minimum size for applying GPGPU
#define	 MIN_SIZE_APPLYING_GPGPU	2048/9
#define	 MIN_SIZE_APPLYING_GPGPU2	2048*2
#define	 MIN_SIZE_APPLYING_GPGPU3	2048/9

#include "matrix/multiplication.hpp"
#include "matrix/hadamard.hpp"
#include "matrix/transpose.hpp"
#include "matrix/plus_eqd.hpp"
#include "matrix/plus.hpp"
#include "matrix/scalar_prod.hpp"
#include "matrix/prod.hpp"
#include "matrix/zeros.hpp"
#include "matrix/ones.hpp"

#endif
