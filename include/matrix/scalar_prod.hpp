#ifndef _SCALAR_PROD_HPP
#define _SCALAR_PROD_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.


template <typename T>
inline void scara_prod_( T* a, int am, int an, const T c, T* verify = NULL)
{
	//scara_prod_standerd(a, am, an, c);
	//return;

#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU2 || an > MIN_SIZE_APPLYING_GPGPU2)
	{
		//printf("GPU scara_prod_\n");
		scara_prod_gpu(a, am, an, c);
	}
	else
	{
		scara_prod_standerd(a, am, an, c);
	}
#else
	scara_prod_standerd(a, am, an, c);
#endif

	//verify
	if (verify)
	{
		T eps = 0.0;
		for (int i = 0; i < am*an; ++i)
		{
			eps += fabs(a[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}
}

template <typename T>
inline void scara_prod_standerd( T* a, int am, int an, const T c)
{
	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i) a[i] = c*a[i];

}

#if USE_GPU
template <typename T>
inline void scara_prod_gpu( T* a, int am, int an, const T c)
{
	const float cc = static_cast<const float>(c);

#ifndef USE_FLOAT
	std::vector<float> va;
	{
		copy_array(a, am*an, va);
	}
#else
#define va a
#endif

	concurrency::extent<2> e_a(am, an);

	// Copy in
	array_view<float, 2> av_a(e_a, va);

	parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		av_a[idx] *= cc;
	});
	// explicitly about copying out data
	av_a.synchronize();


#ifndef USE_FLOAT
	const int mn = am*an;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i)
	{
		a[i] = va[i];
	}
#else
#undef va
#endif

}
#endif


#endif
