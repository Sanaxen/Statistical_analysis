#ifndef _ONES_HPP
#define _ONES_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.


template <typename T>
inline void ones_(T* a, int am, int an)
{
	//ones_standerd(a, am, an);
	//return;

#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU2 || an > MIN_SIZE_APPLYING_GPGPU2)
	{
		//printf("GPU ones_\n");
		ones_gpu(a, am, an);
	}
	else
	{
		ones_standerd(a, am, an);
	}
#else
	ones_standerd(a, am, an);
#endif
	bool verify = false;
	if ( verify )
	{
		T eps = 0.0;
		const int m = am, n = an;
		for ( int i = 0; i < m*n; ++i )
		{
			eps += fabs(a[i] - 1.0f);
		}
		printf("eps=%f\n", eps);
	}

}

template <typename T>
inline void ones_standerd(T* a, int am, int an)
{
	const int m = am, n = an;

	const int mn = m*n;
#pragma omp parallel for
		for (int i = 0; i < mn; ++i)
		{
			a[i] = dnn_double(1.0);
		}
}


#if USE_GPU

template <typename T>
inline void ones_gpu(T* a, int am, int an)
{
	//printf("GPU ");
	const int m = am, n = an;

#ifndef USE_FLOAT
	std::vector<float> va(m*n);
	//copy_array(a, am*an, va);
#else
#define va a
#endif

	concurrency::extent<2> e_a(am, an);

	// Copy in
	array_view<float, 2> av_a(e_a, va); 
	av_a.discard_data();

	// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
	concurrency::parallel_for_each(av_a.extent, [=](index<2> idx) restrict(amp,cpu)
	{
		av_a[idx] = 1.0f;
	});
	// explicitly about copying out data
	//av_a.synchronize();


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
