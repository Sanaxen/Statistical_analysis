#ifndef _TRANSPOSE_HPP
#define _TRANSPOSE_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.



template <typename T>
inline void transpose_(const T* a, int am, int an, T* ret, T* verify = NULL)
{
	//transpose_standerd(a, am, an, ret);
	//return;

#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU transpose_\n");
		transpose_gpu(a, am, an, ret);
	}
	else
	{
		transpose_standerd(a, am, an, ret);
	}
#else
	transpose_standerd(a, am, an, ret);
#endif

	//verify
	if (verify)
	{
		T eps = 0.0;
		for (int i = 0; i < am*an; ++i)
		{
			eps += fabs(ret[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}
}

template <typename T>
inline void transpose_standerd(const T* a, int am, int an,  T* ret)
{
	const int mn = am*an;

#pragma omp parallel for
 	for( int i = 0; i < mn; ++i ){
 		int idx1 = i/am, idx2 = i%am;
 		ret[am*idx1+idx2] = a[an*idx2+idx1];
 	}
}

#if USE_GPU
template <typename T>
inline void transpose_gpu(const T* a, int am, int an,  T* ret)
{
#ifndef USE_FLOAT
	std::vector<float> va;
	std::vector<float> vresult(am*an);
	{
		copy_array(a, am*an, va);
	}
#else
#define va a
#define vresult ret
#endif

	concurrency::extent<2> e_a(am, an), e_c(an, am);

	// Copy in
	array_view<const float, 2> av_a(e_a, va); 
	array_view<float, 2> av_c(e_c, vresult);

    av_c.discard_data();
    parallel_for_each(av_a.extent, [=] (index<2> idx) restrict(amp,cpu) 
    {
		index<2> transpose_idx(idx[1], idx[0]);
        av_c[transpose_idx] = av_a[idx];
    });
	// explicitly about copying out data
	av_c.synchronize();


#ifndef USE_FLOAT
	const int mn = am*an;
#pragma omp parallel for
	for ( int i = 0; i < mn; ++i )
	{
		ret[i] = vresult[i];
	}
#else
#undef va
#undef vresult
#endif

}
#endif

#endif
