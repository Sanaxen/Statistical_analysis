#ifndef _MULTIPLICATION_HPP
#define _MULTIPLICATION_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.


template <typename T>
inline void mull_(const T* a, int am, int an, const T* b, int bm, int bn, T* ret, T* verify = NULL)
{
#if USE_GPU
	if (am > MIN_SIZE_APPLYING_GPGPU || an > MIN_SIZE_APPLYING_GPGPU || bm > MIN_SIZE_APPLYING_GPGPU || bn > MIN_SIZE_APPLYING_GPGPU)
	{
		//printf("GPU mull_\n");
		int stat = -1;
		stat = mull_gpu_tiled<dnn_double, 16>(a, am, an, b, bm, bn, ret);
		if (stat == -1)
		{
			stat = mull_gpu_tiled<dnn_double, 8>(a, am, an, b, bm, bn, ret);
		}
		//if (stat == -1)
		//{
		//	stat = mull_gpu_tiled<double, 4>(a, am, an, b, bm, bn, ret);
		//}
		//if (stat == -1)
		//{
		//	stat = mull_gpu_tiled<double, 2>(a, am, an, b, bm, bn, ret);
		//}
		if (stat == -1)
		{
			//printf("GPU mull_\n");
			mull_gpu<dnn_double>(a, am, an, b, bm, bn, ret);
		}
	}
	else
	{
		mull_Unrolling(a, am, an, b, bm, bn, ret);
	}
#else
	//mull_standerd(a, am, an, b, bm, bn, ret);
	mull_Unrolling(a, am, an, b, bm, bn, ret);
#endif
	//verify
	if ( verify )
	{
		T eps = 0.0;
		const int m = am, n = bn, l = an;
		for ( int i = 0; i < m*n; i++ )
		{
			eps += fabs(ret[i] - verify[i]);
		}
		printf("eps=%f\n", eps);
	}

}

template <typename T>
inline void mull_standerd(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int m = am, n = bn, l = an;

#pragma omp parallel for
		for (int i = 0; i < m; ++i)
			for (int j = 0; j < n; ++j) {
				T sum = 0.0;
				for (int k = 0; k < l; ++k)
					sum += a[i*an + k]*b[k*bn + j];
				ret[n*i + j] = sum;
			}
}

template <typename T>
inline void mull_Unrolling(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	const int m = am, n = bn, l = an;
#pragma omp parallel
	{
		const int mn = m*n;
#pragma omp for
		for (int i = 0; i < mn; ++i)
		{
			ret[i] = 0.0;
		}

#pragma omp for
		for (int i = 0; i < m; ++i)
		{
			for (int k = 0; k < l; ++k)
			{
				const T mm = a[i*an + k];

				//ƒAƒ“ƒ[ƒŠƒ“ƒO
#if 0
				const int N = 4;
				const int NN = n / N;
				for (int j = 0; j < NN; ++j)
				{
					const int kbn = k*bn + N*j;
					const int ni = n*i + N*j;
					ret[ni] += mm*b[kbn];
					ret[ni + 1] += mm*b[kbn + 1];
					ret[ni + 2] += mm*b[kbn + 2];
					ret[ni + 3] += mm*b[kbn + 3];
				}
#endif

#if 10
				const int N = 8;
				const int NN = n/N;
				for (int j = 0; j < NN; ++j)
				{
					const int kbn = k*bn + N*j;
					const int ni = n*i + N*j;
					ret[ni] += mm*b[kbn];
					ret[ni + 1] += mm*b[kbn + 1];
					ret[ni + 2] += mm*b[kbn + 2];
					ret[ni + 3] += mm*b[kbn + 3];
					ret[ni + 4] += mm*b[kbn + 4];
					ret[ni + 5] += mm*b[kbn + 5];
					ret[ni + 6] += mm*b[kbn + 6];
					ret[ni + 7] += mm*b[kbn + 7];
				}
#endif

#if 0
				const int N = 16;
				const int NN = n / N;
				for (int j = 0; j < NN; ++j)
				{
					const int kbn = k*bn + N*j;
					const int ni = n*i + N*j;
					ret[ni] += mm*b[kbn];
					ret[ni + 1] += mm*b[kbn + 1];
					ret[ni + 2] += mm*b[kbn + 2];
					ret[ni + 3] += mm*b[kbn + 3];
					ret[ni + 4] += mm*b[kbn + 4];
					ret[ni + 5] += mm*b[kbn + 5];
					ret[ni + 6] += mm*b[kbn + 6];
					ret[ni + 7] += mm*b[kbn + 7];
					ret[ni + 8] += mm*b[kbn + 8];
					ret[ni + 9] += mm*b[kbn + 9];
					ret[ni + 10] += mm*b[kbn + 10];
					ret[ni + 11] += mm*b[kbn + 11];
					ret[ni + 12] += mm*b[kbn + 12];
					ret[ni + 13] += mm*b[kbn + 13];
					ret[ni + 14] += mm*b[kbn + 14];
					ret[ni + 15] += mm*b[kbn + 15];
				}
#endif
				for ( int j=(NN == 0)?0:n-n%N; j < n; j += 1)
				{
					ret[n*i + j] += mm*b[k*bn + j];
				}
			}
		}
	}
}

#if USE_GPU

template <typename T>
inline void mull_gpu(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	//printf("GPU ");
	const int m = am, n = bn, l = an;

#ifndef USE_FLOAT
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult(am*bn);

#pragma omp parallel
#pragma omp sections nowait
	{
#pragma omp section 
		{
			copy_array(a, am*an, va);
		}
#pragma omp section
		{
			copy_array(b, bm*bn, vb);
		}
	}
#else
#define va a
#define vb b
#define vresult ret
#endif

	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	// Copy in
	array_view<const float, 2> av_a(e_a, va); 
	array_view<const float, 2> av_b(e_b, vb); 
	array_view<float, 2> av_c(e_c, vresult);
	av_c.discard_data();

	// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
	concurrency::parallel_for_each(av_c.extent, [=](index<2> idx) restrict(amp,cpu)
		{
			float result = 0;

			for(int i = 0; i < av_a.extent[1]; ++i)
			{
				index<2> idx_a(idx[0], i);
				index<2> idx_b(i, idx[1]);

				result += av_a[idx_a] * av_b[idx_b];
			}

			av_c[idx] = result;
		});
	// explicitly about copying out data
	av_c.synchronize();


#ifndef USE_FLOAT
	const int mn = am*bn;
#pragma omp parallel for
	for ( int i = 0; i < mn; ++i )
	{
		ret[i] = vresult[i];
	}
#else
#undef va
#undef vb
#undef vresult
#endif

}

template <typename T>
inline void mull_gpu1(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	//printf("GPU ");
	const int m = am, n = bn, l = an;

#ifndef USE_FLOAT
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult(am*bn);

#pragma omp parallel
#pragma omp sections nowait
	{
#pragma omp section 
		{
			copy_array(a, am*an, va);
		}
#pragma omp section
		{
			copy_array(b, bm*bn, vb);
		}
	}
#else
#define va a
#define vb b
#define vresult ret
#endif

	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	// Copy in
	array_view<const float, 2> av_a(e_a, va);
	array_view<const float, 2> av_b(e_b, vb);
	array_view<float, 2> av_c(e_c, vresult);
	//av_c.discard_data();

	// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
	concurrency::parallel_for_each(av_c.extent, [=](index<2> idx) restrict(amp, cpu)
	{
		float result = 0;

		for (int i = 0; i < av_a.extent[1]; ++i)
		{
			index<2> idx_a(idx[0], i);
			index<2> idx_b(i, idx[1]);

			result += av_a[idx_a] * av_b[idx_b];
		}

		av_c[idx] += result;
	});
	// explicitly about copying out data
	//av_c.synchronize();


#ifndef USE_FLOAT
	const int mn = am*bn;
#pragma omp parallel for
	for (int i = 0; i < mn; ++i)
	{
		ret[i] += vresult[i];
	}
#else
#undef va
#undef vb
#undef vresult
#endif

}

template <typename T>
inline void mull_gpu2(const std::vector<T*>& a, int am, int an, const std::vector<T*>& b, int bm, int bn, std::vector<T*>& ret)
{
	//printf("GPU ");
	const int m = am, n = bn, l = an;

	const int nummat = a.size();
	const int nummat2 = b.size();
	const int nummat3 = ret.size();

	//printf("nummat %d nummat2 %d nummat3 %d\n", nummat, nummat2, nummat3);
	//fflush(stdout);

	if ((nummat == nummat2 && nummat == nummat3) || nummat == nummat3 && nummat2 == 1)
	{
		//printf("OK!!\n");
	}
	else
	{
		fprintf(stderr, "mull_gpu2 call ERROR!!\n");
		exit(-1);
	}

#ifndef USE_FLOAT
	std::vector<std::vector<float>> va(a.size());
	std::vector<std::vector<float>> vb(b.size());
	std::vector<std::vector<float>> vresult(ret.size());

	if ((nummat == nummat2 && nummat == nummat3))
	{
#pragma omp parallel for
		for (int k = 0; k < nummat; k++)
		{
			copy_array(a[k], am*an, va[k]);
			copy_array(b[k], bm*bn, vb[k]);
			vresult[k].resize(am*bn);
		}
	}
	else
	{
#pragma omp parallel for
		for (int k = 0; k < nummat; k++)
		{
			copy_array(a[k], am*an, va[k]);
			vresult[k].resize(am*bn);
		}
		copy_array(b[0], bm*bn, vb[0]);
	}
#else
#define va a
#define vb b
#define vresult ret
#endif

	std::vector<shared_ptr<array_view<const float, 2>>> av_a0(nummat);
	std::vector<shared_ptr<array_view<const float, 2>>> av_b0(nummat2);
	std::vector<shared_ptr<array_view<float, 2>>> av_c0(nummat);


	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	// Copy in
#pragma omp parallel
	{
#pragma omp for
		for (int k = 0; k < nummat; k++)
		{
			av_a0[k].reset(new array_view<const float, 2>(e_a, va[k]));

			if (nummat == nummat2 && nummat == nummat3)
			{
				av_b0[k].reset(new array_view<const float, 2>(e_b, vb[k]));
			}
			else
			{
				if (k == 0)
				{
					av_b0[0].reset(new array_view<const float, 2>(e_b, vb[0]));
				}
			}
			av_c0[k].reset(new array_view<float, 2>(e_c, vresult[k]));
			av_c0[k]->discard_data();
		}

#pragma omp for
		for (int k = 0; k < nummat; k++)
		{
			const auto& av_a = *av_a0[k];

			const int kk = (nummat == nummat2 && nummat == nummat3) ? k : 0;
			const auto& av_b = *av_b0[kk];
			auto& av_c = *av_c0[k];

			// Compute - outer 2 for loops of CPU is replaced by a parallel_for_each
			concurrency::parallel_for_each(av_c.extent, [=](index<2> idx) restrict(amp, cpu)
			{
				float result = 0;

				for (int i = 0; i < av_a.extent[1]; ++i)
				{
					index<2> idx_a(idx[0], i);
					index<2> idx_b(i, idx[1]);

					result += av_a[idx_a] * av_b[idx_b];
				}

				av_c[idx] = result;
			});
			// explicitly about copying out data
			av_c.synchronize();
		}
	}
#ifndef USE_FLOAT
	const int mn = am*bn;
#pragma omp parallel for
	for (int k = 0; k < a.size(); k++)
	{
		for (int i = 0; i < mn; ++i)
		{
			ret[k][i] = vresult[k][i];
		}
	}
#else
#undef va
#undef vb
#undef vresult
#endif
}


template <typename T, int tile_size>
inline int mull_gpu_tiled(const T* a, int am, int an, const T* b, int bm, int bn, T* ret)
{
	if (!(am%tile_size == 0 && bn%tile_size == 0 && an%tile_size == 0))
	{
		//printf("mull_gpu_tiled size error.\n");
		return -1;
	}
	//printf("mull_gpu_tiled size OK.\n");
	
	const int m = am, n = bn, l = an;
#ifndef USE_FLOAT
	std::vector<float> va;
	std::vector<float> vb;
	std::vector<float> vresult;
	{
		copy_array(a, am*an, va);
		copy_array(b, bm*bn, vb);
		vresult.resize(am*bn);
	}
#else
#define va a
#define vb b
#define vresult ret
#endif


	concurrency::extent<2> e_a(am, an), e_b(bm, bn), e_c(am, bn);

	array_view<const float, 2> av_a(e_a, va);
	array_view<const float, 2> av_b(e_b, vb);
	array_view<float, 2> av_c(e_c, vresult);

	concurrency::extent<2> compute_domain(e_c);

	parallel_for_each(compute_domain.tile<tile_size, tile_size>(), [=](tiled_index<tile_size, tile_size> tidx) restrict(amp)
	{
		float temp_c = 0;

		index<2> localIdx = tidx.local;
		index<2> globalIdx = tidx.global;

		for (int i = 0; i < an; i += tile_size)
		{
			tile_static float localB[tile_size][tile_size];
			tile_static float localA[tile_size][tile_size];

			localA[localIdx[0]][localIdx[1]] = av_a(globalIdx[0], i + localIdx[1]);
			localB[localIdx[0]][localIdx[1]] = av_b(i + localIdx[0], globalIdx[1]);

			tidx.barrier.wait();

			for (unsigned k = 0; k < tile_size; k++)
			{
				temp_c += localA[localIdx[0]][k] * localB[k][localIdx[1]];
			}

			tidx.barrier.wait();
		}

		av_c[tidx] = temp_c;
	});
	// copying out data is implicit - when array_view goes out of scope data is synchronized	//av_c.synchronize();
	av_c.synchronize();

#ifndef USE_FLOAT
	const int mn = am*bn;
	//copy_array(vresult, mn, ret);

#pragma omp parallel for
	for (int i = 0; i < mn; i++)
	{
		ret[i] = vresult[i];
	}
#else
#undef va
#undef vb
#undef vresult
#endif

	return 0;
}

#endif


#endif
