#ifndef _MATH_UTIL_HPP
#define _MATH_UTIL_HPP

	template <class T>
	inline T clamp(T x, T min, T max)
	{
		if (x < min)
			return min;
		if (max < x)
			return max;
		return x;
	}

#endif
