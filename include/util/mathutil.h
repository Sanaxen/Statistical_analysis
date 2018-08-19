#ifndef _MATHUTIL_H
#define _MATHUTIL_H
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define MAX_ITER 400

//Gamma function
inline double gamma(const double x, int &ier)
{
	double err, g, s, t, v, w, y;
	long k;

	ier = 0;

	if (x > 5.0) {
		v = 1.0 / x;
		s = ((((((-0.000592166437354 * v + 0.0000697281375837) * v +
			0.00078403922172) * v - 0.000229472093621) * v -
			0.00268132716049) * v + 0.00347222222222) * v +
			0.0833333333333) * v + 1.0;
		g = 2.506628274631001 * exp(-x) * pow(x, x - 0.5) * s;
	}

	else {

		err = 1.0e-20;
		w = x;
		t = 1.0;

		if (x < 1.5) {

			if (x < err) {
				k = (long)x;
				y = (double)k - x;
				if (fabs(y) < err || fabs(1.0 - y) < err)
					ier = -1;
			}

			if (ier == 0) {
				while (w < 1.5) {
					t /= w;
					w += 1.0;
				}
			}
		}

		else {
			if (w > 2.5) {
				while (w > 2.5) {
					w -= 1.0;
					t *= w;
				}
			}
		}

		w -= 2.0;
		g = (((((((0.0021385778 * w - 0.0034961289) * w +
			0.0122995771) * w - 0.00012513767) * w + 0.0740648982) * w +
			0.0815652323) * w + 0.411849671) * w + 0.422784604) * w +
			0.999999926;
		g *= t;
	}

	return g;
}

class Distribution
{
public:
	/*
	Solving the nonlinear equation (f (x) = 0) by the bisection method
		x1, x2: Initial value
		eps1: End condition 1 Åbx(k+1)-x(k)ÅbÅÉeps1
		eps2: End condition 2 Åbf(x(k))ÅbÅÉeps2
		max: maximum number of trials
		ind: Actual number of trials (Error ind < 0) 
	*/
	inline double bisection(double p, double x1, double x2, double eps1=1.0e-6, double eps2=1.0e-10, int max=100)
	{
		double f0, f1, f2, x0 = 0.0;
		int sw;

		f1 = func(p, x1);
		f2 = func(p, x2);

		if (f1*f2 > 0.0)
			status = -1;

		else {
			status = 0;
			if (f1*f2 == 0.0)
				x0 = (f1 == 0.0) ? x1 : x2;
			else {
				sw = 0;
				while (sw == 0 && status >= 0) {
					sw = 1;
					status += 1;
					x0 = 0.5 * (x1 + x2);
					f0 = func(p, x0);

					if (fabs(f0) > eps2) {
						if (status <= max) {
							if (fabs(x1 - x2) > eps1 && fabs(x1 - x2) > eps1*fabs(x2)) {
								sw = 0;
								if (f0*f1 < 0.0) {
									x2 = x0;
									f2 = f0;
								}
								else {
									x1 = x0;
									f1 = f0;
								}
							}
						}
						else
							status = -1;
					}
				}
			}
		}
		return x0;
	}

	/*
	Solving the nonlinear equation (f (x) = 0) by the Newton method
		x0: Initial value
		eps1: End condition 1 Åbx(k+1)-x(k)ÅbÅÉeps1
		eps2: End condition 2 Åbf(x(k))ÅbÅÉeps2
		max: maximum number of trials
		ind: Actual number of trials (Error ind < 0)
	*/
	inline double newton(double p, double x0, double eps1=1.0e-6, double eps2=1.0e-10, int max=100)
	{
		double g, dg, x, x1;
		int sw;

		x1 = x0;
		x = x1;
		status = 0;
		sw = 0;

		while (sw == 0 && status >= 0) {

			sw = 1;
			status += 1;
			g = func(p, x1);

			if (fabs(g) > eps2) {
				if (status <= max) {
					dg = d_func(x1);
					if (fabs(dg) > eps2) {
						x = x1 - g / dg;
						if (fabs(x - x1) > eps1 && fabs(x - x1) > eps1*fabs(x)) {
							x1 = x;
							sw = 0;
						}
					}
					else
						status = -1;
				}
				else
					status = -1;
			}
		}

		return x;
	}

public:
	int status;

	//1.0 - p - P(X > x)
	virtual double func(double p, double x) = 0;

	// d P(X = x)/ dx
	virtual double d_func(double x) = 0;

	virtual double distribution(double x, double *w) = 0;
	virtual double p_value(double p) = 0;
};


class Standard_normal_distribution:public Distribution
{
	inline double func(double p, double x)
	{
		double y;

		return 1.0 - p - distribution(x, &y);
	}
	inline double d_func(double x)
	{
		return 0.0;
	}

public:

	Standard_normal_distribution() { status = -1; }
	~Standard_normal_distribution() {}

	/*
	Calculation of standard normal distribution N (0, 1)
		w : P(X = x)
		return : P(X < x)
	*/
	inline double distribution(double x, double *w)
	{
		double pi = M_PI;
		double y, z, P;
		/*
		PDF(Probability density function)
		*/
		*w = exp(-0.5 * x * x) / sqrt(2.0*pi);
		/*
		Probability distribution function
		*/
		y = 0.70710678118654 * fabs(x);
		z = 1.0 + y * (0.0705230784 + y * (0.0422820123 +
			y * (0.0092705272 + y * (0.0001520143 + y * (0.0002765672 +
				y * 0.0000430638)))));
		P = 1.0 - pow(z, -16.0);

		if (x < 0.0)
			P = 0.5 - 0.5 * P;
		else
			P = 0.5 + 0.5 * P;

		return P;
	}

	/*
	The p% value of the standard normal distribution N (0, 1)ÅiP(X > u) = 0.01pÅj
	*/
	inline double p_value(double p)
	{
		return bisection(p, -7.0, 7.0);
	}
};

class Student_t_distribution:public Distribution
{
	int dof;      // Degree of freedom

	inline double func(double p, double x)
	{
		double y;

		return distribution(x, &y) - 1.0 + p;
	}

	inline double d_func(double x)
	{
		double y, z;

		z = distribution(x, &y);

		return y;
	}

	Standard_normal_distribution nrm;

public:

	inline Student_t_distribution(int dof_):dof(dof_)
	{
		status = -1;
	}

	/* 
	Calculation of t distributionÅiP(X = tt), P(X < tt)Åj
		dd : P(X = tt)
		df : Degree of freedom 
		return : P(X < tt)
	*/
	inline double distribution(double tt, double *dd)
	{
		int df = dof;
		status = 0;
		double pi = 4.0 * atan(1.0);
		double p, pp, sign, t2, u, x;
		int ia, i1;

		sign = (tt < 0.0) ? -1.0 : 1.0;
		if (fabs(tt) < 1.0e-10)
			tt = sign * 1.0e-10;
		t2 = tt * tt;
		x = t2 / (t2 + df);

		if (df % 2 != 0) {
			u = sqrt(x*(1.0 - x)) / pi;
			p = 1.0 - 2.0 * atan2(sqrt(1.0 - x), sqrt(x)) / pi;
			ia = 1;
		}

		else {
			u = sqrt(x) * (1.0 - x) / 2.0;
			p = sqrt(x);
			ia = 2;
		}

		if (ia != df) {
			for (i1 = ia; i1 <= df - 2; i1 += 2) {
				p += 2.0 * u / i1;
				u *= (1.0 + i1) / i1 * (1.0 - x);
			}
		}

		*dd = u / fabs(tt);
		pp = 0.5 + 0.5 * sign * p;

		return pp;
	}

	// p% value of t distributionÅiP(X > u) = 0.01pÅj
	inline double p_value(double p)
	{
		const double pi = M_PI;
		double c, df, df2, e, pis, p2, tt = 0.0, t0, x, yq;

		status = 0;
		pis = sqrt(pi);
		df = (double)dof;
		df2 = 0.5 * df;
		// Degree of freedom=1
		if (dof == 1)
			tt = tan(pi*(0.5 - p));

		else {
			// Degree of freedom=2
			if (dof == 2) {
				c = (p > 0.5) ? -1.0 : 1.0;
				p2 = (1.0 - 2.0 * p);
				p2 *= p2;
				tt = c * sqrt(2.0 * p2 / (1.0 - p2));
			}
			// Degree of freedom>2
			else {

				yq = nrm.p_value(p);
				status = nrm.status;

				if (status >= 0) {

					x = 1.0 - 1.0 / (4.0 * df);
					e = x * x - yq * yq / (2.0 * df);

					if (e > 0.5)
						t0 = yq / sqrt(e);
					else {
						x = sqrt(df) / (pis * p * df * gamma(df2, status) / gamma(df2 + 0.5, status));
						t0 = pow(x, 1.0 / df);
					}
					tt = newton(p, t0);
				}
			}
		}
		return tt;
	}
};

class F_distribution :public Distribution
{
	int dof1;      // Degree of freedom1
	int dof2;      // Degree of freedom2


	inline double func(double p, double x)
	{
		double y;

		return distribution(x, &y) - 1.0 + p;
	}

	inline double d_func(double x)
	{
		double y, z;

		z = distribution(x, &y);

		return y;
	}

	Standard_normal_distribution nrm;

public:

	inline F_distribution(int dof1_, int dof2_) :dof1(dof1_), dof2(dof2_)
	{
		status = -1;
	}

	/*
	Calculation of F distributionÅiP(X = tt), P(X < tt)Åj
		dd : P(X = tt)
		df : Degree of freedom
		return : P(X < tt)
	*/
	inline double distribution(double ff, double *dd)
	{
		const int df1 = dof1;
		const int df2 = dof2;
		const double pi = M_PI;
		double pp, u, x;
		int ia, ib, i1;

		if (ff < 1.0e-10)
			ff = 1.0e-10;

		x = ff * df1 / (ff * df1 + df2);

		if (df1 % 2 == 0) {
			if (df2 % 2 == 0) {
				u = x * (1.0 - x);
				pp = x;
				ia = 2;
				ib = 2;
			}
			else {
				u = 0.5 * x * sqrt(1.0 - x);
				pp = 1.0 - sqrt(1.0 - x);
				ia = 2;
				ib = 1;
			}
		}

		else {
			if (df2 % 2 == 0) {
				u = 0.5 * sqrt(x) * (1.0 - x);
				pp = sqrt(x);
				ia = 1;
				ib = 2;
			}
			else {
				u = sqrt(x*(1.0 - x)) / pi;
				pp = 1.0 - 2.0 * atan2(sqrt(1.0 - x), sqrt(x)) / pi;
				ia = 1;
				ib = 1;
			}
		}

		if (ia != df1) {
			for (i1 = ia; i1 <= df1 - 2; i1 += 2) {
				pp -= 2.0 * u / i1;
				u *= x * (i1 + ib) / i1;
			}
		}

		if (ib != df2) {
			for (i1 = ib; i1 <= df2 - 2; i1 += 2) {
				pp += 2.0 * u / i1;
				u *= (1.0 - x) * (i1 + df1) / i1;
			}
		}

		*dd = u / ff;

		return pp;
	}

	/*
	The p% value of the F distribution ÅiP(X > u) = 0.01pÅj
	*/
	inline double p_value(double p)
	{
		double a, a1, b, b1, df1, df2, e, ff = 0.0, f0, x, y1, y2, yq;
		int sw = 0;

		status = 0;
		while (sw >= 0) {

			df1 = 0.5 * (dof1 - sw);
			df2 = 0.5 * dof2;
			a = 2.0 / (9.0 * (dof1 - sw));
			a1 = 1.0 - a;
			b = 2.0 / (9.0 * dof2);
			b1 = 1.0 - b;

			yq = nrm.p_value(p);
			status = nrm.status;

			e = b1 * b1 - b * yq * yq;

			if (e > 0.8 || (dof1 + dof2 - sw) <= MAX_ITER)
				sw = -1;
			else {
				sw += 1;
				if ((dof1 - sw) == 0)
					sw = -2;
			}
		}

		if (sw == -2)
			status = -1;

		else {
			if (e > 0.8) {
				x = (a1 * b1 + yq * sqrt(a1*a1*b + a*e)) / e;
				f0 = pow(x, 3.0);
			}
			else {
				y1 = pow((double)dof2, df2 - 1.0);
				y2 = pow((double)dof1, df2);
				x = gamma(df1 + df2, status) / gamma(df1, status) / gamma(df2, status) * 2.0 * y1 / y2 / p;
				f0 = pow(x, 2.0 / dof2);
			}
			ff = newton(p, f0);
		}

		return ff;
	}
};



#endif
