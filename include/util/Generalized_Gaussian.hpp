#ifndef _Generalized_Gaussian_HPP
//Copyright (c) 2021, Sanaxn
//All rights reserved.

#define Generalized_Gaussian_HPP

//https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function
inline double Generalized_Gaussian(double beta, double rho, double mu, double x)
{
	double a = pow(beta, rho / 2.0)*pow(fabs(x - mu), rho);
	double b = 2.0*tgamma(1 + 1.0 / rho);
	double c = pow(beta, 0.5) / b;
	return c * exp(-a);
}

class gg_random
{
public:
	double rho;		//shape parameter[Œ`ó•ê”]
	double beta;	//scale parameter[Ú“x•ê”]
	double mu;
	std::uniform_real_distribution<> s;
	std::gamma_distribution<> gamma;
	std::default_random_engine engine;

	double inv_rho = 1;
	double inv_sqrt_beta = 1;
	inline gg_random() {}
	inline gg_random(double beta_, double rho_, double mu_)
	{
		beta = beta_;
		rho = rho_;
		mu = mu_;
		if (fabs(beta) < 1.0e-16)
		{
			if (beta > 0) beta = 1.0e-16;
			else beta = -1.0e-16;
		}
		if (fabs(rho) < 1.0e-16)
		{
			if (rho > 0) rho = 1.0e-16;
			else rho = -1.0e-16;
		}

		inv_rho = 1.0 / rho;
		inv_sqrt_beta = pow(beta, -0.5);
		s = std::uniform_real_distribution<>(-1.0, 1.0);
		gamma = std::gamma_distribution<>(1.0, inv_rho);
		engine = std::default_random_engine(0);
	}
	inline void seed(int s)
	{
		engine = std::default_random_engine(s);
	}

	inline double rand()
	{
		double y = gamma(engine);
		//double Y = pow(fabs(y) + 1.0e-16, 1.0 / rho);
		//double p = mu + pow(beta, -0.5)*s(engine)*Y;

		double Y = pow(fabs(y) + 1.0e-16, inv_rho);
		double p = mu + inv_sqrt_beta * s(engine)*Y;

		return p;
	}
};


#endif
