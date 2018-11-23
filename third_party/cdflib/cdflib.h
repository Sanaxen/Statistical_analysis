/*
Author
Barry Brown, James Lovato, Kathy Russell,
Department of Biomathematics,
University of Texas,
Houston, Texas.

Copyright (c) 2018, Sanaxn
CDFLIB into a single header
*/
inline double algdiv ( double *a, double *b );
inline double alnrel ( double *a );
inline double apser ( double *a, double *b, double *x, double *eps );
inline double bcorr ( double *a0, double *b0 );
inline double beta ( double a, double b );
inline double beta_asym ( double *a, double *b, double *lambda, double *eps );
inline double beta_frac ( double *a, double *b, double *x, double *y, double *lambda,
  double *eps );
inline void beta_grat ( double *a, double *b, double *x, double *y, double *w,
  double *eps,int *ierr );
inline void beta_inc ( double *a, double *b, double *x, double *y, double *w,
  double *w1, int *ierr );
inline void beta_inc_values ( int *n_data, double *a, double *b, double *x, double *fx );
inline double beta_log ( double *a0, double *b0 );
inline double beta_pser ( double *a, double *b, double *x, double *eps );
inline double beta_rcomp ( double *a, double *b, double *x, double *y );
inline double beta_rcomp1 ( int *mu, double *a, double *b, double *x, double *y );
inline double beta_up ( double *a, double *b, double *x, double *y, int *n, double *eps );
inline void binomial_cdf_values ( int *n_data, int *a, double *b, int *x, double *fx );
inline void cdfbet ( int *which, double *p, double *q, double *x, double *y,
  double *a, double *b, int *status, double *bound );
inline void cdfbin ( int *which, double *p, double *q, double *s, double *xn,
  double *pr, double *ompr, int *status, double *bound );
inline void cdfchi ( int *which, double *p, double *q, double *x, double *df,
  int *status, double *bound );
inline void cdfchn ( int *which, double *p, double *q, double *x, double *df,
  double *pnonc, int *status, double *bound );
inline void cdff ( int *which, double *p, double *q, double *f, double *dfn,
  double *dfd, int *status, double *bound );
inline void cdffnc ( int *which, double *p, double *q, double *f, double *dfn,
  double *dfd, double *phonc, int *status, double *bound );
inline void cdfgam ( int *which, double *p, double *q, double *x, double *shape,
  double *scale, int *status, double *bound );
inline void cdfnbn ( int *which, double *p, double *q, double *s, double *xn,
  double *pr, double *ompr, int *status, double *bound );
inline void cdfnor ( int *which, double *p, double *q, double *x, double *mean,
  double *sd, int *status, double *bound );
inline void cdfpoi ( int *which, double *p, double *q, double *s, double *xlam,
  int *status, double *bound );
inline void cdft ( int *which, double *p, double *q, double *t, double *df,
  int *status, double *bound );
inline void chi_noncentral_cdf_values ( int *n_data, double *x, double *lambda, 
  int *df, double *cdf );
inline void chi_square_cdf_values ( int *n_data, int *a, double *x, double *fx );
inline void cumbet ( double *x, double *y, double *a, double *b, double *cum,
  double *ccum );
inline void cumbin ( double *s, double *xn, double *pr, double *ompr,
  double *cum, double *ccum );
inline void cumchi ( double *x, double *df, double *cum, double *ccum );
inline void cumchn ( double *x, double *df, double *pnonc, double *cum,
  double *ccum );
inline void cumf ( double *f, double *dfn, double *dfd, double *cum, double *ccum );
inline void cumfnc ( double *f, double *dfn, double *dfd, double *pnonc,
  double *cum, double *ccum );
inline void cumgam ( double *x, double *a, double *cum, double *ccum );
inline void cumnbn ( double *s, double *xn, double *pr, double *ompr,
  double *cum, double *ccum );
inline void cumnor ( double *arg, double *result, double *ccum );
inline void cumpoi ( double *s, double *xlam, double *cum, double *ccum );
inline void cumt ( double *t, double *df, double *cum, double *ccum );
inline double dbetrm ( double *a, double *b );
inline double dexpm1 ( double *x );
inline double dinvnr ( double *p, double *q );
inline void dinvr ( int *status, double *x, double *fx,
  unsigned long *qleft, unsigned long *qhi );
inline double dlanor ( double *x );
inline double dpmpar ( int *i );
inline void dstinv ( double *zsmall, double *zbig, double *zabsst,
  double *zrelst, double *zstpmu, double *zabsto, double *zrelto );
inline double dstrem ( double *z );
inline void dstzr ( double *zxlo, double *zxhi, double *zabstl, double *zreltl );
inline double dt1 ( double *p, double *q, double *df );
inline void dzror ( int *status, double *x, double *fx, double *xlo,
  double *xhi, unsigned long *qleft, unsigned long *qhi );
static void E0000 ( int IENTRY, int *status, double *x, double *fx,
  unsigned long *qleft, unsigned long *qhi, double *zabsst,
  double *zabsto, double *zbig, double *zrelst,
  double *zrelto, double *zsmall, double *zstpmu );
static void E0001 ( int IENTRY, int *status, double *x, double *fx,
  double *xlo, double *xhi, unsigned long *qleft,
  unsigned long *qhi, double *zabstl, double *zreltl,
  double *zxhi, double *zxlo );
inline void erf_values ( int *n_data, double *x, double *fx );
inline double error_f ( double *x );
inline double error_fc ( int *ind, double *x );
inline double esum ( int *mu, double *x );
inline double eval_pol ( double a[], int *n, double *x );
inline double exparg ( int *l );
inline void f_cdf_values ( int *n_data, int *a, int *b, double *x, double *fx );
inline void f_noncentral_cdf_values ( int *n_data, int *a, int *b, double *lambda, 
  double *x, double *fx );
inline double fifdint ( double a );
inline double fifdmax1 ( double a, double b );
inline double fifdmin1 ( double a, double b );
inline double fifdsign ( double mag, double sign );
long fifidint ( double a );
long fifmod ( long a, long b );
inline double fpser ( double *a, double *b, double *x, double *eps );
inline void ftnstop ( char *msg );
inline double gam1 ( double *a );
inline void gamma_inc ( double *a, double *x, double *ans, double *qans, int *ind );
inline void gamma_inc_inv ( double *a, double *x, double *x0, double *p, double *q,
  int *ierr );
inline void gamma_inc_values ( int *n_data, double *a, double *x, double *fx );
inline double gamma_ln1 ( double *a );
inline double gamma_log ( double *a );
inline void gamma_rat1 ( double *a, double *x, double *r, double *p, double *q,
  double *eps );
inline void gamma_values ( int *n_data, double *x, double *fx );
inline double gamma_x ( double *a );
inline double gsumln ( double *a, double *b );
int ipmpar ( int *i );
inline void negative_binomial_cdf_values ( int *n_data, int *f, int *s, double *p, 
  double *cdf );
inline void normal_cdf_values ( int *n_data, double *x, double *fx );
inline void poisson_cdf_values ( int *n_data, double *a, int *x, double *fx );
inline double psi ( double *xx );
inline void psi_values ( int *n_data, double *x, double *fx );
inline double rcomp ( double *a, double *x );
inline double rexp ( double *x );
inline double rlog ( double *x );
inline double rlog1 ( double *x );
inline void student_cdf_values ( int *n_data, int *a, double *x, double *fx );
inline double stvaln ( double *p );
inline void timestamp ( void );
