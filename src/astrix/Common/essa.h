#ifndef ESSA_H
#define ESSA_H

namespace astrix {

// CUDA does not have a double precision version of ldexp 
__host__ __device__
inline double ldexp(double x, double N)
{
  return x*exp2(N);
}
 
// Swap a and b if a smaller
template<typename T>
__host__ __device__
void SwapIfSmaller(T& a, T& b)
{
  if (a < b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
}

// Write product of floats a*b*c*d as sum of doubles
__host__ __device__
inline void SplitProduct(float a, float b, float c, float d,
			 double& G, double& H, double& I, double& J)
{
  const double splitter = (1 << 29) + 1;
  
  double AB = (double) a*(double) b;
  double CD = (double) c*(double) d;
  
  double t = AB*splitter;
  double t_h = t - (t - AB);
  float ab_h = (float) t_h;
  float ab_l = (float) (AB - t_h);

  double u = CD*splitter;
  double u_h = u - (u - CD);
  float cd_h = (float) u_h;
  float cd_l = (float) (CD - u_h);

  G = (double) ab_h*(double) cd_h;
  H = (double) ab_h*(double) cd_l;
  I = (double) ab_l*(double) cd_h;
  J = (double) ab_l*(double) cd_l;
}
  
// Find the exact sign of the sum of N numbers
template<typename T, const int N>
__host__ __device__
real ExactSignSum(T *r)
{
  const T zero = T(0);
  const T one = T(1);
  const T two = T(2);

  while (true) {
    // Make sure largest value goes first, smallest value last
    for (int i = 0; i < N; i++) {
      SwapIfSmaller(r[i], r[N - 1]);
      SwapIfSmaller(r[0], r[i]);
    }

    // Count number of positive and negative values
    int m = 0, n = 0;
    for (int i = 0; i < N; i++) {
      if (r[i] > zero) m++;
      if (r[i] < zero) n++;
    }

    // Termination criteria
    if (m == 0 && n == 0) return (real) 0.0;
    if (n == 0) return (real) 1.0;
    if (m == 0) return (real) -1.0;
    
    int F = 0;
    T tmp = std::frexp(-r[N - 1], &F);
    if (r[0] >= ldexp(n , F)) return (real) 1.0;

    int E = 0;
    tmp = std::frexp(r[0], &E);
    if (-r[N - 1] >= ldexp(m, E)) return (real) -1.0;
    
    if (E == F) {
      r[0] = r[0] + r[N - 1];
      r[N - 1] = zero;
    }
    
    if (E > F) {
      T u = ldexp(one, F - 1);
      if (-r[N - 1] != u) u = two*u;
      r[0] = r[0] - u;
      r[N - 1] = u + r[N - 1];
    }
    
    if (F > E) {
      T v = ldexp(one, E - 1);
      if (r[0] != v) v = two*v;
      r[N - 1] = r[N - 1] + v;
      r[0] = r[0] - v;
    }
  }
}  

}

#endif
