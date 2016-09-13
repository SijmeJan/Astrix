/*! \file inlineMath.h 
\brief Header for inlined math functions.
*/
#ifndef ASTRIX_MATH
#define ASTRIX_MATH

namespace astrix {

//###################################################
// Return square of number
//###################################################

template<typename T> 
__host__ __device__ inline T Sq(const T a) {
  return a*a;
}

//###################################################
// Return sign of number (Sign(0) = 1)
//###################################################
template<typename T> 
__host__ __device__ inline int Sign(const T a) {
  return a >= (T) 0 ? 1 : -1;
}

}

#endif
