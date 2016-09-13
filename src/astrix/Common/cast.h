/*! \file cast.h
    \brief Header file for unified casting.
*/
#ifndef CAST_H
#define CAST_H

namespace astrix {

//######################################################################
//! Cast int to real
//######################################################################

__host__ __device__ inline real int2real(int a)
{
#ifdef __CUDA_ARCH__
#if USE_DOUBLE==1
  return __int2double_rn(a);
#else
  return __int2float_rn(a);
#endif
#else
  return (real) a;
#endif
}

}

#endif
