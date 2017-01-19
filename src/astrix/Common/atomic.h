/*! \file atomic.h
    \brief Header file for unified atomic operations.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ATOMIC_H
#define ATOMIC_H

//######################################################################
//! Atomic add for double
//######################################################################
/*! Manual (and therefore slow) atomic add for double precision. Computes \a x + \a y atomically, returning the old value of \a x.
\param address Address of \a x
\param val Value of \a y
*/

__device__ inline double AstrixAtomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ inline float AstrixAtomicAdd(float* address, float val)
{
  return atomicAdd(address, val);
}

__device__ inline int AstrixAtomicAdd(int* address, int val)
{
  return atomicAdd(address, val);
}

namespace astrix {

//######################################################################
// Atomic add wrapper; on host do normal add
//######################################################################

template<typename T>
__host__ __device__
T AtomicAdd(T *x, T y)
{
#ifndef __CUDA_ARCH__
  T old = *x;
  *x += y;
  return old;
#else
  return AstrixAtomicAdd(x, y);
#endif
}

//######################################################################
// Atomic max wrapper; on host do normal max
//######################################################################

template<typename T>
__host__ __device__
T AtomicMax(T *x, T y)
{
#ifndef __CUDA_ARCH__
  T old = *x;
  *x = std::max(old, y);
  return old;
#else
  return atomicMax(x, y);
#endif
}

//######################################################################
// Atomic CAS wrapper; on host do normal CAS
//######################################################################

template<typename T>
__host__ __device__
T AtomicCAS(T *x, T cmp, T y)
{
#ifndef __CUDA_ARCH__
  T old = *x;
  if (*x == cmp) *x = y;
  return old;
#else
  return atomicCAS(x, cmp, y);
#endif
}

//######################################################################
// Atomic Exchange wrapper; on host do normal exchange
//######################################################################

template<typename T>
__host__ __device__
T AtomicExch(T *x, T y)
{
#ifndef __CUDA_ARCH__
  T old = *x;
  *x = y;
  return old;
#else
  return atomicExch(x, y);
#endif
}

}

#endif
