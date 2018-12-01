/*! \file inlineMath.h
\brief Header for inlined math functions.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
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
// Return cube of number
//###################################################

template<typename T>
__host__ __device__ inline T Cb(const T a) {
  return a*a*a;
}

//###################################################
// Return sign of number (Sign(0) = 1)
//###################################################
template<typename T>
__host__ __device__ inline int Sign(const T a) {
  return a >= (T) 0 ? 1 : -1;
}

}  // namespace astrix

#endif
