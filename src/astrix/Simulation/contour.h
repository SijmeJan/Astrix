/*! \file contour.h
\brief Contour integration routines

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_CONTOUR_H
#define ASTRIX_CONTOUR_H

namespace astrix {

// Product of two linear functions
__host__ __device__ inline real
c_int(real fa, real fb, real ga, real gb)
{
  return ((fa + fb)*(ga + gb) + fa*ga + fb*gb)/(real) 6.0;
}

}

#endif  // ASTRIX_CONTOUR_H
