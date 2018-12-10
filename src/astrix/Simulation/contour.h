/*! \file contour.h
\brief Contour integration routines

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_CONTOUR_H
#define ASTRIX_CONTOUR_H

namespace astrix {

//! Product of two linear functions
__host__ __device__ inline real
c_int(real fa, real fb, real ga, real gb)
{
  return ((fa + fb)*(ga + gb) + fa*ga + fb*gb)/(real) 6.0;
}

//! Product of two linear functions and exp(k*x)
__host__ __device__ inline real
c_int(real fa, real fb, real ga, real gb, real xa, real xb, real k)
{
  real d = k*(xb - xa);

  real Kaa = 1.0/3.0 + d/12.0 + d*d/60.0 + d*d*d/360.0 + d*d*d*d/2520.0;
  real Kab = 1.0/6.0 + d/12.0 + d*d/40.0 + d*d*d/180.0 + d*d*d*d/1008.0;
  real Kbb = 1.0/3.0 + d/4.00 + d*d/10.0 + d*d*d/36.00 + d*d*d*d/168.00;

  return (fa*ga*Kaa + (fa*gb + fb*ga)*Kab + fb*gb*Kbb)*exp(k*xa);
}

}

#endif  // ASTRIX_CONTOUR_H
