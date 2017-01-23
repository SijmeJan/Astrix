// -*-c++-*-
/*! \file predicates.cu
\brief Functions for Predicates class


\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cmath>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./predicates.h"
#include "../../Device/device.h"

namespace astrix {

real* Predicates::GetParamPointer(int cudaFlag) const
{
  if (cudaFlag == 1) return param->GetDevicePointer();
  return param->GetHostPointer();
}

//! Fill parameter vector for Predicates
/*! Compute various parameters related to roundoff errors needed by the Predicates class. For example, what is the smallest number that can be added to one without causing roundoff?
\param *pResult Pointer to array to store result in. Must be able to hold 15 real's.
*/
__host__ __device__
void InitPredicates(real *pResult)
{
  real check, lastcheck;
  int every_other;

  every_other = 1;
  real epsilon = 1.0;
  real splitter = 1.0;
  check = 1.0;
  // Repeatedly divide `epsilon' by two until it is too small to add to
  //   one without causing roundoff.  (Also check if the sum is equal to
  //   the previous sum, for machines that round up instead of using exact
  //   rounding.  Not that this library will work on such machines anyway.
  do {
    lastcheck = check;
    epsilon *= (real) 0.5;
    if (every_other) {
      splitter *= (real) 2.0;
    }
    every_other = !every_other;
    check = (real) 1.0 + epsilon;
  } while ((check != (real) 1.0) && (check != lastcheck));
  splitter += (real) 1.0;

  real resulterrbound = ((real)3.0 + (real)8.0*epsilon)*epsilon;
  real ccwerrboundA = ((real)3.0 + (real)16.0*epsilon)*epsilon;
  real ccwerrboundB = ((real)2.0 + (real)12.0*epsilon)*epsilon;
  real ccwerrboundC = ((real)9.0 + (real)64.0*epsilon)*epsilon*epsilon;
  real o3derrboundA = ((real)7.0 + (real)56.0*epsilon)*epsilon;
  real o3derrboundB = ((real)3.0 + (real)28.0*epsilon)*epsilon;
  real o3derrboundC = ((real)26.0 + (real)288.0*epsilon)*epsilon*epsilon;
  real iccerrboundA = ((real)10.0 + (real)96.0*epsilon)*epsilon;
  real iccerrboundB = ((real)4.0 + (real)48.0*epsilon)*epsilon;
  real iccerrboundC = ((real)44.0 + (real)576.0*epsilon)*epsilon*epsilon;
  real isperrboundA = ((real)16.0 + (real)224.0*epsilon)*epsilon;
  real isperrboundB = ((real)5.0 + (real)72.0*epsilon)*epsilon;
  real isperrboundC = ((real)71.0 + (real)1408.0*epsilon)*epsilon*epsilon;

  pResult[0] = splitter;
  pResult[1] = epsilon;
  pResult[2] = resulterrbound;
  pResult[3] = ccwerrboundA;
  pResult[4] = ccwerrboundB;
  pResult[5] = ccwerrboundC;
  pResult[6] = o3derrboundA;
  pResult[7] = o3derrboundB;
  pResult[8] = o3derrboundC;
  pResult[9] = iccerrboundA;
  pResult[10] = iccerrboundB;
  pResult[11] = iccerrboundC;
  pResult[12] = isperrboundA;
  pResult[13] = isperrboundB;
  pResult[14] = isperrboundC;
}

//######################################################################
//! Kernel computing parameters for Predicates
//######################################################################

__global__ void
devInitPredicates(real *pResult)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i == 0)
    InitPredicates(pResult);
}

//#########################################################################
// Constructor for predicates
//#########################################################################

Predicates::Predicates(Device *device)
{
  // Create param array on host
  param = new Array<real>(1, 0, (unsigned int) 15);

  // Calculate parameters on host
  real *pParamHost = param->GetHostPointer();
  InitPredicates(pParamHost);

  // Also calculate them on CUDA device (whether actually using CUDA or not)
  if (device->GetDeviceCount() > 0) {
    param->CopyToDevice();
    real *pParamDevice = param->GetDevicePointer();
    devInitPredicates<<<1, 1>>>(pParamDevice);
  }
}

//#########################################################################
// Destructor for predicates
//#########################################################################

Predicates::~Predicates()
{
  delete param;
}

}  // namespace astrix
