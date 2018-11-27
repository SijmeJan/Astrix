// -*-c++-*-
/*! \file clock.cu
\brief Sort coordinates in counterclockwise order

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <fstream>
#include <sstream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//#########################################################################
//! Compute polar angle with respect to origin
//#########################################################################

template<class T, class S>
__global__ void
devComputeAngle(int N, T *pSource, const T origin, S *pAngle)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    S dx = pSource[i].x - origin.x;
    S dy = pSource[i].y - origin.y;

    pAngle[i] = atan2(dy, dx);

    i += gridDim.x*blockDim.x;
  }
}

//#####################################################################
// Sort array (float2/double2) in counterclockwise order around origin
//#####################################################################

template <class T>
template <class S>
void Array<T>::SortCounterClock(const T origin)
{
  // Compute angle
  Array<S> *angle = new Array<S>(1, cudaFlag, size);
  S *pAngle = angle->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devComputeAngle<T, S>,
                                       (size_t) 0, 0);

    devComputeAngle<<<nBlocks, nThreads>>>(size, deviceVec, origin, pAngle);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (int i = 0; i < size; i++) {
      S dx = hostVec[i].x - origin.x;
      S dy = hostVec[i].y - origin.y;

      pAngle[i] = atan2(dy, dx);
    }
  }

  // Sort vertexCoordinatesToAdd according to angle
  angle->SortByKey(this);

  delete angle;
}

//###################################################
// Instantiate
//###################################################

template void Array<float2>::SortCounterClock<float>(float2 origin);
template void Array<double2>::SortCounterClock<double>(double2 origin);

}
