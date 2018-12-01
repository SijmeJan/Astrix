// -*-c++-*-
/*! \file intrinsic.cu
\brief Convert multi-dimensional arrays to CUDA intrinsic types

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//! Kernel: set
//######################################################################

template<class T, class S>
__global__ void
devMakeIntrinsic2D(int N, T *array, int realSize, S *res)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    res[i].x = array[i + 0*realSize];
    res[i].y = array[i + 1*realSize];

    i += gridDim.x*blockDim.x;
  }
}

//########################################################
// Create float2/double2 array from 2D float/double array
//########################################################

template <class T>
template <class S>
void Array<T>::MakeIntrinsic2D(Array<S> *result)
{
  result->SetSize(size);
  S *res = result->GetPointer(0);

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devMakeIntrinsic2D<T, S>,
                                       (size_t) 0, 0);

    devMakeIntrinsic2D<<<nBlocks, nThreads>>>(size, deviceVec,
                                              realSize, res);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (unsigned int i = 0; i < size; i++) {
      res[i].x = hostVec[i + 0*realSize];
      res[i].y = hostVec[i + 1*realSize];
    }
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::MakeIntrinsic2D(Array<float2> *result);
template void Array<double>::MakeIntrinsic2D(Array<double2> *result);

}  // namespace astrix
