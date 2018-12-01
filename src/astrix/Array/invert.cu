// -*-c++-*-
/*! \file invert.cu
\brief Functions for swapping 1 and 0 entries in Array

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//! Kernel: Switch elements 1 to 0, 0 to 1
//######################################################################

template<class T>
__global__ void
devInvert(int N, T *array, int realSize, int nDims)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    for (int n = 0; n < nDims; n++) {
      int ret = 0;
      if (array[i + n*realSize] == 0) ret = 1;
      array[i + n*realSize] = ret;
    }

    i += gridDim.x*blockDim.x;
  }
}

//##########################################
// Switch elements 1->0, 0->1
//##########################################

template <class T>
void Array<T>::Invert()
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInvert<T>,
                                       (size_t) 0, 0);

    devInvert<<<nBlocks, nThreads>>>(size, deviceVec,
                                     realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        int ret = 0;
        if (hostVec[i + n*realSize] == 0) ret = 1;
        hostVec[i + n*realSize] = ret;
      }
    }
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<int>::Invert();
template void Array<unsigned int>::Invert();

}  // namespace astrix
