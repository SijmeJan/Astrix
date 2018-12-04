// -*-c++-*-
/*! \file value.cu
\brief Functions for setting all elements to single value

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
//! Kernel: Set all array elements to specific value
//######################################################################

template<class T>
__global__ void
devSetToValue(int N, T *array, T value,
              unsigned int startIndex, unsigned int endIndex,
              int realSize, int nDims)
{
  for (int n = 0; n < nDims; n++) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + startIndex;

    while (i < endIndex) {
      array[i + n*realSize] = value;
      i += gridDim.x*blockDim.x;
    }
  }
}

//##########################################################
// Set array elements to specific value from certain offset
//##########################################################

template <class T>
void Array<T>::SetToValue(T value, int startIndex, int endIndex)
{
  unsigned int si = (unsigned int) startIndex;
  unsigned int ei = (unsigned int) endIndex;

  if (startIndex == -1) si = 0;
  if (endIndex == -1) ei = size;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetToValue<T>,
                                       (size_t) 0, 0);

    devSetToValue<<<nBlocks, nThreads>>>(size, deviceVec, value,
                                         si, ei, realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = si; i < ei; i++)
        hostVec[i + n*realSize] = value;
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::SetToValue(float value,
                                       int startIndex,
                                       int endIndex);

//###################################################

template void Array<double>::SetToValue(double value,
                                        int startIndex,
                                        int endIndex);

//###################################################

template void Array<int>::SetToValue(int value,
                                     int startIndex,
                                     int endIndex);

//###################################################

template void Array<unsigned int>::SetToValue(unsigned int value,
                                              int startIndex,
                                              int endIndex);
}  // namespace astrix
