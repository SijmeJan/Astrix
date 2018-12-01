// -*-c++-*-
/*! \file add.cu
\brief Functions for adding single value to whole array

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {


//###################################################
//! Add value to range in array on device
//###################################################

template<class T>
__global__ void
devAddValue(unsigned int startIndex, unsigned int endIndex,
            T *data, T value)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x + startIndex;

  while (i < endIndex) {
    data[i] += value;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// Add value to range in array
//###################################################

template <class T>
void Array<T>::AddValue(T value, unsigned int startIndex,
                        unsigned int endIndex)
{
  T *data = GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddValue<T>,
                                       (size_t) 0, 0);

    devAddValue<<<nBlocks, nThreads>>>(startIndex, endIndex, deviceVec, value);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = startIndex; i < endIndex; i++) data[i] += value;
  }
}

//###################################################
//! Multiply on device
//###################################################

template<class T>
__global__ void
devMultValue(unsigned int startIndex, unsigned int endIndex,
            T *data, T value)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x + startIndex;

  while (i < endIndex) {
    data[i] *= value;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// Multiply range of array by value
//###################################################

template <class T>
void Array<T>::MultiplyValue(T value, unsigned int startIndex,
                             unsigned int endIndex)
{
  T *data = GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devMultValue<T>,
                                       (size_t) 0, 0);

    devMultValue<<<nBlocks, nThreads>>>(startIndex, endIndex, deviceVec, value);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = startIndex; i < endIndex; i++) data[i] *= value;
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<int>::AddValue(int value,
                                   unsigned int startIndex,
                                   unsigned int endIndex);
template void Array<unsigned int>::AddValue(unsigned int value,
                                            unsigned int startIndex,
                                            unsigned int endIndex);

template void Array<int>::MultiplyValue(int value,
                                        unsigned int startIndex,
                                        unsigned int endIndex);
template void Array<unsigned int>::MultiplyValue(unsigned int value,
                                                 unsigned int startIndex,
                                                 unsigned int endIndex);

}  // namespace astrix
