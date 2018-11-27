// -*-c++-*-
/*! \file compact.cu
\brief Functions for compacting array

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <thrust/remove.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//###################################################
// Remove values from array
//###################################################

template <class T>
int Array<T>::RemoveValue(T value)
{
  int newSize = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> iter;

    iter = thrust::remove(dev_ptr, dev_ptr + size, value);

    newSize = iter - dev_ptr;
  }
  if (cudaFlag == 0) {
    T *iter = thrust::remove(hostVec, hostVec + size, value);
    newSize = iter - hostVec;
  }

  return newSize;
}

//###################################################
// Remove values from array
//###################################################

template <class T>
int Array<T>::RemoveValue(T value, int maxIndex)
{
  int newSize = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> iter;

    iter = thrust::remove(dev_ptr, dev_ptr + maxIndex, value);

    newSize = iter - dev_ptr;
  }
  if (cudaFlag == 0) {
    int *iter = thrust::remove(hostVec, hostVec + maxIndex, value);
    newSize = iter - hostVec;
  }

  return newSize;
}

//######################################################################
//! Kernel for compacting array
//######################################################################

template<class T>
__global__ void
devCompact(int N, T *destArray, T *srcArray,
           int *keepFlag, int *keepFlagScan,
           int realSize, int nDims)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    if (keepFlag[i] == 1) {
      for (unsigned int n = 0; n < nDims; n++)
        destArray[keepFlagScan[i] + n*realSize] = srcArray[i + n*realSize];
    }
    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// Compact array
//###################################################

template <class T>
void Array<T>::Compact(int nKeep,
                       Array<int> *keepFlag,
                       Array<int> *keepFlagScan)
{
  int *pKeepFlag = keepFlag->GetPointer();
  int *pKeepFlagScan = keepFlagScan->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCompact<T>,
                                       (size_t) 0, 0);

    T *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(T)));
    gpuErrchk(cudaMemcpy(temp, deviceVec,
                         nDims*realSize*sizeof(T),
                         cudaMemcpyDeviceToDevice));

    devCompact<<<nBlocks, nThreads>>>(size, deviceVec, temp,
                                      pKeepFlag, pKeepFlagScan,
                                      realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(temp));
  }

  if (cudaFlag == 0) {
    // Temporary array
    T *temp = (T *)malloc(nDims*realSize*sizeof(T));
    memcpy(temp, hostVec, nDims*realSize*sizeof(T));

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        if (pKeepFlag[i] == 1)
          hostVec[pKeepFlagScan[i] + n*realSize] = temp[i + n*realSize];

    free(temp);
  }

  SetSize(nKeep);
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::Compact(int nKeep,
                                    Array<int> *keepFlag,
                                    Array<int> *keepFlagScan);

//###################################################

template void Array<double>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);

//###################################################

template void Array<int>::Compact(int nKeep,
                                  Array<int> *keepFlag,
                                  Array<int> *keepFlagScan);

//###################################################

template void
Array<unsigned int>::Compact(int nKeep,
                             Array<int> *keepFlag,
                             Array<int> *keepFlagScan);

//###################################################

template void Array<int2>::Compact(int nKeep,
                                   Array<int> *keepFlag,
                                   Array<int> *keepFlagScan);

template void Array<int3>::Compact(int nKeep,
                                   Array<int> *keepFlag,
                                   Array<int> *keepFlagScan);
template void Array<float2>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);
template void Array<float3>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);
template void Array<float4>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);
template void Array<double2>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);
template void Array<double3>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);
template void Array<double4>::Compact(int nKeep,
                                     Array<int> *keepFlag,
                                     Array<int> *keepFlagScan);


template int Array<int>::RemoveValue(int value);
template int Array<int>::RemoveValue(int value, int maxIndex);
template int Array<unsigned int>::RemoveValue(unsigned int value);

}  // namespace astrix
