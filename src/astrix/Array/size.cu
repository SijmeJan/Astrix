// -*-c++-*-
/*! \file size.cu
\brief Functions for setting size of Array

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
#include "../Common/nvtxEvent.h"

namespace astrix {

//###################################################
// Return size of array
//###################################################

template <class T>
unsigned int Array<T>::GetSize() const
{
  return size;
}

//###################################################
// Return real size of array
//###################################################

template <class T>
unsigned int Array<T>::GetRealSize() const
{
  return realSize;
}

//###################################################
// Return dimension of array
//###################################################

template <class T>
unsigned int Array<T>::GetDimension() const
{
  return nDims;
}

//###################################################
// Set size of array on host
//###################################################

template <class T>
void Array<T>::SetSizeHost(unsigned int _size)
{
  size = _size;

  // New physical size
  unsigned int realSizeNew = ((size + dynArrayStep)/dynArrayStep)*dynArrayStep;

  // Adjust physical size if not big enough or at least two times too big
  if (realSize < realSizeNew || 2*realSizeNew < realSize) {
    // Need to allocate more memory
    if (realSize < realSizeNew) {
      // Reallocate memory
      hostVec = (T *)realloc(hostVec, nDims*realSizeNew*sizeof(T));

      // Shift data to right
      for (int n = nDims - 1; n > 0; n--)
        for (int i = realSize - 1; i >= 0; i--)
          hostVec[n*realSizeNew + i] = hostVec[n*realSize + i];
    }

    if (2*realSizeNew < realSize) {
      // Shift data to left
      for (unsigned int n = 1; n < nDims; n++)
        memcpy(&(hostVec[n*realSizeNew]),
               &(hostVec[n*realSize]),
               realSizeNew*sizeof(T));

      // Reallocate memory
      hostVec = (T *)realloc(hostVec, nDims*realSizeNew*sizeof(T));
    }

    int alloc = (int)nDims*((int)realSizeNew-(int)realSize)*(int)sizeof(T);
    if (alloc > 0) memAllocatedHost += alloc;
    if (alloc < 0) memAllocatedHost -= std::abs(alloc);

    realSize = realSizeNew;
  }
}

//###################################################
// Set size of array on device
//###################################################

template <class T>
void Array<T>::SetSizeDevice(unsigned int _size)
{
  unsigned int sizeNew = _size;

  // New physical size
  unsigned int realSizeNew =
    ((sizeNew + dynArrayStep)/dynArrayStep)*dynArrayStep;

  // Adjust physical size if not big enough or at least two times too big
  if (realSize < realSizeNew ||
      2*realSizeNew < realSize ||
      (nDims > 1 && realSize != realSizeNew)) {
    // Manual realloc on device
    T *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSizeNew*sizeof(T)));

    unsigned int nToCopy = size;
    if (sizeNew < size) nToCopy = sizeNew;

    for (unsigned int n = 0; n < nDims; n++)
      gpuErrchk(cudaMemcpy(&(temp[n*realSizeNew]), &(deviceVec[n*realSize]),
                           nToCopy*sizeof(T),
                           cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  int alloc = (int)nDims*((int)realSizeNew-(int)realSize)*(int)sizeof(T);
  if (alloc > 0) memAllocatedDevice += alloc;
  if (alloc < 0) memAllocatedDevice -= std::abs(alloc);

  realSize = realSizeNew;
  size = sizeNew;
}

//###################################################
// Set size of array, either on host or device
//###################################################

template <class T>
void Array<T>::SetSize(unsigned int _size)
{
  if (cudaFlag == 0) SetSizeHost(_size);

  if (cudaFlag == 1) SetSizeDevice(_size);
}

//###################################################
// Instantiate
//###################################################

template unsigned int Array<double>::GetSize() const;
template unsigned int Array<double>::GetRealSize() const;
template unsigned int Array<double>::GetDimension() const;
template void Array<double>::SetSize(unsigned int _size);

//###################################################

template unsigned int Array<float>::GetSize() const;
template unsigned int Array<float>::GetRealSize() const;
template unsigned int Array<float>::GetDimension() const;
template void Array<float>::SetSize(unsigned int _size);

//###################################################

template unsigned int Array<int>::GetSize() const;
template unsigned int Array<int>::GetDimension() const;
template unsigned int Array<int>::GetRealSize() const;
template void Array<int>::SetSize(unsigned int _size);

//###################################################

template unsigned int Array<unsigned int>::GetSize() const;
template unsigned int Array<unsigned int>::GetRealSize() const;
template unsigned int Array<unsigned int>::GetDimension() const;
template void Array<unsigned int>::SetSize(unsigned int _size);

//###################################################

template unsigned int Array<int2>::GetSize() const;
template void Array<int2>::SetSize(unsigned int _size);

template unsigned int Array<int3>::GetSize() const;
template void Array<int3>::SetSize(unsigned int _size);

template unsigned int Array<float2>::GetSize() const;
template void Array<float2>::SetSize(unsigned int _size);

template void Array<float3>::SetSize(unsigned int _size);
template unsigned int Array<float3>::GetRealSize() const;

template void Array<float4>::SetSize(unsigned int _size);
template unsigned int Array<float4>::GetRealSize() const;

template unsigned int Array<double2>::GetSize() const;
template void Array<double2>::SetSize(unsigned int _size);

template void Array<double3>::SetSize(unsigned int _size);
template unsigned int Array<double3>::GetRealSize() const;

template void Array<double4>::SetSize(unsigned int _size);
template unsigned int Array<double4>::GetRealSize() const;

}  // namespace astrix
