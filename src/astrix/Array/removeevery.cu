// -*-c++-*-
/*! \file removeevery.cu
\brief Functions for removing every nth entry

\section LICENSE
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
//! Fill array of flags whether to keep entry
//######################################################################

__global__ void
devFillKeepFlag(int N, int *pKeepFlag, int start, int step)
{
  int i = (blockIdx.x*blockDim.x + threadIdx.x)*step + start;

  // Remove every start + step*i
  while (i < N) {
    pKeepFlag[i] = 0;
    i += gridDim.x*blockDim.x*step;
  }
}

//#####################################################################
// Remove every start + step*i
//#####################################################################

template <class T>
int Array<T>::RemoveEvery(int start, int step)
{
  Array<int> *keepFlag = new Array<int>(1, cudaFlag, size);
  Array<int> *keepFlagScan = new Array<int>(1, cudaFlag, size);

  keepFlag->SetToValue(1);

  int *pKeepFlag = keepFlag->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillKeepFlag,
                                       (size_t) 0, 0);

    devFillKeepFlag<<<nBlocks, nThreads>>>(size, pKeepFlag, start, step);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (unsigned int i = start; i < size; i += step)
      pKeepFlag[i] = 0;
  }

  int newSize = keepFlag->ExclusiveScan(keepFlagScan);
  Compact(newSize, keepFlag, keepFlagScan);

  delete keepFlag;
  delete keepFlagScan;

  return newSize;
}

//############################################################################
// Remove every start + step*i, reverse index A
//############################################################################

template <class T>
template <class S>
int Array<T>::RemoveEvery(int start, int step, Array<S> *A)
{
  Array<int> *keepFlag = new Array<int>(1, cudaFlag, size);
  Array<int> *keepFlagScan = new Array<int>(1, cudaFlag, size);

  keepFlag->SetToValue(1);

  int *pKeepFlag = keepFlag->GetPointer();
  int *pKeepFlagScan = keepFlagScan->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillKeepFlag,
                                       (size_t) 0, 0);

    devFillKeepFlag<<<nBlocks, nThreads>>>(size, pKeepFlag, start, step);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    for (unsigned int i = start; i < size; i += step)
      pKeepFlag[i] = 0;
  }

  int newSize = keepFlag->ExclusiveScan(keepFlagScan);
  Compact(newSize, keepFlag, keepFlagScan);
  A->InverseReindex(pKeepFlagScan);

  delete keepFlag;
  delete keepFlagScan;

  return newSize;
}

//############################################################################
// Remove elements from start to end
//############################################################################

template <class T>
void Array<T>::Remove(int start, int end)
{
  int sizeNew = size - (end - start);

  // New physical size
  unsigned int realSizeNew =
    ((sizeNew + dynArrayStep)/dynArrayStep)*dynArrayStep;

  T *temp;

  if (cudaFlag == 1) {
    // Manual realloc on device
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         realSizeNew*sizeof(T)));

    // Copy 0 to start to new array
    gpuErrchk(cudaMemcpy(&(temp[0]), &(deviceVec[0]),
                         start*sizeof(T),
                         cudaMemcpyDeviceToDevice));
    // Copy end to size to new array
    gpuErrchk(cudaMemcpy(&(temp[start]), &(deviceVec[end]),
                         (size - end)*sizeof(T),
                         cudaMemcpyDeviceToDevice));

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;

    int alloc = ((int)realSizeNew-(int)realSize)*(int)sizeof(T);
    if (alloc > 0) memAllocatedDevice += alloc;
    if (alloc < 0) memAllocatedDevice -= std::abs(alloc);
  }

  if (cudaFlag == 0) {
    temp = (T *)malloc(realSizeNew*sizeof(T));

    // Copy 0 to start to new array
    memcpy(&(temp[0]), &(hostVec[0]), start*sizeof(T));

    // Copy end to size to new array
    memcpy(&(temp[start]), &(hostVec[end]), (size - end)*sizeof(T));


    free(hostVec);
    hostVec = temp;

    int alloc = ((int)realSizeNew-(int)realSize)*(int)sizeof(T);
    if (alloc > 0) memAllocatedHost += alloc;
    if (alloc < 0) memAllocatedHost -= std::abs(alloc);
  }


  realSize = realSizeNew;
  size = sizeNew;
}

//###################################################
// Instantiate
//###################################################

template int Array<float>::RemoveEvery(int start, int step);
template int Array<float>::RemoveEvery(int start, int step, Array<int> *A);
template int Array<float>::RemoveEvery(int start, int step, Array<int3> *A);

//###################################################

template int Array<double>::RemoveEvery(int start, int step);
template int Array<double>::RemoveEvery(int start, int step, Array<int> *A);
template int Array<double>::RemoveEvery(int start, int step, Array<int3> *A);

//###################################################

template int Array<int>::RemoveEvery(int start, int step);
template int Array<int>::RemoveEvery(int start, int step, Array<int> *A);
template int Array<int>::RemoveEvery(int start, int step, Array<int3> *A);

//###################################################

template int Array<unsigned int>::RemoveEvery(int start, int step);
template int Array<unsigned int>::RemoveEvery(int start, int step,
                                              Array<int> *A);

//###################################################

template int Array<int2>::RemoveEvery(int start, int step, Array<int3> *A);
template int Array<float2>::RemoveEvery(int start, int step, Array<int3> *A);
template int Array<double2>::RemoveEvery(int start, int step, Array<int3> *A);

template void Array<float2>::Remove(int start, int end);
template void Array<double2>::Remove(int start, int end);

}  // namespace astrix
