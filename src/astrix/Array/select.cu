// -*-c++-*-
/*! \file select.cu
\brief Functions for selecting part of Array

\section LICENSE
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

//###################################################
//
//###################################################

template<class T>
__global__ void
devSelectLargerThan(int N, T *data, T value, int *pSelectFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    int ret = 0;
    if (data[i] > value) ret = 1;
    pSelectFlag[i] = ret;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
//
//###################################################

template <class T>
template <class S>
int Array<T>::SelectLargerThan(T value, Array<S> *A)
{
  Array<int> *selectFlag = new Array<int>(1, cudaFlag, size);
  int *pSelectFlag = selectFlag->GetPointer();
  Array<int> *selectFlagScan = new Array<int>(1, cudaFlag, size);

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSelectLargerThan<T>,
                                       (size_t) 0, 0);

    devSelectLargerThan<<<nBlocks, nThreads>>>(size, deviceVec, value,
                                               pSelectFlag);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = 0; i < size; i++) {
      int ret = 0;
      if (hostVec[i] > value) ret = 1;
      pSelectFlag[i] = ret;
    }
  }

  int nSelect = selectFlag->ExclusiveScan(selectFlagScan);

  Compact(nSelect, selectFlag, selectFlagScan);
  A->Compact(nSelect, selectFlag, selectFlagScan);

  delete selectFlag;
  delete selectFlagScan;

  return nSelect;
}

//###################################################
//
//###################################################

template<class T>
__global__ void
devSelectWhereDifferent(int N, T *data, T *compareData, int *pSelectFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    int ret = 0;
    if (data[i] != compareData[i]) ret = 1;
    pSelectFlag[i] = ret;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
//
//###################################################

template <class T>
template <class S>
int Array<T>::SelectWhereDifferent(Array<T> *A, Array<S> *B)
{
  Array<int> *selectFlag = new Array<int>(1, cudaFlag, size);
  int *pSelectFlag = selectFlag->GetPointer();
  Array<int> *selectFlagScan = new Array<int>(1, cudaFlag, size);

  T *compareData = A->GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSelectWhereDifferent<T>,
                                       (size_t) 0, 0);

    devSelectWhereDifferent<<<nBlocks, nThreads>>>(size, deviceVec,
                                                   compareData, pSelectFlag);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = 0; i < size; i++) {
      int ret = 0;
      if (hostVec[i] != compareData[i]) ret = 1;
      pSelectFlag[i] = ret;
    }
  }

  int nSelect = selectFlag->ExclusiveScan(selectFlagScan);

  Compact(nSelect, selectFlag, selectFlagScan);
  B->Compact(nSelect, selectFlag, selectFlagScan);

  delete selectFlag;
  delete selectFlagScan;

  return nSelect;
}

//###################################################
// Instantiate
//###################################################

template int Array<int>::SelectLargerThan(int value, Array<double> *A);
template int Array<int>::SelectLargerThan(int value, Array<double2> *A);
template int Array<int>::SelectLargerThan(int value, Array<float> *A);
template int Array<int>::SelectLargerThan(int value, Array<float2> *A);

template int Array<int>::SelectWhereDifferent(Array<int> *A,
                                              Array<double> *B);
template int Array<int>::SelectWhereDifferent(Array<int> *A,
                                              Array<double2> *B);
template int Array<int>::SelectWhereDifferent(Array<int> *A,
                                              Array<float> *B);
template int Array<int>::SelectWhereDifferent(Array<int> *A,
                                              Array<float2> *B);

}
