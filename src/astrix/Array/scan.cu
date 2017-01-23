// -*-c++-*-
/*! \file scan.cu
\brief Scanning functions

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################
// Exclusive scan, returns sum
//######################################################

template <class T>
T Array<T>::ExclusiveScan(Array<T> *result)
{
  if (size == 0) return 0;

  T *pResult = result->GetPointer();
  T total = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> dev_ptr_result(pResult);

    thrust::exclusive_scan(dev_ptr, dev_ptr + size, dev_ptr_result);

    T temp1, temp2;
    GetSingleValue(&temp1, size - 1);
    result->GetSingleValue(&temp2, size - 1);
    total = temp1 + temp2;
  }
  if (cudaFlag == 0) {
    thrust::exclusive_scan(hostVec, hostVec + size, pResult);

    total = pResult[size - 1] + hostVec[size - 1];
  }

  return total;
}

//######################################################
// Exclusive scan of first N elements, returns sum
//######################################################

template <class T>
T Array<T>::ExclusiveScan(Array<T> *result, unsigned int N)
{
  if (N == 0) return 0;

  T *pResult = result->GetPointer();
  T total = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> dev_ptr_result(pResult);

    thrust::exclusive_scan(dev_ptr, dev_ptr + N, dev_ptr_result);

    T temp1, temp2;
    GetSingleValue(&temp1, N - 1);
    result->GetSingleValue(&temp2, N - 1);
    total = temp1 + temp2;
  }
  if (cudaFlag == 0) {
    thrust::exclusive_scan(hostVec, hostVec + N, pResult);

    total = pResult[N-1] + hostVec[N-1];
  }

  return total;
}

//###################################################
// Instantiate
//###################################################

template int Array<int>::ExclusiveScan(Array<int> *result);
template int Array<int>::ExclusiveScan(Array<int> *result,
                                       unsigned int N);

//###################################################

template unsigned int
Array<unsigned int>::ExclusiveScan(Array<unsigned int> *result,
                                   unsigned int N);
template unsigned int
Array<unsigned int>::ExclusiveScan(Array<unsigned int> *result);

}  // namespace astrix
