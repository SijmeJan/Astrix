// -*-c++-*-
/*! \file sort.cu
\brief Functions for sorting Array

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//############################################################
//! Structure holding sorting operator
//############################################################

struct subSortCompare
{
  //! Sort operator; use first element, but if equal, use second
  template <typename Tuple0, typename Tuple1>
  __device__ __host__ bool operator()(const Tuple0 &t0, const Tuple1 &t1)
  {
    int a1 = thrust::get<0>(t0);
    int a2 = thrust::get<0>(t1);
    int b1 = thrust::get<1>(t0);
    int b2 = thrust::get<1>(t1);

    if (a1 == a2) return b1 <= b2;

    return a1 < a2;
  }
};

//########################################################################
//! Structure holding sorting operator
//########################################################################

struct subSubSortCompare
{
  //! Sort operator; use first element, but if equal, use second, then third
  template <typename Tuple0, typename Tuple1>
  __device__ __host__ bool operator()(const Tuple0 &t0, const Tuple1 &t1)
  {
    int a1 = thrust::get<0>(t0);
    int a2 = thrust::get<0>(t1);
    int b1 = thrust::get<1>(t0);
    int b2 = thrust::get<1>(t1);
    int c1 = thrust::get<2>(t0);
    int c2 = thrust::get<2>(t1);

    if (a1 == a2) {
      if (b1 == b2) return c1 <= c2;
      return b1 < b2;
    }
    return a1 < a2;
  }
};

//###################################################
// Sort array, producing indexing array
//###################################################

template <class T>
template <class S>
void Array<T>::SortByKey(Array<S> *indexArray)
{
  S *index = indexArray->GetPointer();

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<S> dev_ptr_index(index);
    thrust::sort_by_key(dev_ptr, dev_ptr + size, dev_ptr_index);
  }
  if (cudaFlag == 0) {
    thrust::sort_by_key(hostVec, &(hostVec[size]), index);
  }
}

//##########################################################
// Sort first N elements of array, producing indexing array
//##########################################################

template <class T>
template <class S>
void Array<T>::SortByKey(Array<S> *indexArray, unsigned int N)
{
  S *index = indexArray->GetPointer();

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<S> dev_ptr_index(index);
    thrust::sort_by_key(dev_ptr, dev_ptr + N, dev_ptr_index);
  }
  if (cudaFlag == 0) {
    thrust::sort_by_key(hostVec, &(hostVec[N]), index);
  }
}

//###################################################
// Sort array, in case of equal use array B
//###################################################

template <class T>
void Array<T>::Sort(Array<T> *arrayB)
{
  T *B = arrayB->GetPointer();

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> dev_ptr_B(B);

    thrust::sort
      (thrust::make_zip_iterator(thrust::make_tuple(dev_ptr, dev_ptr_B)),
       thrust::make_zip_iterator(thrust::make_tuple(dev_ptr + size,
                                                    dev_ptr_B + size)),
       subSortCompare());
  }
  if (cudaFlag == 0) {
    thrust::sort
      (thrust::make_zip_iterator(thrust::make_tuple(hostVec, B)),
       thrust::make_zip_iterator(thrust::make_tuple(&(hostVec[size]),
                                                    &(B[size]))),
       subSortCompare());
  }
}

//##########################################################################

template void Array<float>::SortByKey(Array<unsigned int> *indexArray);

//##########################################################################

template void Array<double>::SortByKey(Array<unsigned int> *indexArray);

//##########################################################################

template void Array<int>::Sort(Array<int> *arrayB);
template void Array<int>::SortByKey(Array<unsigned int> *indexArray);
template void Array<int>::SortByKey(Array<unsigned int> *indexArray,
                                    unsigned int N);
template void Array<int>::SortByKey(Array<int> *indexArray, unsigned int N);

//##########################################################################

template void Array<unsigned int>::SortByKey(Array<unsigned int> *indexArray);
template void Array<unsigned int>::SortByKey(Array<unsigned int> *indexArray,
                                             unsigned int N);

template void Array<unsigned int>::SortByKey(Array<int> *indexArray);

//##########################################################################

template void Array<unsigned int>::SortByKey(Array<float2> *indexArray);
template void Array<unsigned int>::SortByKey(Array<double2> *indexArray);
template void Array<float>::SortByKey(Array<float2> *indexArray);
template void Array<double>::SortByKey(Array<double2> *indexArray);

}  // namespace astrix
