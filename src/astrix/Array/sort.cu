// -*-c++-*-
/*! \file sort.cu
\brief Functions for sorting Array
*/
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

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

}
