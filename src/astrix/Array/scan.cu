// -*-c++-*-
/*! \file scan.cu
\brief Scanning functions
*/
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

  //std::cout << size << " " << deviceVec << " " << pResult[0] << std::endl;

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

}
