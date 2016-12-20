// -*-c++-*-
/*! \file inner.cu
\brief Functions for inner product
*/
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {
  
//###################################################
// Calculate inner product with A
//###################################################

template <class T>
T Array<T>::InnerProduct(Array<T> *A)
{
  T result = 0;

  T *pA = A->GetPointer();
  
  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    thrust::device_ptr<T> dev_ptrA(&(pA[0]));

    result = thrust::inner_product(dev_ptrA, dev_ptrA + size, dev_ptr, (T) 0.0);
  }
  if (cudaFlag == 0) {
    result = thrust::inner_product(pA, pA + size, hostVec, (T) 0.0);
  }

  return result;
}

//###################################################
// Instantiate
//###################################################

template float Array<float>::InnerProduct(Array<float> *A);

//###################################################

template double Array<double>::InnerProduct(Array<double> *A);

}
