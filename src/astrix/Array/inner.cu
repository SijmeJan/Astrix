// -*-c++-*-
/*! \file inner.cu
\brief Functions for inner product

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
#include <thrust/inner_product.h>
#include <iostream>

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

}  // namespace astrix
