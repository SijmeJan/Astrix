// -*-c++-*-
/*! \file single.cu
\brief Functions for reading and manipulating individual values of Array

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
// Set single value
//###################################################

template <class T>
void Array<T>::SetSingleValue(T value, int position)
{
  T *data = GetPointer();
  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(&(data[position]), &value,
                         sizeof(T),
                         cudaMemcpyHostToDevice));
  } else {
    data[position] = value;
  }
}

//###################################################
// Set single value in dimension N
//###################################################

template <class T>
void Array<T>::SetSingleValue(T value, int position,
                              unsigned int N)
{
  T *data = GetPointer();
  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(&(data[position + N*realSize]), &value,
                         sizeof(T), cudaMemcpyHostToDevice));
  } else {
    data[position + N*realSize] = value;
  }
}

//###################################################
// Get single value
//###################################################

template <class T>
void Array<T>::GetSingleValue(T *value, int position)
{
  T *data = GetPointer();
  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(value, &(data[position]),
                         sizeof(T),
                         cudaMemcpyDeviceToHost));
  } else {
    value[0] = data[position];
  }
}

//###################################################
// Get single value from dimension N
//###################################################

template <class T>
void Array<T>::GetSingleValue(T *value, int position,
                              unsigned int N)
{
  T *data = GetPointer();
  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(value, &(data[position + N*realSize]),
                         sizeof(T),
                         cudaMemcpyDeviceToHost));
  } else {
    value[0] = data[position + N*realSize];
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<double>::SetSingleValue(double value,
                                            int position);
template void Array<double>::SetSingleValue(double value,
                                            int position,
                                            unsigned int N);
template void Array<double>::GetSingleValue(double *value,
                                            int position);
template void Array<double>::GetSingleValue(double *value,
                                            int position,
                                            unsigned int N);

//###################################################

template void Array<float>::SetSingleValue(float value,
                                           int position);
template void Array<float>::SetSingleValue(float value,
                                           int position,
                                           unsigned int N);
template void Array<float>::GetSingleValue(float *value,
                                           int position);
template void Array<float>::GetSingleValue(float *value,
                                           int position,
                                           unsigned int N);

//###################################################

template void Array<int>::SetSingleValue(int value,
                                         int position);
template void Array<int>::SetSingleValue(int value,
                                         int position,
                                         unsigned int N);
template void Array<int>::GetSingleValue(int *value,
                                         int position);
template void Array<int>::GetSingleValue(int *value,
                                         int position,
                                         unsigned int N);

//###################################################

template void Array<unsigned int>::SetSingleValue(unsigned int value,
                                                  int position);
template void Array<unsigned int>::GetSingleValue(unsigned int *value,
                                                  int position);

//###################################################

template void Array<int2>::SetSingleValue(int2 value,
                                          int position);
template void Array<int2>::SetSingleValue(int2 value,
                                          int position,
                                          unsigned int N);
template void Array<int2>::GetSingleValue(int2 *value,
                                          int position);
template void Array<int2>::GetSingleValue(int2 *value,
                                          int position,
                                          unsigned int N);

//###################################################

template void Array<int3>::SetSingleValue(int3 value,
                                          int position);
template void Array<int3>::SetSingleValue(int3 value,
                                          int position,
                                          unsigned int N);
template void Array<int3>::GetSingleValue(int3 *value,
                                          int position);
template void Array<int3>::GetSingleValue(int3 *value,
                                          int position,
                                          unsigned int N);

//###################################################

template void Array<float2>::SetSingleValue(float2 value,
                                            int position);
template void Array<float2>::SetSingleValue(float2 value,
                                            int position,
                                            unsigned int N);
template void Array<float2>::GetSingleValue(float2 *value,
                                            int position);
template void Array<float2>::GetSingleValue(float2 *value,
                                            int position,
                                            unsigned int N);

//###################################################

template void Array<double2>::SetSingleValue(double2 value,
                                            int position);
template void Array<double2>::SetSingleValue(double2 value,
                                            int position,
                                            unsigned int N);
template void Array<double2>::GetSingleValue(double2 *value,
                                            int position);
template void Array<double2>::GetSingleValue(double2 *value,
                                            int position,
                                            unsigned int N);

}
