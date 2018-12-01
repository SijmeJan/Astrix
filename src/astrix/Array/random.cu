// -*-c++-*-
/*! \file random.cu
\brief Functions setting entries to random values

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <cstdlib>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//###################################################
// Fill array with random numbers
//###################################################

template <class T>
void Array<T>::SetToRandom()
{
  // Seed random generator
  srand(3);

  T *temp = new T[size];
  for (unsigned int i = 0; i < size; i++) temp[i] = rand();

  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(deviceVec, temp, size*sizeof(T),
                         cudaMemcpyHostToDevice));
  } else {
    memcpy(hostVec, temp, size*sizeof(T));
  }

  delete[] temp;
}

//###################################################
// Instantiate
//###################################################

template void Array<unsigned int>::SetToRandom();

}  // namespace astrix
