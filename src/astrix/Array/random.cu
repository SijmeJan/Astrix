// -*-c++-*-
/*! \file random.cu
\brief Functions setting entries to random values
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

}
