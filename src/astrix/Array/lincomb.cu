// -*-c++-*-
/*! \file lincomb.cu
\brief Functions for taking linear combinations

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//! Kernel: take linear combination of Arrays
//######################################################################

template<class T>
__global__ void
devLinComb3(int N, int nDims, int realSize, T *pA,
            T a1, T *pA1,
            T a2, T *pA2,
            T a3, T *pA3)
{
  for (int n = 0; n < nDims; n++) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      pA[i + n*realSize] =
        a1*pA1[i + n*realSize] +
        a2*pA2[i + n*realSize] +
        a3*pA3[i + n*realSize];

      i += gridDim.x*blockDim.x;
    }
  }
}

//##########################################
// Take linear combination of Arrays
//##########################################

template <class T>
void Array<T>::LinComb(T a1, Array<T> *A1,
                       T a2, Array<T> *A2,
                       T a3, Array<T> *A3)
{
  T *pA1 = A1->GetPointer();
  T *pA2 = pA1;
  T *pA3 = pA1;
  if (A2 != 0)
    pA2 = A2->GetPointer();
  if (A3 != 0)
    pA3 = A3->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devLinComb3<T>,
                                       (size_t) 0, 0);

    devLinComb3<<<nBlocks, nThreads>>>(size, nDims, realSize, deviceVec,
                                       a1, pA1,
                                       a2, pA2,
                                       a3, pA3);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    int realSizeA1 = A1->GetRealSize();
    int realSizeA2 = A2->GetRealSize();
    int realSizeA3 = A3->GetRealSize();

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        hostVec[i + n*realSize] =
          a1*pA1[i + n*realSizeA1] +
          a2*pA2[i + n*realSizeA2] +
          a3*pA3[i + n*realSizeA3];
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::LinComb(float a1, Array<float> *A1,
                                    float a2, Array<float> *A2,
                                    float a3, Array<float> *A3);

//###################################################

template void Array<double>::LinComb(double a1, Array<double> *A1,
                                     double a2, Array<double> *A2,
                                     double a3, Array<double> *A3);

}  // namespace astrix
