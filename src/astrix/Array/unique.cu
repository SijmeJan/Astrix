// -*-c++-*-
/*! \file unique.cu
\brief Functions for compacting array
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//
//######################################################################

template<class T>
__global__ void
devScatterUnique(int N, int *pA, int *pB, T* srcArray, int ignoreValue, T value)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x + 1;

  while (i < N) {
    if (pA[i - 1] == pA[i] && pA[i] != ignoreValue)
        srcArray[pB[i]] = value;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// At a unique entry of A, i, (ignoring ignoreValue)
// set hostVec[B[i]] = value.
//###################################################

template <class T>
void Array<T>::ScatterUnique(Array<int> *A, Array<int> *B,
                             int maxIndex, int ignoreValue, T value)
{
  int *pA = A->GetPointer();
  int *pB = B->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads, devScatterUnique<T>, (size_t) 0, 0);

    devScatterUnique<<<nBlocks, nThreads>>>
      (maxIndex, pA, pB, deviceVec, ignoreValue, value);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 1; i < maxIndex; i++)
      if (pA[i - 1] == pA[i] && pA[i] != ignoreValue)
        hostVec[pB[i]] = value;
  }

}

//###################################################

template void Array<int>::ScatterUnique(Array<int> *A, Array<int> *B,
                                        int maxIndex, int ignoreValue,
                                        int value);
template void Array<float2>::ScatterUnique(Array<int> *A, Array<int> *B,
                                           int maxIndex, int ignoreValue,
                                           float2 value);

}
