// -*-c++-*-
/*! \file diff.cu
\brief Functions to replace Array content by difference with other Array*/

#include "./array.h"
#include "../Common/cudaLow.h"
#include "../Common/helper_math.h"

namespace astrix {

//######################################################################
//! Kernel replacing array A content with difference with B
//######################################################################

template<class T>
__global__ void
devSetToDiff(int N, T *array, T *pA, T *pB, int realSize, int nDims,
             unsigned int rSA, unsigned int rSB)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    for (unsigned int n = 0; n < nDims; n++)
      array[i + n*realSize] = pA[i + n*rSA] - pB[i + n*rSB];
    i += gridDim.x*blockDim.x;
  }
}

//##########################################
//
//##########################################

template <class T>
void Array<T>::SetToDiff(Array<T> *A, Array<T> *B)
{
  T *pA = A->GetPointer();
  T *pB = B->GetPointer();

  // We assume size of all Arrays is equal; however, realSize may differ
  unsigned int rSA = A->GetRealSize();
  unsigned int rSB = B->GetRealSize();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetToDiff<T>,
                                       (size_t) 0, 0);

    devSetToDiff<<<nBlocks, nThreads>>>(size, deviceVec, pA, pB,
                                        realSize, nDims, rSA, rSB);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }


  if (cudaFlag == 0)
    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        hostVec[i + n*realSize] = pA[i + n*rSA] - pB[i + n*rSB];
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::SetToDiff(Array<float> *A,
                                      Array<float> *B);
template void Array<double>::SetToDiff(Array<double> *A,
                                       Array<double> *B);
template void Array<float4>::SetToDiff(Array<float4> *A,
                                       Array<float4> *B);
template void Array<double4>::SetToDiff(Array<double4> *A,
                                       Array<double4> *B);

}
