// -*-c++-*-
/*! \file lincomb.cu
\brief Functions for taking linear combinations
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//
//######################################################################

template<class T>
__global__ void
devLinComb1(int N, int nDims, int realSize, T *pA,
           T a1, T *pA1)
{
  for (int n = 0; n < nDims; n++) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      pA[i + n*realSize] =
        a1*pA1[i + n*realSize];

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//
//######################################################################

template<class T>
__global__ void
devLinComb2(int N, int nDims, int realSize, T *pA,
            T a1, T *pA1,
            T a2, T *pA2)
{
  for (int n = 0; n < nDims; n++) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      pA[i + n*realSize] =
        a1*pA1[i + n*realSize] +
        a2*pA1[i + n*realSize];

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//
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

//######################################################################
//
//######################################################################

template<class T>
__global__ void
devLinComb4(int N, int nDims, int realSize, T *pA,
            T a1, T *pA1,
            T a2, T *pA2,
            T a3, T *pA3,
            T a4, T *pA4)
{
  for (int n = 0; n < nDims; n++) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      pA[i + n*realSize] =
        a1*pA1[i + n*realSize] +
        a2*pA2[i + n*realSize] +
        a3*pA3[i + n*realSize] +
        a4*pA4[i + n*realSize];

      i += gridDim.x*blockDim.x;
    }
  }
}

//##########################################
//
//##########################################

template <class T>
void Array<T>::LinComb(T a1, Array<T> *A1)
{
  T *pA1 = A1->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devLinComb1<T>,
                                       (size_t) 0, 0);

    devLinComb1<<<nBlocks, nThreads>>>(size, nDims, realSize, deviceVec,
                                       a1, pA1);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    int realSizeA1 = A1->GetRealSize();

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        hostVec[i + n*realSize] =
          a1*pA1[i + n*realSizeA1];
  }
}

//##########################################
//
//##########################################

template <class T>
void Array<T>::LinComb(T a1, Array<T> *A1,
                       T a2, Array<T> *A2)
{
  T *pA1 = A1->GetPointer();
  T *pA2 = A2->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devLinComb2<T>,
                                       (size_t) 0, 0);

    devLinComb2<<<nBlocks, nThreads>>>(size, nDims, realSize, deviceVec,
                                       a1, pA1,
                                       a2, pA2);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    int realSizeA1 = A1->GetRealSize();
    int realSizeA2 = A2->GetRealSize();

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        hostVec[i + n*realSize] =
          a1*pA1[i + n*realSizeA1] +
          a2*pA2[i + n*realSizeA2];
  }
}

//##########################################
//
//##########################################

template <class T>
void Array<T>::LinComb(T a1, Array<T> *A1,
                       T a2, Array<T> *A2,
                       T a3, Array<T> *A3)
{
  T *pA1 = A1->GetPointer();
  T *pA2 = A2->GetPointer();
  T *pA3 = A3->GetPointer();

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

template void Array<float>::LinComb(float a1, Array<float> *A1);
template void Array<float>::LinComb(float a1, Array<float> *A1,
                                    float a2, Array<float> *A2);
template void Array<float>::LinComb(float a1, Array<float> *A1,
                                    float a2, Array<float> *A2,
                                    float a3, Array<float> *A3);

//###################################################

template void Array<double>::LinComb(double a1, Array<double> *A1);
template void Array<double>::LinComb(double a1, Array<double> *A1,
                                     double a2, Array<double> *A2);
template void Array<double>::LinComb(double a1, Array<double> *A1,
                                     double a2, Array<double> *A2,
                                     double a3, Array<double> *A3);

}
