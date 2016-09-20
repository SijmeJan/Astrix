// -*-c++-*-
/*! \file equal.cu
\brief Functions setting Array equal to other Array
*/
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//###################################################
// Set all array elements equal to other Array
//###################################################

template <class T>
void Array<T>::SetEqual(const Array *B)
{
  if (cudaFlag == 1)
    gpuErrchk(cudaMemcpy(deviceVec, B->GetDevicePointer(),
			 nDims*realSize*sizeof(T),
			 cudaMemcpyDeviceToDevice));

  if (cudaFlag == 0)
    for (unsigned int i = 0; i < nDims; i++)
      memcpy(GetHostPointer(i), B->GetHostPointer(i), size*sizeof(T));
}

//##########################################################
// Set dimension N of A equal to dimension M of intrinsic B 
//##########################################################

template <class T, class S>
__global__ void 
devSetEqualComb(int N, T *destArray, S *srcArray, 
		unsigned int realSize, int n, int m)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < N) {
    if (m == 0)
      destArray[i + n*realSize] = srcArray[i].x;
    if (m == 1)
      destArray[i + n*realSize] = srcArray[i].y;

    i += gridDim.x*blockDim.x;
  }
}

//##########################################################
// Set dimension N of A equal to dimension M of intrinsic B 
//##########################################################

template <class T>
template <class S>
void Array<T>::SetEqualComb(const Array<S> *B, unsigned int N, unsigned int M)
{
  S *pB = B->GetPointer();
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devSetEqualComb<T, S>, 
				       (size_t) 0, 0);

    devSetEqualComb<<<nBlocks, nThreads>>>(size, deviceVec, pB,
						  realSize, N, M);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0) {
    if (M == 0)
      for (unsigned int i = 0; i < size; i++) hostVec[i + N*realSize] = pB[i].x;
    if (M == 1)
      for (unsigned int i = 0; i < size; i++) hostVec[i + N*realSize] = pB[i].y;
  }
}

//###################################################
// Set dimension N of A equal to dimension M of B
//###################################################

template <class T>
void Array<T>::SetEqual(const Array<T> *B, unsigned int N, unsigned int M)
{
  if (cudaFlag == 1)
    gpuErrchk(cudaMemcpy(GetDevicePointer(N), B->GetDevicePointer(M),
			 realSize*sizeof(T),
			 cudaMemcpyDeviceToDevice));

  if (cudaFlag == 0)
    memcpy(GetHostPointer(N), B->GetHostPointer(M), size*sizeof(T));
}

//###################################################
// Set all array elements equal to other Array
//###################################################

template <class T>
void Array<T>::SetEqual(const Array *B, int startPosition)
{
  if (cudaFlag == 1)
    for (unsigned int n = 0; n < nDims; n++)
      gpuErrchk(cudaMemcpy(&(deviceVec[n*realSize + startPosition]), 
			   B->GetDevicePointer(n),
			   B->GetSize()*sizeof(T),
			   cudaMemcpyDeviceToDevice));

  if (cudaFlag == 0)
    for (unsigned int n = 0; n < nDims; n++)
      memcpy(&(hostVec[n*realSize + startPosition]), 
	     B->GetHostPointer(n), 
	     B->GetSize()*sizeof(T));
}

//###################################################
// Concatenate with array
//###################################################

template <class T>
void Array<T>::Concat(Array<T> *A)
{
  int sizeA = A->GetSize();
  T *pA = A->GetPointer();

  int oldSize = size;
  SetSize(size + sizeA);
  T *data = GetPointer();

  if (cudaFlag == 1) {
    gpuErrchk(cudaMemcpy(&(data[oldSize]), pA,
			 sizeA*sizeof(T),
			 cudaMemcpyDeviceToDevice));
  } else {
    memcpy(&data[oldSize], pA, sizeA*sizeof(T));
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<double>::SetEqual(const Array *B);
template void Array<double>::SetEqual(const Array *B,
				      unsigned int N, unsigned int M);
template void Array<double>::SetEqual(const Array *B, int startPosition);

//###################################################

template void Array<float>::SetEqual(const Array *B);
template void Array<float>::SetEqual(const Array *B,
				     unsigned int N, unsigned int M);
template void Array<float>::SetEqual(const Array *B, int startPosition);

//###################################################

template void Array<int>::SetEqual(const Array *B);
template void Array<int>::SetEqual(const Array *B, int startPosition);
template void Array<int>::Concat(Array *A);

//###################################################

template void Array<unsigned int>::SetEqual(const Array *B);
template void Array<unsigned int>::SetEqual(const Array *B, int startPosition);

template void Array<float>::SetEqualComb(const Array<float2> *B,
					 unsigned int N, unsigned int M);
template void Array<float2>::SetEqual(const Array *B);
template void Array<float4>::SetEqual(const Array *B);
template void Array<double>::SetEqualComb(const Array<double2> *B,
					  unsigned int N, unsigned int M);
template void Array<double2>::SetEqual(const Array *B);
template void Array<double4>::SetEqual(const Array *B);

}
