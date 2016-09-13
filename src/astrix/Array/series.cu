// -*-c++-*-
/*! \file series.cu
\brief Functions for setting array to series
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//! Kernel: Set a[i] = i
//######################################################################

template<class T>
__global__ void 
devSetToSeries(T *array, int realSize, int nDims, 
	       unsigned int startIndex, unsigned int endIndex)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x + startIndex;

    while (i < endIndex) {
      array[i + n*realSize] = i;
      i += gridDim.x*blockDim.x;
    }
  }
}

//##########################################################
// Set a[i] = i
//##########################################################

template <class T>
void Array<T>::SetToSeries()
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devSetToSeries<T>, 
				       (size_t) 0, 0);

    devSetToSeries<<<nBlocks, nThreads>>>(deviceVec,
					  realSize, nDims,
					  0, size);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0)
    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++) 
	hostVec[i + n*realSize] = i;
}

//##########################################################
// Set a[i] = i in range of indices
//##########################################################

template <class T>
void Array<T>::SetToSeries(unsigned int startIndex, 
			   unsigned int endIndex)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devSetToSeries<T>, 
				       (size_t) 0, 0);

    devSetToSeries<<<nBlocks, nThreads>>>(deviceVec,
					  realSize, nDims,
					  startIndex, endIndex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  }

  if (cudaFlag == 0)
    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = startIndex; i < endIndex; i++) 
	hostVec[i + n*realSize] = i;
}

//###################################################
// Instantiate
//###################################################

template void Array<int>::SetToSeries();
template void Array<int>::SetToSeries(unsigned int startIndex, 
					    unsigned int endIndex);

//###################################################

template void Array<unsigned int>::SetToSeries();
template void Array<unsigned int>::SetToSeries(unsigned int startIndex, 
						     unsigned int endIndex);

}
