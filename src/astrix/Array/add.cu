// -*-c++-*-
/*! \file add.cu
\brief Functions for adding single value to whole array
*/
#include <cuda_runtime_api.h>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {


//###################################################
//! Add value to range in array on device
//###################################################

template<class T>
__global__ void
devAddValue(unsigned int startIndex, unsigned int endIndex,
            T *data, T value)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x + startIndex;

  while (i < endIndex) {
    data[i] += value;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// Add value to range in array
//###################################################

template <class T>
void Array<T>::AddValue(T value, unsigned int startIndex,
                        unsigned int endIndex)
{
  T *data = GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddValue<T>,
                                       (size_t) 0, 0);

    devAddValue<<<nBlocks, nThreads>>>(startIndex, endIndex, deviceVec, value);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = startIndex; i < endIndex; i++) data[i] += value;
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<int>::AddValue(int value,
                                   unsigned int startIndex,
                                   unsigned int endIndex);
template void Array<unsigned int>::AddValue(unsigned int value,
                                            unsigned int startIndex,
                                            unsigned int endIndex);

}
