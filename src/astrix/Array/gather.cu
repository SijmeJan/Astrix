// -*-c++-*-
/*! \file gather.cu
\brief Gather and Scatter functions for Arrays
*/
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//#####################################################
//! Kernel: if map[i] != value then out[i] = in[map[i]]
//#####################################################

template<class T>
__global__ void
devGatherIf(T *deviceVec, T *pIn, int *pMap, int value, int maxIndex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < maxIndex) {
    if (pMap[i] != value)
      deviceVec[i] = pIn[pMap[i]];

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// if map[i] != value then out[i] = in[map[i]]
//###################################################

template <class T>
void Array<T>::GatherIf(Array<T> *in, Array<int> *map, int value, int maxIndex)
{
  int *pMap = map->GetPointer();
  T *pIn = in->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devGatherIf<T>,
                                       (size_t) 0, 0);

    devGatherIf<<<nBlocks, nThreads>>>(deviceVec, pIn, pMap, value, maxIndex);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < maxIndex; i++)
      if (pMap[i] != value)
        hostVec[i] = pIn[pMap[i]];
  }
}

//#####################################################
//! Kernel: out[i] = in[map[i]]
//#####################################################

template<class T>
__global__ void
devGather(T *deviceVec, T *pIn, int *pMap, int maxIndex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < maxIndex) {
    deviceVec[i] = pIn[pMap[i]];

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// out[i] = in[map[i]]
//###################################################

template <class T>
void Array<T>::Gather(Array<T> *in, Array<int> *map, int maxIndex)
{
  int *pMap = map->GetPointer();
  T *pIn = in->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devGather<T>,
                                       (size_t) 0, 0);

    devGather<<<nBlocks, nThreads>>>(deviceVec, pIn, pMap, maxIndex);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < maxIndex; i++)
      hostVec[i] = pIn[pMap[i]];
  }
}

//#####################################################
//! Kernel: out[map[i]] = in[i]
//#####################################################

template<class T>
__global__ void
devScatter(T *deviceVec, T *pIn, int *pMap, int maxIndex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < maxIndex) {
    deviceVec[pMap[i]] = pIn[i];

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// out[map[i]] = in[i]
//###################################################

template <class T>
void Array<T>::Scatter(Array<T> *in, Array<int> *map, int maxIndex)
{
  int *pMap = map->GetPointer();
  T *pIn = in->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devScatter<T>,
                                       (size_t) 0, 0);

    devScatter<<<nBlocks, nThreads>>>(deviceVec, pIn, pMap, maxIndex);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < maxIndex; i++)
      hostVec[pMap[i]] = pIn[i];
  }
}

//#####################################################
//! Kernel: out[map[i]] = i
//#####################################################

template<class T, class S>
__global__ void
devScatterSeries(T *deviceVec, S *pMap, int maxIndex,
                 unsigned int mapDim, unsigned int mapRS)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < maxIndex) {
    for (unsigned int n = 0; n < mapDim; n++)
      deviceVec[pMap[i + n*mapRS]] = (T)i;

    i += gridDim.x*blockDim.x;
  }
}

//###################################################
// out[map[i]] = i
//###################################################

template <class T>
template <class S>
void Array<T>::ScatterSeries(Array<S> *map, unsigned int maxIndex)
{
  S *pMap = map->GetPointer(0);
  //unsigned int maxIndex = map->GetSize();
  unsigned int mapDim = map->GetDimension();
  unsigned int mapRS = map->GetRealSize();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devScatterSeries<T, S>,
                                       (size_t) 0, 0);

    devScatterSeries<<<nBlocks, nThreads>>>(deviceVec, pMap, maxIndex,
                                            mapDim, mapRS);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int i = 0; i < maxIndex; i++)
      for (unsigned int n = 0; n < mapDim; n++)
        hostVec[pMap[i + n*mapRS]] = (T)i;
  }
}

//##########################################################################

template void Array<int>::GatherIf(Array<int> *in,
                                   Array<int> *map,
                                   int value, int maxIndex);
template void Array<int>::Gather(Array<int> *in,
                                 Array<int> *map,
                                 int maxIndex);
template void Array<int>::Scatter(Array<int> *in,
                                  Array<int> *map,
                                  int maxIndex);

//##########################################################################

template void Array<unsigned int>::GatherIf(Array<unsigned int> *in,
                                            Array<int> *map,
                                            int value, int maxIndex);
template void Array<unsigned int>::Gather(Array<unsigned int> *in,
                                          Array<int> *map,
                                          int maxIndex);
template void Array<unsigned int>::Scatter(Array<unsigned int> *in,
                                           Array<int> *map,
                                           int maxIndex);
template void Array<unsigned int>::ScatterSeries(Array<unsigned int> *map,
                                                 unsigned int maxIndex);

//##########################################################################

}
