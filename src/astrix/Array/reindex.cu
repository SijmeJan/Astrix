// -*-c++-*-
/*! \file reindex.cu
\brief Functions for reindexing array
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
//! Kernel: reindex array, a[i] = a[reindex[i]]
//######################################################################

template<class T>
__global__ void
devReindex(int N, T *destArray, T *srcArray, unsigned int *reindex,
           int realSize, int nDims)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      destArray[i + n*realSize] = srcArray[reindex[i] + n*realSize];
      i += gridDim.x*blockDim.x;
    }
  }
}

//###################################################
// Reindex array: a[i] = a[reindex[i]]
//###################################################

template <class T>
void Array<T>::Reindex(unsigned int *reindex)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devReindex<T>,
                                       (size_t) 0, 0);

    T *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(T)));

    devReindex<<<nBlocks, nThreads>>>(size, temp, deviceVec, reindex,
                                      realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    T *temp = (T *)malloc(nDims*realSize*sizeof(T));

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < size; i++)
        temp[i + n*realSize] = hostVec[reindex[i] + n*realSize];

    free(hostVec);
    hostVec = temp;
  }
}

//#######################################################
// Reindex array: a[i] = a[reindex[i]] (first N elements)
//#######################################################

template <class T>
void Array<T>::Reindex(unsigned int *reindex, unsigned int N)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devReindex<T>,
                                       (size_t) 0, 0);

    T *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(T)));

    devReindex<<<nBlocks, nThreads>>>(N, temp, deviceVec, reindex,
                                      realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    T *temp = (T *)malloc(nDims*realSize*sizeof(T));

    for (unsigned int n = 0; n < nDims; n++)
      for (unsigned int i = 0; i < N; i++)
        temp[i + n*realSize] = hostVec[reindex[i] + n*realSize];

    free(hostVec);
    hostVec = temp;
  }
}

//######################################################################
//! Kernel: inverse reindex array, a[i] = reindex[a[i]]
//######################################################################

__global__ void
devInverseReindex(int N, int *destArray, int *srcArray,
                  unsigned int *reindex, int realSize, int nDims,
                  int maxValue, bool ignoreValue)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      int ret = -1;
      int tmp = srcArray[i + n*realSize];
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize] = ret;

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//! Kernel: inverse reindex array, a[i] = reindex[a[i]]
//######################################################################

__global__ void
devInverseReindexInt2Bool(int N, int2 *destArray, int2 *srcArray,
                          unsigned int *reindex, int realSize, int nDims,
                          int maxValue, bool ignoreValue)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      int ret = -1;
      int tmp = srcArray[i + n*realSize].x;
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize].x = ret;

      ret = -1;
      tmp = srcArray[i + n*realSize].y;
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize].y = ret;

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//! Kernel: inverse reindex array, a[i] = reindex[a[i]]
//######################################################################

__global__ void
devInverseReindexInt3Bool(int N, int3 *destArray, int3 *srcArray,
                          unsigned int *reindex, int realSize, int nDims,
                          int maxValue, bool ignoreValue)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      int ret = -1;
      int tmp = srcArray[i + n*realSize].x;
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize].x = ret;

      ret = -1;
      tmp = srcArray[i + n*realSize].y;
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize].y = ret;

      ret = -1;
      tmp = srcArray[i + n*realSize].z;
      if (tmp != -1 || ignoreValue == false) {
        int addValue = 0;
        while (tmp >= maxValue) {
          tmp -= maxValue;
          addValue += maxValue;
        }
        while (tmp < 0) {
          tmp += maxValue;
          addValue -= maxValue;
        }
        ret = (int) reindex[tmp] + addValue;
      }
      destArray[i + n*realSize].z = ret;

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//! Kernel: inverse reindex array, a[i] = reindex[a[i]]
//######################################################################

__global__ void
devInverseReindexInt(int N, int *destArray, int *srcArray,
                     int *reindex, int realSize, int nDims)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      int tmp = srcArray[i + n*realSize];
      destArray[i + n*realSize] = reindex[tmp];

      i += gridDim.x*blockDim.x;
    }
  }
}

//######################################################################
//! Kernel: inverse reindex array, a[i] = reindex[a[i]]
//######################################################################

__global__ void
devInverseReindexInt3(int N, int3 *destArray, int3 *srcArray,
                      int *reindex, int realSize, int nDims)
{
  for (unsigned int n = 0; n < nDims; n++) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    while (i < N) {
      int tmp = srcArray[i + n*realSize].x;
      destArray[i + n*realSize].x = reindex[tmp];
      tmp = srcArray[i + n*realSize].y;
      destArray[i + n*realSize].y = reindex[tmp];
      tmp = srcArray[i + n*realSize].z;
      destArray[i + n*realSize].z = reindex[tmp];

      i += gridDim.x*blockDim.x;
    }
  }
}

//###################################################
// Inverse reindex array: a[i] = reindex[a[i]]
//###################################################

template <>
void Array<int>::InverseReindex(int *reindex)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInverseReindexInt,
                                       (size_t) 0, 0);

    int *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(int)));

    devInverseReindexInt<<<nBlocks, nThreads>>>(size, temp, deviceVec, reindex,
                                                realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    int *temp = (int *)malloc(nDims*realSize*sizeof(int));

    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        int tmp = hostVec[i + n*realSize];
        temp[i + n*realSize] = reindex[tmp];
      }
    }

    free(hostVec);
    hostVec = temp;
  }
}

//###################################################
// Inverse reindex array: a[i] = reindex[a[i]]
//###################################################

template <>
void Array<int3>::InverseReindex(int *reindex)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInverseReindexInt,
                                       (size_t) 0, 0);

    int3 *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(int3)));

    devInverseReindexInt3<<<nBlocks, nThreads>>>(size, temp, deviceVec, reindex,
                                                 realSize, nDims);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    int3 *temp = (int3 *)malloc(nDims*realSize*sizeof(int3));

    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        int tmp = hostVec[i + n*realSize].x;
        temp[i + n*realSize].x = reindex[tmp];
        tmp = hostVec[i + n*realSize].y;
        temp[i + n*realSize].y = reindex[tmp];
        tmp = hostVec[i + n*realSize].z;
        temp[i + n*realSize].z = reindex[tmp];
      }
    }

    free(hostVec);
    hostVec = temp;
  }
}

//###################################################
// Inverse reindex array: a[i] = reindex[a[i]]
//###################################################

template <>
void Array<int>::InverseReindex(unsigned int *reindex,
                                int maxValue,
                                bool ignoreValue)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInverseReindex,
                                       (size_t) 0, 0);

    int *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(int)));

    devInverseReindex<<<nBlocks, nThreads>>>(size, temp, deviceVec, reindex,
                                             realSize, nDims, maxValue,
                                             ignoreValue);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    int *temp = (int *)malloc(nDims*realSize*sizeof(int));

    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        temp[i + n*realSize] = -1;
        int tmp = hostVec[i + n*realSize];
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize] = (int) reindex[tmp] + addValue;
        }
      }
    }

    free(hostVec);
    hostVec = temp;
  }
}

//###################################################
// Inverse reindex array: a[i] = reindex[a[i]]
//###################################################

template <>
void Array<int2>::InverseReindex(unsigned int *reindex,
                                 int maxValue,
                                 bool ignoreValue)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInverseReindexInt2Bool,
                                       (size_t) 0, 0);

    int2 *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(int2)));

    devInverseReindexInt2Bool<<<nBlocks, nThreads>>>
      (size, temp, deviceVec, reindex, realSize, nDims, maxValue, ignoreValue);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    int2 *temp = (int2 *)malloc(nDims*realSize*sizeof(int2));

    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        temp[i + n*realSize].x = -1;
        int tmp = hostVec[i + n*realSize].x;
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize].x = (int) reindex[tmp] + addValue;
        }

        temp[i + n*realSize].y = -1;
        tmp = hostVec[i + n*realSize].y;
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize].y = (int) reindex[tmp] + addValue;
        }
      }
    }

    free(hostVec);
    hostVec = temp;
  }
}

//###################################################
// Inverse reindex array: a[i] = reindex[a[i]]
//###################################################

template <>
void Array<int3>::InverseReindex(unsigned int *reindex,
                                 int maxValue,
                                 bool ignoreValue)
{
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInverseReindexInt3Bool,
                                       (size_t) 0, 0);

    int3 *temp;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&temp),
                         nDims*realSize*sizeof(int3)));

    devInverseReindexInt3Bool<<<nBlocks, nThreads>>>
      (size, temp, deviceVec, reindex, realSize, nDims, maxValue, ignoreValue);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaFree(deviceVec));
    deviceVec = temp;
  }

  if (cudaFlag == 0) {
    // Temporary array
    int3 *temp =
      (int3 *)malloc(nDims*realSize*sizeof(int3));

    for (unsigned int n = 0; n < nDims; n++) {
      for (unsigned int i = 0; i < size; i++) {
        temp[i + n*realSize].x = -1;
        int tmp = hostVec[i + n*realSize].x;
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize].x = (int) reindex[tmp] + addValue;
        }

        temp[i + n*realSize].y = -1;
        tmp = hostVec[i + n*realSize].y;
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize].y = (int) reindex[tmp] + addValue;
        }

        temp[i + n*realSize].z = -1;
        tmp = hostVec[i + n*realSize].z;
        if (tmp != -1 || ignoreValue == false) {
          int addValue = 0;
          while (tmp >= maxValue) {
            tmp -= maxValue;
            addValue += maxValue;
          }
          while (tmp < 0) {
            tmp += maxValue;
            addValue -= maxValue;
          }
          temp[i + n*realSize].z = (int) reindex[tmp] + addValue;
        }
      }
    }

    free(hostVec);
    hostVec = temp;
  }
}

//###################################################
// Instantiate
//###################################################

template void Array<float>::Reindex(unsigned int *reindex);
template void Array<float>::Reindex(unsigned int *reindex, unsigned int N);

//###################################################

template void Array<double>::Reindex(unsigned int *reindex);
template void Array<double>::Reindex(unsigned int *reindex, unsigned int N);

//###################################################

template void Array<int>::Reindex(unsigned int *reindex);
template void Array<int>::Reindex(unsigned int *reindex, unsigned int N);

//###################################################

template void Array<unsigned int>::Reindex(unsigned int *reindex);

//###################################################

template void Array<int2>::Reindex(unsigned int *reindex);
template void Array<int3>::Reindex(unsigned int *reindex);
template void Array<float2>::Reindex(unsigned int *reindex);
template void Array<float4>::Reindex(unsigned int *reindex);
template void Array<double2>::Reindex(unsigned int *reindex);
template void Array<double4>::Reindex(unsigned int *reindex);

}
