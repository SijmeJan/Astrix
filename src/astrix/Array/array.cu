// -*-c++-*-
/*! \file array.cu
\brief Constructors and destructors for Array objects

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//#########################################################################
// Default constructor: 1-dimensional, no cuda
//#########################################################################

template <class T>
Array<T>::Array()
{
  dynArrayStep = 128;

  nDims = 1;
  cudaFlag = 0;
  hostVec = 0;
  deviceVec = 0;
  size = 0;
  realSize = dynArrayStep;

  // Allocate initial memory
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));
  memAllocatedHost += nDims*realSize*sizeof(T);
}

//#########################################################################
// Constructor, specifying dimension and cuda use
//#########################################################################

template <class T>
Array<T>::Array(unsigned int _nDims, int _cudaFlag)
{
  dynArrayStep = 128;

  nDims = _nDims;
  cudaFlag = _cudaFlag;
  hostVec = 0;
  deviceVec = 0;
  size = 0;
  realSize = dynArrayStep;

  // Allocate initial memory
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));
  memAllocatedHost += nDims*realSize*sizeof(T);

  if (cudaFlag == 1) {
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceVec),
                         nDims*realSize*sizeof(T)));
    memAllocatedDevice += nDims*realSize*sizeof(T);
  }
}

//#########################################################################
// Constructor, specifying dimension, cuda use and size
//#########################################################################

template <class T>
Array<T>::Array(unsigned int _nDims,
                int _cudaFlag, unsigned int _size)
{
  dynArrayStep = 128;

  nDims = _nDims;
  cudaFlag = _cudaFlag;
  hostVec = 0;
  deviceVec = 0;
  size = _size;
  realSize = ((size + dynArrayStep)/dynArrayStep)*dynArrayStep;

  // Allocate initial memory
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));
  memAllocatedHost += nDims*realSize*sizeof(T);

  if (cudaFlag == 1) {
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceVec),
                         nDims*realSize*sizeof(T)));
    memAllocatedDevice += nDims*realSize*sizeof(T);
  }
}

//#########################################################################
// Constructor, specifying dimension, cuda use, size and dynArrayStep
//#########################################################################

template <class T>
Array<T>::Array(unsigned int _nDims, int _cudaFlag,
                unsigned int _size, int _dynArrayStep)
{
  dynArrayStep = _dynArrayStep;

  nDims = _nDims;
  cudaFlag = _cudaFlag;
  hostVec = 0;
  deviceVec = 0;
  size = _size;
  realSize = ((size + dynArrayStep)/dynArrayStep)*dynArrayStep;

  // Allocate initial memory
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));
  memAllocatedHost += nDims*realSize*sizeof(T);

  if (cudaFlag == 1) {
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceVec),
                         nDims*realSize*sizeof(T)));
    memAllocatedDevice += nDims*realSize*sizeof(T);
  }
}

//#########################################################################
// Destructor
//#########################################################################

template <class T>
Array<T>::~Array()
{
  // Free host memory
  free(hostVec);
  memAllocatedHost -= nDims*realSize*sizeof(T);

  if (cudaFlag == 1) {
    // Free device memory
    gpuErrchk(cudaFree(deviceVec));
    memAllocatedDevice -= nDims*realSize*sizeof(T);
  }
}

//###################################################
// Instantiate
//###################################################

template Array<double>::Array();
template Array<double>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<double>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<double>::~Array();

//#############################################################################

template Array<double4>::Array();
template Array<double4>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<double4>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<double4>::~Array();

//#############################################################################

template Array<double3>::Array();
template Array<double3>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<double3>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<double3>::~Array();

//#############################################################################

template Array<double2>::Array();
template Array<double2>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<double2>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<double2>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size,
                              int _dynArrayStep);
template Array<double2>::~Array();

//#############################################################################

template Array<float>::Array();
template Array<float>::Array(unsigned int _nDims,
                             int _cudaFlag);
template Array<float>::Array(unsigned int _nDims,
                             int _cudaFlag,
                             unsigned int _size);
template Array<float>::~Array();

//#############################################################################

template Array<float4>::Array();
template Array<float4>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<float4>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<float4>::~Array();

//#############################################################################

template Array<float3>::Array();
template Array<float3>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<float3>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<float3>::~Array();

//#############################################################################

template Array<float2>::Array();
template Array<float2>::Array(unsigned int _nDims,
                              int _cudaFlag);
template Array<float2>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size);
template Array<float2>::Array(unsigned int _nDims,
                              int _cudaFlag,
                              unsigned int _size,
                              int _dynArrayStep);
template Array<float2>::~Array();

//#############################################################################

template Array<int>::Array();
template Array<int>::Array(unsigned int _nDims,
                           int _cudaFlag);
template Array<int>::Array(unsigned int _nDims,
                           int _cudaFlag,
                           unsigned int _size);
template Array<int>::Array(unsigned int _nDims,
                           int _cudaFlag,
                           unsigned int _size,
                           int _dynArrayStep);
template Array<int>::~Array();

//#############################################################################

template Array<int4>::Array();
template Array<int4>::Array(unsigned int _nDims,
                            int _cudaFlag);
template Array<int4>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size);
template Array<int4>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size,
                            int _dynArrayStep);
template Array<int4>::~Array();

//#############################################################################

template Array<int3>::Array();
template Array<int3>::Array(unsigned int _nDims,
                            int _cudaFlag);
template Array<int3>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size);
template Array<int3>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size,
                            int _dynArrayStep);
template Array<int3>::~Array();

//#############################################################################

template Array<int2>::Array();
template Array<int2>::Array(unsigned int _nDims,
                            int _cudaFlag);
template Array<int2>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size);
template Array<int2>::Array(unsigned int _nDims,
                            int _cudaFlag,
                            unsigned int _size,
                            int _dynArrayStep);
template Array<int2>::~Array();

//#############################################################################

template Array<unsigned int>::Array();
template Array<unsigned int>::Array(unsigned int _nDims,
                                    int _cudaFlag);
template Array<unsigned int>::Array(unsigned int _nDims,
                                    int _cudaFlag,
                                    unsigned int _size);
template Array<unsigned int>::~Array();

//#############################################################################

}
