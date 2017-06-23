// -*-c++-*-
/*! \file fillsub.cu
\brief Functions for determining substitution triangles for repairing edges in Mesh after flipping

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./delaunay.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Fill triangle substitution Array

  When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles

\param i Index in \a *pEnd to consider
\param *pEnd Pointer to list of edges that need to be flipped
\param *pTsub Pointer to output array containing substitution triangles
\param *pEt Pointer edge triangles*/
//#########################################################################

__host__ __device__
void FillTriangleSubstituteSingle(int i, int *pEnd, int *pTsub, int2 *pEt)
{
  int e = pEnd[i];
  int t1 = pEt[e].x;
  int t2 = pEt[e].y;

  pTsub[t1] = t2;
  pTsub[t2] = t1;
}

//#########################################################################
/*! \brief Kernel filling triangle substitution Array

  When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles

\param nNonDel Number of edges to be flipped
\param *pEnd Pointer to list of edges that need to be flipped
\param *pTsub Pointer to output array containing substitution triangles
\param *pEt Pointer edge triangles*/
//#########################################################################

__global__ void
devFillTriangleSubstitute(int nNonDel, int *pEnd, int *pTsub, int2 *pEt)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nNonDel) {
    FillTriangleSubstituteSingle(i, pEnd, pTsub, pEt);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles. These triangles are stored in the Array \a triangleSubstitute

\param *connectivity Pointer to basic Mesh data
\param nNonDel Number of edges to be flipped*/
//#########################################################################

void Delaunay::FillTriangleSubstitute(Connectivity * const connectivity,
                                      const int nNonDel)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  // Fill triangle substitution array
  int *pEnd = edgeNonDelaunay->GetPointer();

  int *pTsub = triangleSubstitute->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillTriangleSubstitute,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devFillTriangleSubstitute<<<nBlocks, nThreads>>>
      (nNonDel, pEnd, pTsub, pEt);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int i = 0; i < nNonDel; i++)
      FillTriangleSubstituteSingle(i, pEnd, pTsub, pEt);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("FillSub.prof", nNonDel, elapsedTime, cudaFlag);
#endif
}

}  // namespace astrix
