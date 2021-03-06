// -*-c++-*-
/*! \file edgerepair.cu
\brief Functions for repairing edges in Mesh after flipping

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

#include "../triangleLow.h"

namespace astrix {

//#########################################################################
/*! \brief Repair edge \a i if necessary

Check if edge is part of both neighbouring triangles. If not, it has been corrupted by a flip and we have to use \a tSub

\param i Edge index to consider
\param *pTsub Pointer to array of substitution triangles
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles (modified)*/
//#########################################################################

__host__ __device__
void SingleEdgeRepair(int i, const int* __restrict__ pTsub,
                      const int3* __restrict__ pTe, int2 *pEt,
                      int printFlag)
{
  // Indices of triangles
  int t1 = pEt[i].x;
  int t2 = pEt[i].y;

  if (t1 != -1) {
    int tS1 = pTsub[t1];

    int e1 = pTe[t1].x;
    int e2 = pTe[t1].y;
    int e3 = pTe[t1].z;

    if (i != e1 && i != e2 && i != e3)
      t1 = tS1;
  }

  if (t2 != -1) {
    int tS2 = pTsub[t2];

    int e1 = pTe[t2].x;
    int e2 = pTe[t2].y;
    int e3 = pTe[t2].z;

    if (i != e1 && i != e2 && i != e3)
      t2 = tS2;
  }

  pEt[i].x = t1;
  pEt[i].y = t2;
}

//#########################################################################
/*! \brief Kernel repair edges if necessary

Check if all edges are part of both their neighbouring triangles. If not, it has been corrupted by a flip and we have to use \a pTsub

\param nEdge Total number of edges in Mesh
\param *pTsub Pointer to array of substitution triangles
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles (modified)*/
//#########################################################################

__global__ void
devEdgeRepair(int nEdge,
              const int* __restrict__ pTsub,
              const int3* __restrict__ pTe,
              int2 *pEt)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    SingleEdgeRepair(i, pTsub, pTe, pEt, 1);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! \brief Kernel repair limited amount of edges if necessary

Check if edges are part of both their neighbouring triangles. If not, it has been corrupted by a flip and we have to use \a pTsub. Only edges part of insertion cavities are checked.

\param nEdgeCheck Total number of edges to check
\param *pEnC Array listing edges to check
\param *pTsub Pointer to array of substitution triangles
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles (modified)*/
//#########################################################################

__global__ void
devEdgeRepairLimit(int nEdgeCheck,
                   const int *pEnC,
                   const int* __restrict__ pTsub,
                   const int3* __restrict__ pTe,
                   int2 *pEt)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdgeCheck) {
    int e = pEnC[i];
    SingleEdgeRepair(e, pTsub, pTe, pEt, 0);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! Repair edges if necessary. An edge flip can corrupt the mesh, resulting in the edge having the wrong neighbouring triangles; fortunately, it is relatively easy to repair by using the previously computed \a triangleSubstitute Array.

\param *connectivity Pointer to basic Mesh data
\param *edgeNeedsChecking pEdgeNeedsChecking[i] = i if edge \a i was part of the insertion cavity of a new vertex. If \a edgeNeedsChecking == 0 then all edges are checked
\param nEdgeCheck Number of edges to check if \a edgeNeedsChecking != 0
*/
//#########################################################################

void Delaunay::EdgeRepair(Connectivity * const connectivity,
                          Array<int> * const edgeNeedsChecking,
                          const int nEdgeCheck)
{
  int nEdge = connectivity->edgeTriangles->GetSize();

#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int *pTsub = triangleSubstitute->GetPointer();

  if (edgeNeedsChecking == 0) {
    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devEdgeRepair,
                                         (size_t) 0, 0);

#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(start, 0) );
#endif
      devEdgeRepair<<<nBlocks, nThreads>>>
        (nEdge, pTsub, pTe, pEt);
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
      for (int i = 0; i < nEdge; i++)
        SingleEdgeRepair(i, pTsub, pTe, pEt, 1);
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(stop, 0) );
      gpuErrchk( cudaEventSynchronize(stop) );
#endif
    }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("EdgeRepair.prof", nEdge, elapsedTime, cudaFlag);
#endif
  } else {
    int *pEnC = edgeNeedsChecking->GetPointer();

    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devEdgeRepairLimit,
                                         (size_t) 0, 0);

#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(start, 0) );
#endif
      devEdgeRepairLimit<<<nBlocks, nThreads>>>
        (nEdgeCheck, pEnC, pTsub, pTe, pEt);
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
      for (int i = 0; i < nEdgeCheck; i++)
        SingleEdgeRepair(pEnC[i], pTsub, pTe, pEt, 0);
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(stop, 0) );
      gpuErrchk( cudaEventSynchronize(stop) );
#endif
    }

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
    WriteProfileFile("EdgeRepair.prof", nEdgeCheck, elapsedTime, cudaFlag);
#endif
  }
}

}  // namespace astrix
