// -*-c++-*-
/*! \file findparallelflip.cu
\brief File containing function to find parallel flip set.

*/ /* \section LICENSE
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
#include "../../Common/atomic.h"
#include "../../Common/profile.h"

//#define NEW_FIND_PARALLEL_FLIP

namespace astrix {

__host__ __device__
void SelectParallelFlip(int i, int *pEdgeNonDelaunay,
                        int *pTriangleTaken,
                        const int2* __restrict__ pEt)
{
  int canBeFlipped = 0;
  int e = pEdgeNonDelaunay[i];
  int t1 = pEt[e].x;
  int t2 = pEt[e].y;

  int tTaken1 = 0;
  if (t1 != -1) tTaken1 = AtomicCAS(&(pTriangleTaken[t1]), 0, 1);
  if (tTaken1 == 0) {
    int tTaken2 = 0;
    if (t2 != -1) tTaken2 = AtomicCAS(&(pTriangleTaken[t2]), 0, 1);
    if (tTaken2 == 0) canBeFlipped = 1;
    else if (t1 != -1) pTriangleTaken[t1] = 0;
  }

  if (canBeFlipped == 0) e = -1;
  pEdgeNonDelaunay[i] = e;
}

__global__ void
devSelectParallelFlip(int nFlip,
                      int *pEdgeNonDelaunay,
                      int *pTriangleTaken,
                      const int2* __restrict__ pEt)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nFlip) {
    SelectParallelFlip(i, pEdgeNonDelaunay, pTriangleTaken, pEt);

    i += gridDim.x*blockDim.x;
  }
}


//############################################################################
/*! \brief Kernel filling array with triangles that are affected by flipping edge

  Triangles affected are those adjacent to the edge to be flipped

\param nFlip Number of edges to be flipped
\param *pTaff Output array containing affected triangles
\param *pTaffEdge Output array containing the index in \a pEnd of the edge to be flipped
\param *pEnd Pointer to array containing edges to be flipped
\param *pEt Pointer to edge triangles*/
//############################################################################

__global__ void
devFillAffectedTriangles(int nFlip, int *pTaff, int *pTaffEdge,
                         int *pEnd, int2 *pEt)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nFlip) {
    int e = pEnd[i];

    pTaffEdge[i] = i;
    pTaffEdge[i + nFlip] = i;

    pTaff[i]         = pEt[e].x;
    pTaff[i + nFlip] = pEt[e].y;

    i += gridDim.x*blockDim.x;
  }
}

//############################################################################
/*! \brief Fill array with triangles that are affected by flipping edge

  Triangles affected are those adjacent to the edge to be flipped

\param nFlip Number of edges to be flipped
\param *triangleAffected Output Array containing affected triangles
\param *triangleAffectedEdge Output Array containing the index in \a pEnd of the edge to be flipped
\param *edgeNonDelaunay Pointer to Array containing edges to be flipped
\param *edgeTriangles Pointer to Array containing neighbouring triangles to edges
\param nFlip Number of edges to be flipped
\param cudaFlag Flag whether to do computation on device. This has to match the parameter cudaFlag of the Arrays*/
//############################################################################

void FillAffectedTriangles(Array<int> * const triangleAffected,
                           Array<int> * const triangleAffectedEdge,
                           const Array<int> *edgeNonDelaunay,
                           Connectivity * const connectivity,
                           const int nFlip,
                           const int cudaFlag)
{
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  int *pEnd = edgeNonDelaunay->GetPointer();

  int *pTaff = triangleAffected->GetPointer();
  int *pTaffEdge = triangleAffectedEdge->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillAffectedTriangles,
                                       (size_t) 0, 0);

    devFillAffectedTriangles<<<nBlocks, nThreads>>>
      (nFlip, pTaff, pTaffEdge, pEnd, pEt);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nFlip; i++) {
      int e = pEnd[i];

      pTaffEdge[i] = i;
      pTaffEdge[i + nFlip] = i;

      pTaff[i]         = pEt[e].x;
      pTaff[i + nFlip] = pEt[e].y;
    }
  }
}

//#############################################################################
/*! Compact the Array \a edgeNonDelaunay into a set of edges that can be flipped in parallel

\param *connectivity Pointer to basic Mesh data
\param nFlip Number of edges that are not Delaunay*/
//#############################################################################

int Delaunay::FindParallelFlipSet(Connectivity * const connectivity,
                                  const int nFlip)
{
#ifdef NEW_FIND_PARALLEL_FLIP

#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int nTriangle = connectivity->triangleVertices->GetSize();

  Array<int> *triangleTaken = new Array<int>(1, cudaFlag, nTriangle);
  triangleTaken->SetToValue(0);
  int *pTriangleTaken = triangleTaken->GetPointer();
  int *pEdgeNonDelaunay = edgeNonDelaunay->GetPointer();

  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSelectParallelFlip,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devSelectParallelFlip<<<nBlocks, nThreads>>>
      (nFlip, pEdgeNonDelaunay, pTriangleTaken, pEt);
#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventRecord(stop, 0) );
  gpuErrchk( cudaEventSynchronize(stop) );
#endif

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int i = 0; i < nFlip; i++)
      SelectParallelFlip(i, pEdgeNonDelaunay, pTriangleTaken, pEt);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

  // Keep only entries >= 0 (note: size of Array not changed!)
  int nFlipParallel = edgeNonDelaunay->RemoveValue(-1, nFlip);

  delete triangleTaken;

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("ParallelFlip.prof", nFlip, elapsedTime, cudaFlag);
#endif

#else

  // Fill triangleAffected and triangleAffectedEdge (direct only)
  FillAffectedTriangles(triangleAffected,
                        triangleAffectedEdge,
                        edgeNonDelaunay,
                        connectivity,
                        nFlip, cudaFlag);

  int firstEdge;
  edgeNonDelaunay->GetSingleValue(&firstEdge, 0);

  // Sort triangleAffected; reindex triangleAffectedEdge
  triangleAffected->SortByKey(triangleAffectedEdge, 2*nFlip);

  // Set edgeNonDelaunay[i] = -1 for non-unique triangles
  edgeNonDelaunay->ScatterUnique(triangleAffected, triangleAffectedEdge,
                                 2*nFlip, -1, -1);

  // Keep only entries >= 0 (note: size of Array not changed!)
  int nFlipParallel = edgeNonDelaunay->RemoveValue(-1, nFlip);

  // Pathological case
  if (nFlipParallel == 0 && nFlip > 0) {
    edgeNonDelaunay->SetSingleValue(firstEdge, 0);
    nFlipParallel = 1;
  }

#endif

  return nFlipParallel;
}

}  // namespace astrix
