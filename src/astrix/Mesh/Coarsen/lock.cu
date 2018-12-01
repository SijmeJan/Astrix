// -*-c++-*-
/*! \file lock.cu
\brief Functions for locking vertices surrounding deletion point to decide which can be deleted in parallel

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
#include "./coarsen.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../../Common/atomic.h"
#include "../triangleLow.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void LockTrianglesSingle(int3 *pTv, int3 *pTe, int2 *pEt,
                         int eTarget,
                         int randomInt,
                         int *pTriangleLock)
{
  // Find neighbouring triangle
  int2 tCollapse = pEt[eTarget];
  if (tCollapse.x == -1) {
    tCollapse.x = tCollapse.y;
    tCollapse.y = -1;
  }

  int eCross = eTarget;                 // Current edge to cross
  int t = tCollapse.x;                  // Current triangle
  int isSegment = (tCollapse.y == -1);  // Flag if eTarget is segment

  while (1) {
    // Set pTiC to maximum of pTiC and pRandom
    int old = AtomicMax(&(pTriangleLock[t]), randomInt);
    // Stop if old pTic[t] was larger
    if (old > randomInt) break;

    int3 E = pTe[t];

    // Update t, eCross
    WalkAroundEdge(eTarget, isSegment, t, eCross, E, pTe, pEt);

    // Done if we reach first triangle
    if (t == tCollapse.x) break;
  }

}

//#########################################################################
//#########################################################################

__global__ void
devLockTriangles(int nRemove, int3 *pTv, int3 *pTe, int2 *pEt,
                 int *pEdgeTarget, unsigned int *pRandom, int *pTriangleLock)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRemove) {
    LockTrianglesSingle(pTv, pTe, pEt,
                        pEdgeTarget[i], pRandom[i], pTriangleLock);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
//#########################################################################

void Coarsen::LockTriangles(Connectivity *connectivity,
                            Array<int> *triangleLock)
{
  triangleLock->SetToValue(-1);

  int nRemove = edgeCollapseList->GetSize();

  int *pEdgeCollapseList = edgeCollapseList->GetPointer();
  int *pTriangleLock = triangleLock->GetPointer();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  // Shuffle points to add to maximise parallelisation
  unsigned int *pRandom = randomUnique->GetPointer();

  // Adjust state
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devLockTriangles,
                                       (size_t) 0, 0);

    devLockTriangles<<<nBlocks, nThreads>>>
      (nRemove, pTv, pTe, pEt, pEdgeCollapseList, pRandom, pTriangleLock);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      LockTrianglesSingle(pTv, pTe, pEt,
                          pEdgeCollapseList[n], pRandom[n], pTriangleLock);
  }

}


}  // namespace astrix
