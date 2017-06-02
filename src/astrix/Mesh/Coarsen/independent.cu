// -*-c++-*-
/*! \file independent.cu
\brief Functions for finding vertices that can be inserted in parallel

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
#include "./coarsen.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../../Common/atomic.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void IndependentSingle(int n, int vRemove, int tStart,
                       int3 *pTv, int3 *pTe, int2 *pEt,
                       int nVertex, int tTarget,
                       unsigned int randomInt,
                       int *pTriangleLock,
                       int *pUniqueFlag)
{
  int ret = 1;

  int t = tStart;
  int tPrev = -1;
  int finished = 0;
  while (!finished) {
    // Edge to cross to next triangle
    int eCross = -1;

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;

    while (a >= nVertex) a -= nVertex;
    while (b >= nVertex) b -= nVertex;
    while (c >= nVertex) c -= nVertex;
    while (a < 0) a += nVertex;
    while (b < 0) b += nVertex;
    while (c < 0) c += nVertex;

    int3 E = pTe[t];
    // Edge on boundary of 'cavity'
    int eBound = -1;
    if (a == vRemove) {
      eCross = E.x;
      eBound = E.y;
    }
    if (b == vRemove) {
      eCross = E.y;
      eBound = E.z;
    }
    if (c == vRemove) {
      eCross = E.z;
      eBound = E.x;
    }

    if (pTriangleLock[t] != randomInt) {
      finished = 1;
      ret = 0;
    }

    // Check two extra triangles
    if ((t == tTarget || tPrev == tTarget) && finished == 0) {
      int t1 = pEt[eBound].x;
      if (t1 == t) t1 = pEt[eBound].y;

      if (t1 != -1) {
        if (pTriangleLock[t1] != randomInt) {
          finished = 1;
          ret = 0;
        }
      }
    }

    tPrev = t;
    int tNext = pEt[eCross].x;
    if (tNext == t) tNext = pEt[eCross].y;
    t = tNext;

    if (t == tStart || t == -1) finished = 1;
  }

  pUniqueFlag[n] = ret;
}

//#########################################################################
//#########################################################################

__global__ void
devIndependent(int nRemove,
               int *pVertexRemove, int *pVertexTriangle,
               int3 *pTv, int3 *pTe, int2 *pEt,
               int nVertex, int *pTriangleTarget,
               unsigned int *pRandom, int *pTriangleLock,
               int *pUniqueFlag)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRemove) {
    IndependentSingle(i, pVertexRemove[i],
                      pVertexTriangle[i],
                      pTv, pTe, pEt,
                      nVertex, pTriangleTarget[i],
                      pRandom[i], pTriangleLock,
                      pUniqueFlag);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
//#########################################################################

void Coarsen::FindIndependent(Connectivity *connectivity,
                              Array<int> *triangleLock,
                              Array<int> *uniqueFlag)
{
  int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  int *pVertexRemove = vertexRemove->GetPointer();
  int *pTriangleTarget = triangleTarget->GetPointer();
  int *pVertexTriangle = vertexTriangle->GetPointer();

  int *pTriangleLock = triangleLock->GetPointer();
  int *pUniqueFlag = uniqueFlag->GetPointer();

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
                                       devIndependent,
                                       (size_t) 0, 0);

    devIndependent<<<nBlocks, nThreads>>>
      (nRemove, pVertexRemove,
       pVertexTriangle,
       pTv, pTe, pEt,
       nVertex, pTriangleTarget,
       pRandom, pTriangleLock,
       pUniqueFlag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      IndependentSingle(n, pVertexRemove[n],
                        pVertexTriangle[n],
                        pTv, pTe, pEt,
                        nVertex, pTriangleTarget[n],
                        pRandom[n], pTriangleLock,
                        pUniqueFlag);
  }

}


}  // namespace astrix
