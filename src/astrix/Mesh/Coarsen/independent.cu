// -*-c++-*-
/*! \file independent.cu
\brief Functions for finding vertices that can be inserted in parallel

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
/*! \brief Check whether all triangles in 'cavity' \n are available

Start at triangle \a tStart and move in clockwise direction around \a vRemove, checking if all triangles are available (i.e. not locked by other deletion point). If all are available, set pUniqueFlag[n] = 1, otherwise pUniqueFlag[n] = 0.

\param n Index of vertex to consider
\param vRemove Vertex to be removed
\param tStart Triangle to start walking around \a vRemove
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param tTarget Target triangle for collapse
\param randomInt Random integer associated with deletion point n
\param *pTriangleLock Pointer to triangle locks
\param *pUniqueFlag Output array of flags whether points can be deleted independent of all others*/
//#########################################################################

__host__ __device__
void IndependentSingle(int n, int3 *pTv, int3 *pTe, int2 *pEt,
                       int *pEdgeCollapseList,
                       unsigned int randomInt,
                       int *pTriangleLock)
{
  int eTarget = pEdgeCollapseList[n];

  // Find neighbouring triangle
  int2 tCollapse = pEt[eTarget];
  if (tCollapse.x == -1) {
    tCollapse.x = tCollapse.y;
    tCollapse.y = -1;
  }

  int eCross = eTarget;                 // Current edge to cross
  int t = tCollapse.x;                  // Current triangle
  int isSegment = (tCollapse.y == -1);  // Flag if eTarget is segment

  int ret = eTarget;
  while (1) {
    if (pTriangleLock[t] != randomInt) {
      ret = -1;
      break;
    }

    int3 E = pTe[t];

    // Update t, eCross
    WalkAroundEdge(eTarget, isSegment, t, eCross, E, pTe, pEt);

    // Done if we reach first triangle
    if (t == tCollapse.x) break;
  }

  pEdgeCollapseList[n] = ret;
}

//#########################################################################
/*! \brief Kernel checking whether all triangles in 'cavities' are available

Check if all triangles are available (i.e. not locked by other deletion points). If all are available, set pUniqueFlag[n] = 1, otherwise pUniqueFlag[n] = 0.

\param n Index of vertex to consider
\param *pVertexRemove Pointer to list of vertices to be removed
\param *pVertexTriangle Pointer to list of starting triangles
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param pTriangleTarget Target triangles for collapse
\param pRandom Random integers associated with deletion points
\param *pTriangleLock Pointer to triangle locks
\param *pUniqueFlag Output array of flags whether points can be deleted independent of all others*/
//#########################################################################

__global__ void
devIndependent(int nRemove,
               int3 *pTv, int3 *pTe, int2 *pEt,
               int *pEdgeCollapseList,
               unsigned int *pRandom,
               int *pTriangleLock)
{
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    IndependentSingle(n, pTv, pTe, pEt,
                      pEdgeCollapseList,
                      pRandom[n], pTriangleLock);

    n += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! For all deletion points, check if all triangles are available (i.e. not locked by other deletion points). If all are available, set pUniqueFlag[n] = 1, otherwise pUniqueFlag[n] = 0.

\param *connectivity Pointer to Mesh connectivity data
\param *triangleLock Pointer to triangle locks
\param *uniqueFlag Output array, specifying whether points can be removed in parallel (=1) or not (=0)*/
//#########################################################################

void Coarsen::FindIndependent(Connectivity *connectivity,
                              Array<int> *triangleLock)
{
  int nRemove = edgeCollapseList->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

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
                                       devIndependent,
                                       (size_t) 0, 0);

    devIndependent<<<nBlocks, nThreads>>>
      (nRemove, pTv, pTe, pEt, pEdgeCollapseList,
       pRandom, pTriangleLock);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      IndependentSingle(n, pTv, pTe, pEt, pEdgeCollapseList,
                        pRandom[n], pTriangleLock);
  }
}

}  // namespace astrix
