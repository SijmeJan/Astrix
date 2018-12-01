// -*-c++-*-
/*! \file flagvertex.cu
\brief Functions for selecting vertices for removal.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../../Common/atomic.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void FillEdgeCollapseListSingle(int e, int2 *pEt, int *pTriangleWantRefine,
                                int *pEdgeCollapseList)
{
  int t1 = pEt[e].x;
  int t2 = pEt[e].y;

  int ret = e;

  if (t1 != -1)
    if (pTriangleWantRefine[t1] != -1) ret = -1;
  if (t2 != -1)
    if (pTriangleWantRefine[t2] != -1) ret = -1;

  pEdgeCollapseList[e] = ret;
}

//#########################################################################
//#########################################################################

__global__
void devFillEdgeCollapseList(int nEdge, int2 *pEt, int *pTriangleWantRefine,
                             int *pEdgeCollapseList)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    FillEdgeCollapseListSingle(i, pEt, pTriangleWantRefine,
                               pEdgeCollapseList);

    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
//#########################################################################

void Coarsen::FillEdgeCollapseList(Connectivity *connectivity,
                                   Array<int> *triangleWantRefine)
{
  int nEdge = connectivity->edgeTriangles->GetSize();

  edgeCollapseList->SetSize(nEdge);
  int *pEdgeCollapseList = edgeCollapseList->GetPointer();

  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int *pTriangleWantRefine = triangleWantRefine->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillEdgeCollapseList,
                                       (size_t) 0, 0);

    devFillEdgeCollapseList<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleWantRefine, pEdgeCollapseList);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++)
      FillEdgeCollapseListSingle(i, pEt, pTriangleWantRefine,
                                 pEdgeCollapseList);
  }

  int nRemove = edgeCollapseList->RemoveValue(-1);
  edgeCollapseList->SetSize(nRemove);
}

}  // namespace astrix
