// -*-c++-*-
/*! \file flagedges.cu
\brief Functions for flagging edges to be checked for Delaunay hood later

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Predicates/predicates.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void FlagEdgesSingle(int n, int3 *pTe, int2 *pEt,
                     int *peTarget, int *pEdgeNeedsChecking)
{
  // Starting edge
  int eStart = peTarget[n];

  // Starting triangle
  int tStart = pEt[eStart].x;
  if (tStart == -1) tStart = pEt[eStart].y;

  // Clockwise
  int e = eStart;
  int t = tStart;
  int finished = 0;
  while (!finished) {
    int eCross = -1;

    int e1 = pTe[t].x;
    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    // Flag edges for checking for Delaunay-hood later
    pEdgeNeedsChecking[e1] = e1;
    pEdgeNeedsChecking[e2] = e2;
    pEdgeNeedsChecking[e3] = e3;

    if (e == e1) eCross = e2;
    if (e == e2) eCross = e3;
    if (e == e3) eCross = e1;

    e = eCross;
    int tNext = pEt[e].x;
    if (tNext == t) tNext = pEt[e].y;
    t = tNext;

    if (t == -1 || e == eStart) finished = 1;
  }

  // Anti-clockwise
  e = eStart;
  t = tStart;
  finished = 0;
  while (!finished) {
    int eCross = -1;

    int e1 = pTe[t].x;
    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    // Flag edges for checking for Delaunay-hood later
    pEdgeNeedsChecking[e1] = e1;
    pEdgeNeedsChecking[e2] = e2;
    pEdgeNeedsChecking[e3] = e3;

    if (e == e1) eCross = e3;
    if (e == e2) eCross = e1;
    if (e == e3) eCross = e2;

    e = eCross;
    int tNext = pEt[e].x;
    if (tNext == t) tNext = pEt[e].y;
    t = tNext;

    if (t == -1 || e == eStart) finished = 1;
  }
}

//#########################################################################
//#########################################################################

__global__
void devFlagEdges(int nRemove, int3 *pTe, int2 *pEt,
                  int *peTarget, int *pEdgeNeedsChecking)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FlagEdgesSingle(n, pTe, pEt, peTarget, pEdgeNeedsChecking);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
//#########################################################################

void Coarsen::FlagEdges(Connectivity *connectivity)
{
  int nEdge = connectivity->edgeTriangles->GetSize();
  edgeNeedsChecking->SetSize(nEdge);
  edgeNeedsChecking->SetToValue(-1);

  int nRemove = edgeCollapseList->GetSize();

  int *pEdgeCollapseList = edgeCollapseList->GetPointer();
  int *pEdgeNeedsChecking = edgeNeedsChecking->GetPointer();

  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devFlagEdges,
       (size_t) 0, 0);

    devFlagEdges<<<nBlocks, nThreads>>>
      (nRemove, pTe, pEt, pEdgeCollapseList, pEdgeNeedsChecking);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FlagEdgesSingle(n, pTe, pEt, pEdgeCollapseList, pEdgeNeedsChecking);
  }
}

}
