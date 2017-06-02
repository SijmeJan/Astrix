// -*-c++-*-
/*! \file adjuststate.cu
\brief Functions for conservatively adjusting state when coarsening Mesh

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Common/helper_math.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/state.h"

namespace astrix {

//#########################################################################
/*! \brief Adjust state when removing vertex \a vRemove

\param vRemove Vertex to be removed
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVertexArea Pointer to vertex areas (Voronoi cells)
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param Px Periodic domain size x
\param Py Periodic domain size y
\param tTarget Target triangle containing vertex to collapse \a vRemove onto
\param *state Pointer to state vector*/
//#########################################################################

template<class realNeq>
__host__ __device__
void AdjustStateCoarsenSingle(int vRemove, int tStart,
                              int3 *pTv, int3 *pTe, int2 *pEt,
                              real *pVertexArea,
                              int nVertex, real2 *pVc,
                              real Px, real Py, int tTarget,
                              realNeq *state)
{
  // Find target vertex (vertex to move vRemove onto)
  int a = pTv[tTarget].x;
  int b = pTv[tTarget].y;
  int c = pTv[tTarget].z;

  // Find target vertex coordinates
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  // Translate target triangle to same side as vRemove
  TranslateTriangleToVertex(vRemove, Px, Py,
                            nVertex, a, b, c, ax, ay, bx, by, cx, cy);

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int vTarget = -1;
  if (a == vRemove) {
    vTarget = b;
    int e = pTe[tTarget].z;
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    if (t1 == -1 || t2 == -1) vTarget = c;
  }
  if (b == vRemove) {
    vTarget = c;
    int e = pTe[tTarget].x;
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    if (t1 == -1 || t2 == -1) vTarget = a;
  }
  if (c == vRemove) {
    vTarget = a;
    int e = pTe[tTarget].y;
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    if (t1 == -1 || t2 == -1) vTarget = b;
  }

  real xTarget = ax;
  real yTarget = ay;
  if (b == vTarget) {
    xTarget = bx;
    yTarget = by;
  }
  if (c == vTarget) {
    xTarget = cx;
    yTarget = cy;
  }

  // Total Voronoi volume all neighbours + vRemove (x6)
  real totalVolume = 6.0*pVertexArea[vRemove];
  // State change for all neighbouring vertices
  realNeq dState = totalVolume*(state[vRemove] - state[vTarget]);

  // Calculate totalVolume
  int t = tStart;
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

    // Neighbour vertex
    int v = -1;
    if (a == vRemove) {
      v = b;
      eCross = pTe[t].x;
    }
    if (b == vRemove) {
      v = c;
      eCross = pTe[t].y;
    }
    if (c == vRemove) {
      v = a;
      eCross = pTe[t].z;
    }

    totalVolume += 6.0*pVertexArea[v];

    int tNext = pEt[eCross].x;
    if (tNext == t) tNext = pEt[eCross].y;
    t = tNext;

    if (t == tStart || t == -1) finished = 1;
  }

  real denominator = 1.0/totalVolume;

  // Calculate dState
  t = tStart;
  finished = 0;
  while (!finished) {
    // Edge to cross to next triangle
    int eCross = -1;

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;

    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    // Translate triangle to same side as vRemove
    TranslateTriangleToVertex(vRemove, Px, Py, nVertex,
                              a, b, c, ax, ay, bx, by, cx, cy);

    // Twice triangle area before removal
    real vBefore = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);

    while (a >= nVertex) a -= nVertex;
    while (b >= nVertex) b -= nVertex;
    while (c >= nVertex) c -= nVertex;
    while (a < 0) a += nVertex;
    while (b < 0) b += nVertex;
    while (c < 0) c += nVertex;

    if (a == vRemove) {
      eCross = pTe[t].x;
    }
    if (b == vRemove) {
      eCross = pTe[t].y;
    }
    if (c == vRemove) {
      eCross = pTe[t].z;
    }

    if (a == vRemove) {
      ax = xTarget;
      ay = yTarget;
    }
    if (b == vRemove) {
      bx = xTarget;
      by = yTarget;
    }
    if (c == vRemove) {
      cx = xTarget;
      cy = yTarget;
    }

    // Twice triangle area after removal
    real vAfter = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);

    // Change in (twice) triangle area
    real dV = vAfter - vBefore;

    if (a != vRemove) {
      dState -= dV*state[a];
      // Need to keep this up to date for Delaunay adjustState
      pVertexArea[a] += dV/6.0;
    }
    if (b != vRemove) {
      dState -= dV*state[b];
      // Need to keep this up to date for Delaunay adjustState
      pVertexArea[b] += dV/6.0;
    }
    if (c != vRemove) {
      dState -= dV*state[c];
      // Need to keep this up to date for Delaunay adjustState
      pVertexArea[c] += dV/6.0;
    }

    int tNext = pEt[eCross].x;
    if (tNext == t) tNext = pEt[eCross].y;
    t = tNext;

    if (t == tStart || t == -1) finished = 1;
  }

  pVertexArea[vTarget] += pVertexArea[vRemove];


  t = tStart;
  finished = 0;
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

    if (a == vRemove) {
      state[b] = state[b] + denominator*dState;
      eCross = pTe[t].x;
    }
    if (b == vRemove) {
      state[c] = state[c] + denominator*dState;
      eCross = pTe[t].y;
    }
    if (c == vRemove) {
      state[a] = state[a] + denominator*dState;
      eCross = pTe[t].z;
    }

    int tNext = pEt[eCross].x;
    if (tNext == t) tNext = pEt[eCross].y;
    t = tNext;

    if (t == tStart || t == -1) finished = 1;
  }
}

//#########################################################################
/*! \brief Kernel for adjusting state when removing vertices

\param nRemove Number of vertices that will be removed
\param *pVertexRemove Pointer to array of vertices to be removed
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVertexArea Pointer to vertex areas (Voronoi cells)
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pTriangleTarget Pointer to array of target triangles containing vertex to move \a vRemove onto
\param *state Pointer to state vector*/
//#########################################################################

template<class realNeq>
__global__
void devAdjustStateCoarsen(int nRemove, int *pVertexRemove,
                           int *pVertexTriangle,
                           int3 *pTv, int3 *pTe, int2 *pEt,
                           real *pVertexArea, int nVertex,
                           real2 *pVc, real Px, real Py,
                           int *pTriangleTarget, realNeq *state)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    AdjustStateCoarsenSingle(pVertexRemove[n],
                             pVertexTriangle[n],
                             pTv, pTe, pEt,
                             pVertexArea, nVertex, pVc,
                             Px, Py, pTriangleTarget[n], state);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Adjust state when removing vertices in order to conserve mass, momentum and energy.

\param *connectivity Pointer to Mesh Connectivity
\param *triangleTarget Pointer to Array containing triangle targets
\param *vertexState Pointer to state vector at vertices
\param *mp Pointer to Mesh Parameters*/
//#########################################################################

template<class realNeq>
void Coarsen::AdjustState(Connectivity *connectivity,
                          Array<realNeq> *vertexState,
                          const MeshParameter *mp)
{
  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  int *pVertexRemove = vertexRemove->GetPointer();
  int *pTriangleTarget = triangleTarget->GetPointer();
  int *pVertexTriangle = vertexTriangle->GetPointer();
  real *pVertexArea = connectivity->vertexArea->GetPointer();

  realNeq *state = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  // Adjust state
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustStateCoarsen<realNeq>,
                                       (size_t) 0, 0);

    devAdjustStateCoarsen<realNeq><<<nBlocks, nThreads>>>
      (nRemove, pVertexRemove, pVertexTriangle,
       pTv, pTe, pEt,
       pVertexArea, nVertex, pVc,
       Px, Py, pTriangleTarget, state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      AdjustStateCoarsenSingle(pVertexRemove[n],
                               pVertexTriangle[n],
                               pTv, pTe, pEt,
                               pVertexArea, nVertex, pVc,
                               Px, Py, pTriangleTarget[n], state);
  }

}

//#############################################################################
// Instantiate
//#############################################################################

template void
Coarsen::AdjustState<real>(Connectivity *connectivity,
                           Array<real> *vertexState,
                           const MeshParameter *mp);
template void
Coarsen::AdjustState<real3>(Connectivity *connectivity,
                            Array<real3> *vertexState,
                            const MeshParameter *mp);
template void
Coarsen::AdjustState<real4>(Connectivity *connectivity,
                            Array<real4> *vertexState,
                            const MeshParameter *mp);

}
