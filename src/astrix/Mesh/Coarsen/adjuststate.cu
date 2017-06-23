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
\param tStart Starting triangle to walk around \a vRemove
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
void AdjustStateCoarsenSingle(int3 *pTv, int3 *pTe, int2 *pEt,
                              real *pVertexArea,
                              int nVertex, real2 *pVc,
                              real Px, real Py, int eTarget,
                              real2 *pEdgeCoordinates,
                              realNeq *pState)
{
  real oneThird = 1.0/3.0;

  int2 tCollapse;
  int3 V, E;
  int v1, v2;
  GetEdgeVertices(eTarget, pTv, pTe, pEt, tCollapse, V, E, v1, v2);
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;

  real xTarget = pEdgeCoordinates[eTarget].x;
  real yTarget = pEdgeCoordinates[eTarget].y;

#ifndef __CUDA_ARCH__
  int printFlag = 0;
  //if (eTarget == 18723) printFlag = 1;
  if (printFlag == 1)
    std::cout << "Adjusting state for collapse of edge " << eTarget
              << " with vertices " << v1 << " and " << v2
              << " to new coordinates x = " << xTarget << ", y = "
              << yTarget << std::endl;
#endif

  int eCross = eTarget;                 // Current edge to cross
  int t = tCollapse.x;                  // Current triangle
  int isSegment = (tCollapse.y == -1);  // Flag if eTarget is segment

  real totalVolume = pVertexArea[v2];   // Total volume around edge
  if (isSegment)
    totalVolume += pVertexArea[v1];
  real newVolume = 0.0;                 // Volume of new vertex

  // Change in state for all neighbouring vertices
  realNeq dState =
    pVertexArea[v1]*pState[v1] +
    pVertexArea[v2]*pState[v2];

  //real totalMassBefore =
  //  pVertexArea[v2]*state::GetDensity<realNeq, CL_ADVECT>(pState[v2]);
  //if (isSegment)
  //  totalMassBefore +=
  //    pVertexArea[v1]*state::GetDensity<realNeq, CL_ADVECT>(pState[v1]);

#ifndef __CUDA_ARCH__
  if (printFlag == 1) {
    std::cout << "  Vertex " << v2
              << ", area " << pVertexArea[v2]
              << ", U " << state::GetDensity<realNeq, CL_ADVECT>(pState[v2])
              << std::endl;
    if (isSegment)
      std::cout << "  Vertex " << v1
                << ", area " << pVertexArea[v1]
                << ", U " << state::GetDensity<realNeq, CL_ADVECT>(pState[v1])
                << std::endl;
  }
#endif


  while (1) {
    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, V.x, V.y, V.z,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    // Twice triangle area before collapse
    real vBefore = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);

    // Replace vertices with new positions
    if (V.x == v1 || V.x == v2) {
      ax = xTarget;
      ay = yTarget;
    }
    if (V.y == v1 || V.y == v2) {
      bx = xTarget;
      by = yTarget;
    }
    if (V.z == v1 || V.z == v2) {
      cx = xTarget;
      cy = yTarget;
    }

    // Twice triangle area after collapse
    real vAfter = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
    // Change in triangle area due to collapse
    real dArea = (real) 0.5*(vAfter - vBefore);

    // Contribution to volume of new vertex
    newVolume += oneThird*(real)0.5*vAfter;

    if (V.x != v1 && V.x != v2)
      dState -= oneThird*dArea*pState[V.x];
    if (V.y != v1 && V.y != v2)
      dState -= oneThird*dArea*pState[V.y];
    if (V.z != v1 && V.z != v2)
      dState -= oneThird*dArea*pState[V.z];

    // Update t, eCross
    WalkAroundEdge(eTarget, isSegment, t, eCross, E, pTe, pEt);

    E = pTe[t];
    V = pTv[t];

    if (pEt[eCross].x == -1 || pEt[eCross].y == -1) {
      int e = NextEdgeCounterClockwise(eCross, E);
      int v = VertexNotPartOfEdge(e, V, E);

      totalVolume += pVertexArea[v];
      //totalMassBefore +=
      //  pVertexArea[v]*state::GetDensity<realNeq, CL_ADVECT>(pState[v]);
#ifndef __CUDA_ARCH__
      if (printFlag == 1) {
        std::cout << "  Vertex " << v
                  << ", area " << pVertexArea[v]
                  << ", U " << state::GetDensity<realNeq, CL_ADVECT>(pState[v])
                  << ", triangle " << t
                  << std::endl;
      }
#endif
    }

    // Done if we reach first triangle
    if (t == tCollapse.x) break;

    int v = VertexNotPartOfEdge(eCross, V, E);
    totalVolume += pVertexArea[v];
    //totalMassBefore +=
    //  pVertexArea[v]*state::GetDensity<realNeq, CL_ADVECT>(pState[v]);

#ifndef __CUDA_ARCH__
    if (printFlag == 1) {
      std::cout << "  Vertex " << v
                << ", area " << pVertexArea[v]
                << ", U " << state::GetDensity<realNeq, CL_ADVECT>(pState[v])
                << ", triangle " << t
                << std::endl;
    }
#endif

  }

  realNeq stateNew = pState[v1];
  dState = (dState - newVolume*stateNew)/totalVolume;

  eCross = eTarget;                 // Current edge to cross
  t = tCollapse.x;                  // Current triangle
  E = pTe[t];
  V = pTv[t];


#ifndef __CUDA_ARCH__
    if (printFlag == 1) {
      std::cout << "  Vertex " << v1
                << ", area " << newVolume
                << ", U " << state::GetDensity<realNeq, CL_ADVECT>(stateNew)
                << std::endl;
    }
#endif

  while (1) {
    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, V.x, V.y, V.z,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    // Twice triangle area before collapse
    real vBefore = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);

    // Replace vertices with new positions
    if (V.x == v1 || V.x == v2) {
      ax = xTarget;
      ay = yTarget;
    }
    if (V.y == v1 || V.y == v2) {
      bx = xTarget;
      by = yTarget;
    }
    if (V.z == v1 || V.z == v2) {
      cx = xTarget;
      cy = yTarget;
    }

    // Twice triangle area after collapse
    real vAfter = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
    // Change in triangle area due to collapse
    real dArea = (real) 0.5*(vAfter - vBefore);

    // Update vertex areas (needed to do Delaunay adjuststate later)
    pVertexArea[V.x] += oneThird*dArea;
    pVertexArea[V.y] += oneThird*dArea;
    pVertexArea[V.z] += oneThird*dArea;

    // Update t, eCross
    WalkAroundEdge(eTarget, isSegment, t, eCross, E, pTe, pEt);

    E = pTe[t];
    V = pTv[t];

    if (pEt[eCross].x == -1 || pEt[eCross].y == -1) {
      int e = NextEdgeCounterClockwise(eCross, E);
      int v = VertexNotPartOfEdge(e, V, E);

      pState[v] += dState;
    }

    // Done if we reach first triangle
    if (t == tCollapse.x) break;

    int v = VertexNotPartOfEdge(eCross, V, E);
    pState[v] += dState;
  }

  // Not sure which one is going to be deleted, so set both
  pState[v1] = stateNew + dState;
  pState[v2] = stateNew + dState;
  pVertexArea[v1] = newVolume;
  pVertexArea[v2] = newVolume;

  /*
  eCross = eTarget;                 // Current edge to cross
  t = tCollapse.x;                  // Current triangle
  E = pTe[t];
  V = pTv[t];
  real totalMassAfter = 0.0;
  real totalVolumeAfter = 0.0;
  if (isSegment) {
    totalMassAfter =
      pVertexArea[v2]*state::GetDensity<realNeq, CL_ADVECT>(pState[v2]);
    totalVolumeAfter = pVertexArea[v2];
  }

  while (1) {
    // Update t, eCross
    WalkAroundEdge(eTarget, isSegment, t, eCross, E, pTe, pEt);

    E = pTe[t];
    V = pTv[t];

    if (pEt[eCross].x == -1 || pEt[eCross].y == -1) {
      int e = NextEdgeCounterClockwise(eCross, E);
      int v = VertexNotPartOfEdge(e, V, E);

      totalVolumeAfter += pVertexArea[v];
      totalMassAfter +=
        pVertexArea[v]*state::GetDensity<realNeq, CL_ADVECT>(pState[v]);
    }

    // Done if we reach first triangle
    if (t == tCollapse.x) break;

    int v = VertexNotPartOfEdge(eCross, V, E);
    totalMassAfter +=
      pVertexArea[v]*state::GetDensity<realNeq, CL_ADVECT>(pState[v]);
    totalVolumeAfter += pVertexArea[v];
  }

#ifndef __CUDA_ARCH__
  if (std::abs(totalMassBefore - totalMassAfter) > 1.0e-10) {
    std::cout << "Error adjusting state for collapse of edge "
              << eTarget << " with triangles "
              << tCollapse.x << " and " << tCollapse.y << std::endl;
    std::cout << "Total mass: " << totalMassBefore << " "
              << totalMassAfter << std::endl;
    std::cout << "Total volume: " << totalVolume << " "
              << totalVolumeAfter << std::endl;
    int qq; std::cin >> qq;
  }

  if (printFlag == 1) {
    int qq; std::cin >> qq;
  }

#endif
  */
}

//#########################################################################
/*! \brief Kernel for adjusting state when removing vertices

\param nRemove Number of vertices that will be removed
\param *pVertexRemove Pointer to array of vertices to be removed
\param *pVertexTriangle Pointer to array of vertex triangles
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
void devAdjustStateCoarsen(int nRemove,
                           int3 *pTv, int3 *pTe, int2 *pEt,
                           real *pVertexArea, int nVertex,
                           real2 *pVc, real Px, real Py,
                           int *pEdgeTarget,
                           real2 *pEdgeCoordinates,
                           realNeq *state)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    AdjustStateCoarsenSingle(pTv, pTe, pEt,
                             pVertexArea, nVertex, pVc,
                             Px, Py, pEdgeTarget[n], pEdgeCoordinates,
                             state);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Adjust state when removing vertices in order to conserve mass, momentum and energy.

\param *connectivity Pointer to Mesh Connectivity
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

  int *pEdgeCollapseList = edgeCollapseList->GetPointer();
  real *pVertexArea = connectivity->vertexArea->GetPointer();
  real2 *pEdgeCoordinates = edgeCoordinates->GetPointer();

  realNeq *state = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int nRemove = edgeCollapseList->GetSize();
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
      (nRemove, pTv, pTe, pEt,
       pVertexArea, nVertex, pVc,
       Px, Py, pEdgeCollapseList, pEdgeCoordinates, state);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      AdjustStateCoarsenSingle(pTv, pTe, pEt,
                               pVertexArea, nVertex, pVc,
                               Px, Py, pEdgeCollapseList[n],
                               pEdgeCoordinates,
                               state);
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

}  // namespace astrix
