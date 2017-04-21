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
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! \brief Adjust state when removing vertex \a vRemove

\param vRemove Vertex that will be removed
\param *vTri Pointer to triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertexArea Pointer to vertex areas (Voronoi cells)
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param tTarget Target triangle containing vertex to move vRemove onto
\param *dens Pointer to density at vertices
\param *momx Pointer to x-momentum at vertices
\param *momy Pointer to y-momentum at vertices
\param *ener Pointer to energy density at vertices
\param G Ration of specific heats
\param *vNeighbour Pointer to array of neighbouring vertices*/
//#########################################################################

__host__ __device__
void AdjustStateCoarsenSingle(int vRemove, int *vTri, const int maxTriPerVert,
                              int3 *pTv, int3 *pTe, int2 *pEt,
                              real *pVertexArea,
                              int nVertex, real2 *pVc,
                              real Px, real Py, int tTarget,
                              real4 *state,
                              //real *dens, real *momx,
                              //real *momy, real *ener,
                              real G, int *vNeighbour) {
  const real zero  = (real) 0.0;

  if (tTarget == -1) return;

  // Find target vertex (vertex to move vRemove onto)
  int a = pTv[tTarget].x;
  int b = pTv[tTarget].y;
  int c = pTv[tTarget].z;

  // Find target vertex coordinates
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  // Translate triangle
  // pERIODIC
  TranslateTriangleToVertex(vRemove, Px, Py,
                            nVertex, a, b, c, ax, ay, bx, by, cx, cy);

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  real vTarget = -1;
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
  for (int i = 0; i < maxTriPerVert; i++) {
    if (vNeighbour[i] != -1) {
      int uniqueFlag = 1;
      for (int j = 0; j < i; j++)
        uniqueFlag *= (vNeighbour[i] != vNeighbour[j]);

      if (uniqueFlag == 1)
        totalVolume += 6.0*pVertexArea[vNeighbour[i]];
    }
  }

  real dens = state[vRemove].x;
  real momx = state[vRemove].y;
  real momy = state[vRemove].z;

  real dDens = 6.0*pVertexArea[vRemove]*dens;
  real dMomx = 6.0*pVertexArea[vRemove]*momx;
  real dMomy = 6.0*pVertexArea[vRemove]*momy;

  real denominator = totalVolume - 6.0*pVertexArea[vRemove];

  for (int i = 0; i < maxTriPerVert; i++) {
    // Change of Voronoi volume
    real dV = zero;

    if (vNeighbour[i] != -1) {
      int uniqueFlag = 1;
      for (int j = 0; j < i; j++)
        uniqueFlag *= (vNeighbour[i] != vNeighbour[j]);

      if (uniqueFlag == 1) {
        for (int j = 0; j < maxTriPerVert; j++) {
          int t1 = vTri[j];

          if (t1 != -1) {
            int a = pTv[t1].x;
            int b = pTv[t1].y;
            int c = pTv[t1].z;

            real ax, bx, cx, ay, by, cy;
            GetTriangleCoordinates(pVc, a, b, c,
                                   nVertex, Px, Py,
                                   ax, bx, cx, ay, by, cy);

            // Replace vRemove by vTarget

            // Translate triangle
            // pERIODIC
            TranslateTriangleToVertex(vRemove, Px, Py, nVertex,
                                      a, b, c, ax, ay, bx, by, cx, cy);

            while (a >= nVertex) a -= nVertex;
            while (b >= nVertex) b -= nVertex;
            while (c >= nVertex) c -= nVertex;
            while (a < 0) a += nVertex;
            while (b < 0) b += nVertex;
            while (c < 0) c += nVertex;

            // Triangle surface before removal
            real vBefore = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);


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

            // Triangle surface after removal
            real vAfter = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);

            if (a == vNeighbour[i] || b == vNeighbour[i] ||
                c == vNeighbour[i]) {
              dV += vBefore - vAfter;
            } else {
              if (vNeighbour[i] == vTarget) dV -= vAfter;
            }
          }
        }

        real dens = state[vNeighbour[i]].x;
        real momx = state[vNeighbour[i]].y;
        real momy = state[vNeighbour[i]].z;

        dDens += dV*dens;
        dMomx += dV*momx;
        dMomy += dV*momy;

        pVertexArea[vNeighbour[i]] -= dV/6.0;
      }
    }

    denominator -= dV;
  }

  denominator = 1.0/denominator;

  for (int i = 0; i < maxTriPerVert; i++) {
    if (vNeighbour[i] != -1) {
      int uniqueFlag = 1;
      for (int j = 0; j < i; j++)
        uniqueFlag *= (vNeighbour[i] != vNeighbour[j]);

      if (uniqueFlag == 1) {
        state[vNeighbour[i]].x += dDens*denominator;
        state[vNeighbour[i]].y += dMomx*denominator;
        state[vNeighbour[i]].z += dMomy*denominator;
      }
    }
  }
}

__host__ __device__
void AdjustStateCoarsenSingle(int vRemove, int *vTri, const int maxTriPerVert,
                              int3 *pTv, int3 *pTe, int2 *pEt,
                              real *pVertexArea,
                              int nVertex, real2 *pVc,
                              real Px, real Py, int tTarget,
                              real3 *state, real G, int *vNeighbour)
{
  // Dummy: coarsening for three equations not supported
}

  __host__ __device__
void AdjustStateCoarsenSingle(int vRemove, int *vTri, const int maxTriPerVert,
                              int3 *pTv, int3 *pTe, int2 *pEt,
                              real *pVertexArea,
                              int nVertex, real2 *pVc,
                              real Px, real Py, int tTarget,
                              real *state, real G, int *vNeighbour)
{
  // Dummy: coarsening for one equation not supported
}

//#########################################################################
/*! \brief Kernel for adjusting state when removing vertices

\param nRemove Number of vertices that will be removed
\param *pVertexRemove Pointer to array of vertices to be removed
\param *pVertexTriangleList Pointer to triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertexArea Pointer to vertex areas (Voronoi cells)
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *triangleTarget Pointer to array of target triangles containing vertex to move vRemove onto
\param *dens Pointer to density at vertices
\param *momx Pointer to x-momentum at vertices
\param *momy Pointer to y-momentum at vertices
\param *ener Pointer to energy density at vertices
\param G Ration of specific heats
\param *pVertexNeighbour Pointer to array of neighbouring vertices*/
//#########################################################################

__global__
void devAdjustStateCoarsen(int nRemove, int *pVertexRemove,
                           int *pVertexTriangleList, int maxTriPerVert,
                           int3 *pTv, int3 *pTe, int2 *pEt,
                           real *pVertexArea, int nVertex,
                           real2 *pVc,
                           real Px, real Py, int *pTriangleTarget,
                           realNeq *state,
                           //real *dens, real *momx, real *momy, real *ener,
                           real G, int *pVertexNeighbour)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    AdjustStateCoarsenSingle(pVertexRemove[n],
                             &(pVertexTriangleList[n*maxTriPerVert]),
                             maxTriPerVert, pTv, pTe, pEt,
                             pVertexArea, nVertex, pVc,
                             Px, Py, pTriangleTarget[n], state,
                             //dens, momx, momy, ener,
                             G,
                             &(pVertexNeighbour[n*maxTriPerVert]));

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Adjust state when removing vertices in order to conserve mass, momentum and energy.

\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *vertexTriangleList Pointer to Array list of triangles sharing vertex
\param *triangleTarget Pointer to Array containing triangle targets
\param *vertexState Pointer to state vector at vertices
\param G ratio of specific heats
\param *vertexNeighbour Pointer to Array containing indices of neighbouring vertices */
//#########################################################################

void Coarsen::AdjustState(Connectivity *connectivity,
                          int maxTriPerVert,
                          Array<int> *vertexTriangleList,
                          Array<int> *triangleTarget,
                          Array<realNeq> *vertexState,
                          real G, const MeshParameter *mp,
                          Array<int> *vertexNeighbour)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexArea->TransformToHost();
      vertexState->TransformToHost();

      vertexNeighbour->TransformToHost();
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexArea->TransformToDevice();
      vertexState->TransformToDevice();

      vertexNeighbour->TransformToDevice();
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

  CalcVertexArea(connectivity, mp);

  int *pVertexRemove = vertexRemove->GetPointer();
  int *pVertexTriangleList = vertexTriangleList->GetPointer();
  int *pTriangleTarget = triangleTarget->GetPointer();
  int *pVertexNeighbour = vertexNeighbour->GetPointer();
  real *pVertexArea = vertexArea->GetPointer();

  realNeq *state = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  // Adjust state
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustStateCoarsen,
                                       (size_t) 0, 0);

    devAdjustStateCoarsen<<<nBlocks, nThreads>>>
      (nRemove, pVertexRemove, pVertexTriangleList,
       maxTriPerVert, pTv, pTe, pEt,
       pVertexArea, nVertex, pVc,
       Px, Py, pTriangleTarget, state,
       G, pVertexNeighbour);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      AdjustStateCoarsenSingle(pVertexRemove[n],
                               &(pVertexTriangleList[n*maxTriPerVert]),
                               maxTriPerVert, pTv, pTe, pEt,
                               pVertexArea, nVertex, pVc,
                               Px, Py, pTriangleTarget[n], state, G,
                               &(pVertexNeighbour[n*maxTriPerVert]));
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexArea->TransformToHost();
      vertexState->TransformToHost();

      vertexNeighbour->TransformToHost();
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexArea->TransformToDevice();
      vertexState->TransformToDevice();

      vertexNeighbour->TransformToDevice();
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

}

}
