// -*-c++-*-
/*! \file flagedge.cu
\brief File containing functions to flag edges to be checked for Delaunay-hood.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Predicates/predicates.h"
#include "../Param/meshparameter.h"
#include "./../triangleLow.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Flag edges of cavity \a i to be checked for Delaunay-hood

Start at the insertion triangle and move in clockwise direction along the edge of the cavity, and flag all edges of every triangle of the cavity to be checked for Delaunay-hood (set pEnc[e] = 1).

\param i Insertion point to consider
\param *pVcAdd Coordinates of insertion points
\param *pElementAdd Insertion triangles or edges
\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVc Pointer to vertex coordinates
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to Predicates object
\param *pParam Pointer to parameter vector associated with Predicates
\param *pEnc Pointer to Array edgeNeedsChecking (output)*/
//#########################################################################

__host__ __device__
void FlagEdgeForChecking(int i, real2 *pVcAdd, int *pElementAdd,
                         int nTriangle, int3 *pTv, int3 *pTe,
                         int2 *pEt, real2 *pVc, int nVertex, real Px,
                         real Py, const Predicates *pred, real *pParam,
                         int *pEnC)
{
  real dx = pVcAdd[i].x;
  real dy = pVcAdd[i].y;

  real dxOld = dx;
  real dyOld = dy;
  // Flag if cavity lies across periodic boundary
  int translateFlag = 0;

  // Start at insertion triangle
  int tStart = pElementAdd[i];
  if (tStart >= nTriangle) {
    // Inserting on edge
    int e = tStart - nTriangle;
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;

    tStart = t1;
    if (t2 > t1) tStart = t2;

    if (t1 != -1 && t2 != -1) {
      // Check if vertex lies in circumcircle of triangle t1
      // This is usually the case but need not be for a periodic mesh
      int a = pTv[t1].x;
      int b = pTv[t1].y;
      int c = pTv[t1].z;

      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c, nVertex, Px, Py,
                             ax, bx, cx, ay, by, cy);

      real det = pred->incircle(ax, ay, bx, by, cx, cy, dx, dy, pParam);


      if (det < (real) 0.0) tStart = t2;
    }
  }

  int t = tStart;

  // Choose starting edge to have two neigbours
  int e[] = {pTe[t].x, pTe[t].y, pTe[t].z};
  int eStart = -1;
  for (int n = 0; n < 3; n++)
    if (pEt[e[n]].x != -1 && pEt[e[n]].y != -1) eStart = e[n];

  int eCrossed = eStart;
  int finished = 0;

  while (finished == 0) {
    // We know t is in cavity; flag edges to be checked
    pEnC[e[0]] = e[0];
    pEnC[e[1]] = e[1];
    pEnC[e[2]] = e[2];

    int tNext = -1;
    int eNext = -1;

    int e1 = -1, e2 = -1, e3 = -1;

    for (int de = 2; de >= 0; de--) {
      // Choose eNext clockwise from eCrossed
      if (eCrossed == e[0]) eNext = e[de % 3];
      if (eCrossed == e[1]) eNext = e[(1 + de) % 3];
      if (eCrossed == e[2]) eNext = e[(2 + de) % 3];

      // Triangle sharing eNext
      int2 tNeighbour = pEt[eNext];
      tNext = tNeighbour.x;
      if (tNext == t) tNext = tNeighbour.y;

      if (tNext != -1) {
        // Check if vertex n lies in circumcircle of triangle tNext
        int a = pTv[tNext].x;
        int b = pTv[tNext].y;
        int c = pTv[tNext].z;

        real ax, bx, cx, ay, by, cy;
        GetTriangleCoordinates(pVc, a, b, c, nVertex, Px, Py,
                               ax, bx, cx, ay, by, cy);

        e1 = pTe[tNext].x;
        e2 = pTe[tNext].y;
        e3 = pTe[tNext].z;

        // Translate (dx, dy) so that it lies on the same side as tNext
        int f = a;
        if (e2 == eNext) f = b;
        if (e3 == eNext) f = c;

        int A = pTv[t].x;
        int B = pTv[t].y;
        int C = pTv[t].z;

        int F = B;
        if (e[1] == eNext) F = C;
        if (e[2] == eNext) F = A;

        real dxNew = dx;
        real dyNew = dy;

        // Indicate that cavity lies across periodic boundary
        if (f != F) translateFlag = 1;
        TranslateVertexToVertex(f, F, Px, Py, nVertex, dxNew, dyNew);

        real det = pred->incircle(ax, ay, bx, by, cx, cy, dxNew, dyNew, pParam);

        // Check if triangle is part of cavity if we translate triangle
        // in stead of vertex
        real det2 = det;
        // Do this only when cavity lies across periodic boundary
        if (translateFlag == 1) {
          // Take coordinates from t (F, F + 1, dOld) and translate f + 2
          real Ax, Bx, Cx, Ay, By, Cy;
          GetTriangleCoordinates(pVc, A, B, C, nVertex, Px, Py,
                                 Ax, Bx, Cx, Ay, By, Cy);
          if (F == A) {
            Bx = dxOld;
            By = dyOld;
          }
          if (F == B) {
            Cx = dxOld;
            Cy = dyOld;
          }
          if (F == C) {
            Ax = dxOld;
            Ay = dyOld;
          }

          real Dx = ax;
          real Dy = ay;
          if (f == a) {
            Dx = cx;
            Dy = cy;
          }
          if (f == c) {
            Dx = bx;
            Dy = by;
          }

          TranslateVertexToVertex(F, f, Px, Py, nVertex, Dx, Dy);

          det2 = pred->incircle(Ax, Ay, Bx, By, Cx, Cy, Dx, Dy, pParam);
       }

        // If triangle not part of cavity, do not move into it
        if (det < (real) 0.0 && det2 < (real) 0.0) {
          tNext = -1;
        } else {
          // Move into tNext; use new coordinates
          dx = dxNew;
          dy = dyNew;
        }
      }

      // Done if trying to move across eStart but failing
      if (eNext == eStart && tNext == -1) {
        finished = 1;
        break;
      }

      // Found a triangle to move into
      if (tNext != -1) break;
    }

    eCrossed = eNext;
    t = tNext;
    e[0] = e1;
    e[1] = e2;
    e[2] = e3;

    // Done if we are moving back into tStart across eStart or if no
    // next triangle can be found
    if ((t == tStart && eCrossed == eStart) || t == -1) finished = 1;
  }
}

//#########################################################################
/*! \brief Kernel flagging edges of cavities to be checked for Delaunay-hood

Upon return, pEnC[i] = i if edge \a i needs to be checked.

\param nRefine Total number of (independent) insertion points
\param *pVcAdd Coordinates of insertion points
\param *pElementAdd Insertion triangles or edges
\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVc Pointer to vertex coordinates
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to Predicates object
\param *pParam Pointer to parameter vector associated with Predicates
\param *pEnc Pointer to Array edgeNeedsChecking (output)*/
//#########################################################################

__global__ void
devFlagEdgesForChecking(int nRefine, real2 *pVcAdd, int *pElementAdd,
                        int nTriangle, int3 *pTv, int3 *pTe,
                        int2 *pEt, real2 *pVc, int nVertex, real Px, real Py,
                        const Predicates *pred, real *pParam, int *pEnC)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    FlagEdgeForChecking(i, pVcAdd, pElementAdd, nTriangle,
                        pTv, pTe, pEt, pVc, nVertex, Px, Py,
                        pred, pParam, pEnC);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! \brief Flag edges to be tested for Delaunay-hood

Upon return, pEdgeNeedsChecking[i] = i if edge \a i needs to be checked. Only edges that are part of insertion cavities need to be checked later.

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to Predicates object
\param *meshParameter Pointer to Mesh parameters*/
//#########################################################################

void Refine::FlagEdgesForChecking(Connectivity * const connectivity,
                                  const Predicates *predicates,
                                  const MeshParameter *meshParameter)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  // Number of triangles and number of insertion points
  unsigned int nTriangle = connectivity->triangleVertices->GetSize();
  unsigned int nRefine = elementAdd->GetSize();

  unsigned int nVertex = connectivity->vertexCoordinates->GetSize();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();

  // Flag edges to be checked for Delaunay-hood
  int *pEnC = edgeNeedsChecking->GetPointer();
  edgeNeedsChecking->SetToValue(-1);

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       (const void *) devFlagEdgesForChecking,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devFlagEdgesForChecking<<<nBlocks, nThreads>>>
      (nRefine, pVcAdd, pElementAdd, nTriangle,
       pTv, pTe, pEt, pVc, nVertex, Px, Py, predicates,
       pParam, pEnC);
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
    for (int n = 0; n < (int) nRefine; n++)
      FlagEdgeForChecking(n, pVcAdd, pElementAdd, nTriangle,
                          pTv, pTe, pEt, pVc, nVertex, Px, Py,
                          predicates, pParam, pEnC);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("FlagEdge.prof", nRefine, elapsedTime, cudaFlag);
#endif
}

}  // namespace astrix
