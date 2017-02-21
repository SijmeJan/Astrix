// -*-c++-*-
/*! \file testencroach.cu
\brief File containing function to test if any points to add encroaches on a segment.

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
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"

namespace astrix {

//##############################################################################
/*! \brief Check if point (\a x, \a y) encroaches any segment in triangle \a t

If point (x,y) is found to encroach on any segment in \a t, adjust \a x and \a y to lie at the centre of the segment and return 1; else return -1.

\param x x-coordinate of point to consider
\param y y-coordinate of point to consider
\param e1IsSegment flag whether first edge of triangle is segment
\param e2IsSegment flag whether second edge of triangle is segment
\param e3IsSegment flag whether third edge of triangle is segment
\param a First vertex of triangle
\param b Second vertex of triangle
\param c Third vertex of triangle
\param e1 First edge of triangle
\param e2 Second edge of triangle
\param e3 Third edge of triangle
\param nVertex Total number of vertices in Mesh
\param *pVc pointer to vertex coordinates
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//##############################################################################

__host__ __device__
int CheckEncroachTriangle(real& x, real& y,
                          const int e1IsSegment,
                          const int e2IsSegment,
                          const int e3IsSegment,
                          const int a, const int b , const int c,
                          const int e1, const int e2, const int e3,
                          const real2* __restrict__ pVc,
                          int nVertex, real Px, real Py)
{
  real zero = (real) 0.0;
  real half = (real) 0.5;

  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  real dot1 = (ax - x)*(bx - x) + (ay - y)*(by - y);
  real dot2 = (bx - x)*(cx - x) + (by - y)*(cy - y);
  real dot3 = (cx - x)*(ax - x) + (cy - y)*(ay - y);

  // Encroached edge
  int e = -1;
  if (e1IsSegment && dot1 < zero) e = e1;
  if (e2IsSegment && dot2 < zero) e = e2;
  if (e3IsSegment && dot3 < zero) e = e3;

  // Move vertex if necessary
  if (e == e1) {
    x = half*(ax + bx);
    y = half*(ay + by);
  }
  if (e == e2) {
    x = half*(bx + cx);
    y = half*(by + cy);
  }
  if (e == e3) {
    x = half*(cx + ax);
    y = half*(cy + ay);
  }

  return e;
}

//######################################################################
/*! \brief Test single point if it encroaches a segment.

This function tests if point to add \a i leads to an encroached segment. We take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encroached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question.

 \param i index of point to be inserted.
 \param *pElementAdd pointer to array of triangles or edges to place vertices on.
 \param *pVcAdd pointer to coordinates of points to be inserted.
 \param *pTv pointer to triangle vertices
 \param *pTe pointer to triangle edges
 \param *pEt pointer to edge triangles.
 \param *pVc pointer vertex coordinates.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y
 \param nTriangle Total number of triangles in Mesh*/
//######################################################################

__host__ __device__
void TestEncroachSingle(int i, int *pElementAdd, real2 *pVcAdd,
                        const int3* __restrict__ pTv,
                        const int3* __restrict__ pTe,
                        const int2* __restrict__ pEt,
                        const real2* __restrict__ pVc,
                        int nVertex, real Px, real Py, int nTriangle)
{
  int tStart = pElementAdd[i];
  int eStart = -1;
  if (tStart >= nTriangle) {
    eStart = tStart - nTriangle;
    tStart = -1;
  }

  // Check if starting on edge
  if (tStart == -1) {
    int t1 = pEt[eStart].x;
    int t2 = pEt[eStart].y;

    tStart = t1;
    if (tStart != -1) {
      int a = pTv[tStart].x;
      int b = pTv[tStart].y;
      int c = pTv[tStart].z;

      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c,
                             nVertex, Px, Py,
                             ax, bx, cx, ay, by, cy);
      real x = pVcAdd[i].x;
      real y = pVcAdd[i].y;

      // PERIODIC
      if ((ax < x && bx < x && cx < x) ||
          (ax > x && bx > x && cx > x) ||
          (ay < y && by < y && cy < y) ||
          (ay > y && by > y && cy > y))
        tStart = t2;
      if (tStart == -1) tStart = t1;
    } else {
      tStart = t2;
    }
  }

  int3 vCheck;
  vCheck.x = pTv[tStart].x;
  vCheck.y = pTv[tStart].y;
  vCheck.z = pTv[tStart].z;

  int vCheckOrig = -1, v = -1;

#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int n = 0; n < 3; n++) {
    if (n == 0) vCheckOrig = vCheck.x;
    if (n == 1) vCheckOrig = vCheck.y;
    if (n == 2) vCheckOrig = vCheck.z;
    if (n == 0) v = vCheck.x;
    if (n == 1) v = vCheck.y;
    if (n == 2) v = vCheck.z;

    while (v >= nVertex) v -= nVertex;
    while (v < 0) v += nVertex;

    int e1 = pTe[tStart].x;
    int e2 = pTe[tStart].y;
    int e3 = pTe[tStart].z;

    int eStart = e1;
    if (n == 1) eStart = e2;
    if (n == 2) eStart = e3;

    int t = tStart;
    int e = eStart;
    int tNext = -1;
    int eNext = -1;

    // Move in clockwise direction around v
    int finished = 0;
    int encroachEdge = -1;
    while (finished == 0) {
      real x = pVcAdd[i].x;
      real y = pVcAdd[i].y;

      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;

      if (a != vCheckOrig && b != vCheckOrig && c != vCheckOrig) {
        int N = nVertex;
        if (a > 3*N || b > 3*N || c > 3*N)
          y = pVcAdd[i].y + Py;
        if (a < -N || b < -N || c < -N)
          y = pVcAdd[i].y - Py;
        if ((a >= N && a < 2*N) || a >= 4*N || (a < -N && a >= -2*N) ||
            (b >= N && b < 2*N) || b >= 4*N || (b < -N && b >= -2*N) ||
            (c >= N && c < 2*N) || c >= 4*N || (c < -N && c >= -2*N))
          x = pVcAdd[i].x + Px;
        if ((a >= 2*N && a < 3*N) || (a < 0 && a >= -N) || a < -3*N ||
            (b >= 2*N && b < 3*N) || (b < 0 && b >= -N) || b < -3*N ||
            (c >= 2*N && c < 3*N) || (c < 0 && c >= -N) || c < -3*N)
          x = pVcAdd[i].x - Px;
      }

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;

      int e1IsSegment = (t11 == -1 || t21 == -1);
      int e2IsSegment = (t12 == -1 || t22 == -1);
      int e3IsSegment = (t13 == -1 || t23 == -1);

      if (e1IsSegment || e2IsSegment || e3IsSegment)
        encroachEdge =
          CheckEncroachTriangle(x, y,
                                e1IsSegment, e2IsSegment, e3IsSegment,
                                a, b, c,
                                e1, e2, e3,
                                pVc, nVertex, Px, Py);

      if (encroachEdge >= 0) {
        finished = 1;
        pElementAdd[i] = nTriangle + encroachEdge;
        pVcAdd[i].x = x;
        pVcAdd[i].y = y;
      }

      int t1 = pEt[e].x;
      int t2 = pEt[e].y;

      // Move across edge e
      tNext = t1;
      if (tNext == t) tNext = t2;

      if (tNext == -1 || tNext == tStart) finished = 1;

      if (tNext != -1) t = tNext;

      e1 = pTe[t].x;
      e2 = pTe[t].y;
      e3 = pTe[t].z;

      // Move across *next* edge
      if (e == e1) eNext = e2;
      if (e == e2) eNext = e3;
      if (e == e3) eNext = e1;
      e = eNext;
    }

    // Did we hit a segment? Reverse!
    if (tNext == -1 && encroachEdge == -1) {
      int e1 = pTe[tStart].x;
      int e2 = pTe[tStart].y;
      int e3 = pTe[tStart].z;

      int eStart = e3;
      if (n == 1) eStart = e1;
      if (n == 2) eStart = e2;

      t = tStart;
      e = eStart;
      eNext = -1;

      int t1 = pEt[e].x;
      int t2 = pEt[e].y;

      // Move across edge e
      tNext = t1;
      if (tNext == t) tNext = t2;

      finished = 0;
      if (tNext == -1) finished = 1;
      if (tNext != -1) t = tNext;

      e1 = pTe[t].x;
      e2 = pTe[t].y;
      e3 = pTe[t].z;

      // Move across *previous* edge
      if (e == e1) eNext = e3;
      if (e == e2) eNext = e1;
      if (e == e3) eNext = e2;
      e = eNext;

      // Move in counterclockwise direction around v
      while (finished == 0) {
        real x = pVcAdd[i].x;
        real y = pVcAdd[i].y;

        int a = pTv[t].x;
        int b = pTv[t].y;
        int c = pTv[t].z;

        if (a != vCheckOrig && b != vCheckOrig && c != vCheckOrig) {
          int N = nVertex;
          if (a > 3*N || b > 3*N || c > 3*N)
            y = pVcAdd[i].y - Py;
          if (a < -N || b < -N || c < -N)
            y = pVcAdd[i].y + Py;
          if ((a >= N && a < 2*N) || a >= 4*N || (a < -N && a >= -2*N) ||
              (b >= N && b < 2*N) || b >= 4*N || (b < -N && b >= -2*N) ||
              (c >= N && c < 2*N) || c >= 4*N || (c < -N && c >= -2*N))
            x = pVcAdd[i].x - Px;
          if ((a >= 2*N && a < 3*N) || (a < 0 && a >= -N) || a < -3*N ||
              (b >= 2*N && b < 3*N) || (b < 0 && b >= -N) || b < -3*N ||
              (c >= 2*N && c < 3*N) || (c < 0 && c >= -N) || c < -3*N)
            x = pVcAdd[i].x + Px;
        }

        int e1 = pTe[t].x;
        int e2 = pTe[t].y;
        int e3 = pTe[t].z;

        int t11 = pEt[e1].x;
        int t21 = pEt[e1].y;
        int t12 = pEt[e2].x;
        int t22 = pEt[e2].y;
        int t13 = pEt[e3].x;
        int t23 = pEt[e3].y;

        int e1IsSegment = (t11 == -1 || t21 == -1);
        int e2IsSegment = (t12 == -1 || t22 == -1);
        int e3IsSegment = (t13 == -1 || t23 == -1);

        if (e1IsSegment || e2IsSegment || e3IsSegment)
          encroachEdge =
             CheckEncroachTriangle(x, y,
                                   e1IsSegment, e2IsSegment, e3IsSegment,
                                   a, b, c,
                                   e1, e2, e3,
                                   pVc, nVertex, Px, Py);

        if (encroachEdge >= 0) {
          finished = 1;
          pElementAdd[i] = nTriangle + encroachEdge;
          pVcAdd[i].x = x;
          pVcAdd[i].y = y;
        }

        int t1 = pEt[e].x;
        int t2 = pEt[e].y;

        // Move across edge e
        tNext = t1;
        if (tNext == t) tNext = t2;

        if (tNext == -1) finished = 1;
        if (tNext != -1) t = tNext;

        e1 = pTe[t].x;
        e2 = pTe[t].y;
        e3 = pTe[t].z;

        // Move across *previous* edge
        if (e == e1) eNext = e3;
        if (e == e2) eNext = e1;
        if (e == e3) eNext = e2;
        e = eNext;
      }
    }
  }
}

//######################################################################
/*! \brief Kernel testing points if they encroach a segment.

This function tests if point to add \a i leads to an encroached segment. We take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encriached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question.

 \param nRefine number of points to be inserted.
 \param *pElementAdd pointer to array of triangles or edges to place vertices on.
 \param *pVcAdd pointer to coordinates of points to be inserted.
 \param *pTv pointer to triangle vertices
 \param *pTe pointer to triangle edges
 \param *pEt pointer to edge triangles.
 \param *pVc pointer vertex coordinates.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y
 \param nTriangle Total number of triangles in Mesh*/
//######################################################################

__global__ void
devTestEncroach(int nRefine, int *pElementAdd, real2 *pVcAdd,
                const int3* __restrict__ pTv,
                const int3* __restrict__ pTe,
                const int2* __restrict__ pEt,
                const real2* __restrict__ pVc,
                int nVertex, real Px, real Py, int nTriangle)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    TestEncroachSingle(i, pElementAdd, pVcAdd,
                       pTv, pTe, pEt, pVc,
                       nVertex, Px, Py, nTriangle);

    i += gridDim.x*blockDim.x;
  }
}

//##############################################################################
/*! This function tests if any vertices to be inserted lead to an encroached segment. For every point \a i to be inserted, we take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encriached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question.

 \param *connectivity pointer to basic mesh data
 \param *meshParameter pointer to class holding parameters of the Mesh
 \param nRefine total number of points to consider*/
//##############################################################################

void Refine::TestEncroach(Connectivity * const connectivity,
                          const MeshParameter *meshParameter,
                          const int nRefine)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  nvtxEvent *nvtxEncroach = new nvtxEvent("TestEncroach", 2);

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, (const void *)
                                       devTestEncroach,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devTestEncroach<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd, pVcAdd,
       pTv, pTe, pEt, pVc, nVertex, Px, Py, nTriangle);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int i = 0; i < nRefine; i++) {
      //std::cout << i << " " << nRefine << " " << pElementAdd[i] << std::endl;
      TestEncroachSingle(i, pElementAdd, pVcAdd,
                         pTv, pTe, pEt, pVc, nVertex, Px, Py, nTriangle);
    }
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("TestEncroach.prof", nRefine, elapsedTime, cudaFlag);
#endif

  delete nvtxEncroach;
}

}  // namespace astrix
