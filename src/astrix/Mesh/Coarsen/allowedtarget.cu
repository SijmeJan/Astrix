// -*-c++-*-
/*! \file allowedtarget.cu
\brief Functions for determining allowed target triangles

\section LICENSE
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
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Predicates/predicates.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void FindAllowedTargetTriangleSingle(int n, int *pVRemove,
                                     int3 *pTv, int3 *pTe, int2 *pEt,
                                     int nVertex, real2 *pVc,
                                     real Px, real Py,
                                     Predicates *pred, real *pParam,
                                     int *pVertexTriangle,
                                     int *pTarget)
{
#ifndef __CUDA_ARCH__
  int printFlag = 0;
  //if (pVRemove[n] == 15081) printFlag = 1;
#endif

  const real zero  = (real) 0.0;

  // Vertex to be removed
  int vRemove = pVRemove[n];

  // Walking clockwise around vRemove, starting in triangle tStart
  int tStart = pVertexTriangle[n];

  int tTarget = -1;

  int t = tStart;
  int finished = 0;

#ifndef __CUDA_ARCH__
  if (printFlag == 1)
    std::cout << "Trying to remove vertex " << vRemove
              << ", starting triangle " << tStart << std::endl;
#endif

  while (!finished) {
    int ret = 1;
    int segmentFlag = 0;

    // Edge to cross to next triangle
    int eCross = -1;

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;

    real Ax, Bx, Cx, Ay, By, Cy;
    GetTriangleCoordinates(pVc, a, b, c,
                           nVertex, Px, Py,
                           Ax, Bx, Cx, Ay, By, Cy);


    // Translate triangle
    // pERIODIC
    TranslateTriangleToVertex(vRemove, Px, Py, nVertex,
                              a, b, c, Ax, Ay, Bx, By, Cx, Cy);

    while (a >= nVertex) a -= nVertex;
    while (b >= nVertex) b -= nVertex;
    while (c >= nVertex) c -= nVertex;
    while (a < 0) a += nVertex;
    while (b < 0) b += nVertex;
    while (c < 0) c += nVertex;

    // Coordinates of target vertex
    real x = zero, y = zero;

    int e1 = pTe[t].x;
    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    int t11 = pEt[e1].x;
    int t21 = pEt[e1].y;
    int t12 = pEt[e2].x;
    int t22 = pEt[e2].y;
    int t13 = pEt[e3].x;
    int t23 = pEt[e3].y;

    // Target vertex is next vertex in counterclockwise direction...
    if (a == vRemove) {
      eCross = e1;
      x = Bx;
      y = By;
      if (t11 == -1 || t21 == -1) segmentFlag = 1;
      // ...unless this would take the vertex away from a segment
      if (t13 == -1 || t23 == -1) {
        segmentFlag = 1;
        x = Cx;
        y = Cy;
      }
    }
    if (b == vRemove) {
      eCross = e2;
      x = Cx;
      y = Cy;
      if (t12 == -1 || t22 == -1) segmentFlag = 1;
      if (t11 == -1 || t21 == -1) {
        segmentFlag = 1;
        x = Ax;
        y = Ay;
      }
    }
    if (c == vRemove) {
      eCross = e3;
      x = Ax;
      y = Ay;
      if (t13 == -1 || t23 == -1) segmentFlag = 1;
      if (t12 == -1 || t22 == -1) {
        segmentFlag = 1;
        x = Bx;
        y = By;
      }
    }

#ifndef __CUDA_ARCH__
    if (printFlag == 1)
      std::cout << "Triangle: " << t
                << ", trying to move vertex to x = " << x
                << ", y = " << y << std::endl;
#endif

    //################################################################
    // Now we have found the target vertex coordinates (x, y) and we
    // know whether we are moving along a segment (segmentFlag = 1)
    //################################################################

    // Count how many illegal triangles are created by moving vertex
    int nBad = 0;

    int t2 = tStart;
    int eCross2 = -1;
    int finished2 = 0;
    while (!finished2) {
      int a = pTv[t2].x;
      int b = pTv[t2].y;
      int c = pTv[t2].z;

      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c,
                             nVertex, Px, Py,
                             ax, bx, cx, ay, by, cy);

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

      int e1 = pTe[t2].x;
      int e2 = pTe[t2].y;
      int e3 = pTe[t2].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;

      // Replace vRemove coord's with those of vTarget
      if (a == vRemove) {
        eCross2 = e1;
        ax = x;
        ay = y;
      }
      if (b == vRemove) {
        eCross2 = e2;
        bx = x;
        by = y;
      }
      if (c == vRemove) {
        eCross2 = e3;
        cx = x;
        cy = y;
      }

      real det = pred->orient2d(ax, ay, bx, by, cx, cy, pParam);

      //#####################################
      // NEW: Also check quality of triangle
      //#####################################

      int qualityCheck = 1;

      // Edge lengths squared
      real la = (bx - ax)*(bx - ax) + (by - ay)*(by - ay);
      real lb = (cx - bx)*(cx - bx) + (cy - by)*(cy - by);
      real lc = (cx - ax)*(cx - ax) + (cy - ay)*(cy - ay);

      real det2 = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
      real invdet2 = (real)1.0/(det2 + (real)1.0e-30);

      // Circumscribed radius (squared)
      real r2 = (real)0.25*la*lb*lc*invdet2*invdet2;

      if (r2 > (real) 2.0*min(la, min(lb, lc)))
        qualityCheck = 0;

#ifndef __CUDA_ARCH__
      if (printFlag == 1)
        std::cout << "  Triangle " << t2
                  << ", det = " << det
                  << ", beta = " << r2/min(la, min(lb, lc))
                  << std::endl;
#endif

      if (det <= zero || qualityCheck == 0)
        nBad++;

      // Next triangle
      if (eCross2 == e1)
        if (t11 != t2) t2 = t11; else t2 = t21;
      if (eCross2 == e2)
        if (t12 != t2) t2 = t12; else t2 = t22;
      if (eCross2 == e3)
        if (t13 != t2) t2 = t13; else t2 = t23;

      if (t2 == -1 || t2 == tStart) finished2 = 1;

    }

    // Too many bad triangles: reject triangle and vTarget
    if (nBad > 2 - segmentFlag) ret = 0;

#ifndef __CUDA_ARCH__
    if (printFlag == 1)
      std::cout << "  Number of bad triangles: " << nBad
                << ", return value " << ret << std::endl;
#endif

    if (segmentFlag == 1) tTarget = -1;

    if (ret == 1) {
      if (tTarget == -1)
        tTarget = t;
    }


    // Next triangle
    if (eCross == e1)
      if (t11 != t) t = t11; else t = t21;
    if (eCross == e2)
      if (t12 != t) t = t12; else t = t22;
    if (eCross == e3)
      if (t13 != t) t = t13; else t = t23;

    if (t == -1 || t == tStart) finished = 1;

  }

  pTarget[n] = tTarget;

  if (tTarget == -1) {
    pVRemove[n] = -1;
    pVertexTriangle[n] = -1;
  }

}

//#########################################################################
//#########################################################################

__global__
void devFindAllowedTargetTriangle(int *pVertexRemove, int nRemove,
                                  int3 *pTv, int3 *pTe, int2 *pEt,
                                  int nVertex, real2 *pVc,
                                  real Px, real Py,
                                  Predicates *pred, real *pParam,
                                  int *pVertexTriangle,
                                  int *pTarget)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FindAllowedTargetTriangleSingle(n, pVertexRemove,
                                    pTv, pTe, pEt,
                                    nVertex, pVc, Px, Py,
                                    pred, pParam,
                                    pVertexTriangle,
                                    pTarget);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Vertices are removed by moving them on top of a neighbouring vertex and subsequently adjusting the connections. However, not all vertices are suited to move another vertex on top of; it may lead to illegal triangles. This function will select suitable target triangles for all vertices to be removed.

\param *vertexTriangleAllowed Pointer to output array flagging whether the corresponding triangle in \a vertexTriangleList is suited as target triangle
\param *vertexTriangleList Pointer to Array of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex*/
//#########################################################################

int Coarsen::FindAllowedTargetTriangles(Connectivity *connectivity,
                                        Predicates *predicates,
                                        const MeshParameter *mp)
{
  int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  triangleTarget->SetSize(nRemove);

  int *pVertexTriangle = vertexTriangle->GetPointer();
  int *pTarget = triangleTarget->GetPointer();

  int *pVertexRemove = vertexRemove->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  // Find allowed target triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devFindAllowedTargetTriangle,
       (size_t) 0, 0);

    devFindAllowedTargetTriangle<<<nBlocks, nThreads>>>
      (pVertexRemove, nRemove,
       pTv, pTe, pEt,
       nVertex, pVc, Px, Py,
       predicates, pParam,
       pVertexTriangle,
       pTarget);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FindAllowedTargetTriangleSingle(n, pVertexRemove,
                                      pTv, pTe, pEt,
                                      nVertex, pVc,
                                      Px, Py,
                                      predicates, pParam,
                                      pVertexTriangle,
                                      pTarget);
  }

  nRemove = vertexRemove->RemoveValue(-1);
  vertexTriangle->RemoveValue(-1);
  triangleTarget->RemoveValue(-1);
  vertexRemove->SetSize(nRemove);
  vertexTriangle->SetSize(nRemove);
  triangleTarget->SetSize(nRemove);
  return nRemove;
}

}
