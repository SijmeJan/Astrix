// -*-c++-*-
/*! \file validcollapse.cu
\brief Functions for determining if edge collapses are valid

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

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Predicates/predicates.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

__host__ __device__
int TriangleQuality(real ax, real ay, real bx, real by, real cx, real cy,
                    real qualityBound)
{
  // Edge lengths squared
  real la = (bx - ax)*(bx - ax) + (by - ay)*(by - ay);
  real lb = (cx - bx)*(cx - bx) + (cy - by)*(cy - by);
  real lc = (cx - ax)*(cx - ax) + (cy - ay)*(cy - ay);

  real det2 = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
  real invdet2 = (real)1.0/(det2 + (real)1.0e-30);

  // Circumscribed radius (squared)
  real r2 = (real)0.25*la*lb*lc*invdet2*invdet2;

  // Skinny triangle; return 0
  if (r2 > qualityBound*min(la, min(lb, lc))) return 0;

  // Quality OK, return 1
  return 1;
}

//#########################################################################
/*! Edge collapse involves replacing an edge \a eTest with a vertex. If the vertices of \a eTest are \a v1 and \a v2, we allow the new vertex \a v to be placed anywhere along \a eTest: \a v = \alpha \a v1 + (1-\alpha) \a v2 for some alpha between zero and unity. However, there are constraints: if, for example, \a v1 is part of a segment and \a v2 is not, we must have \alpha = 1. If both vertices are part of a segment, \a v can only be different from \a v1 if the segment joining \a eTest at \a v1 is parallel to \a eTest. Checking different values of \alpha proceeds in N steps, with \a alpha_i = i/(N-1) where \a i goes from zero to N-1 if there are no constraints. This function returns the minimum and maximum allowed \a i as an int2, given the constraints mentioned above.*/
//#########################################################################

__host__ __device__
int2 MinMaxAlpha(int N, int eTest, int3 *pTv, int3 *pTe, int2 *pEt,
                 int nVertex, real2 *pVc, real Px, real Py,
                 Predicates *pred, real *pParam,
                 int2& tCollapse, int& v1, int& v2,
                 real& v1x, real& v1y, real& v2x, real& v2y,
                 real2& cSegment1, real2& cSegment2)
{
#ifndef __CUDA_ARCH__
  int printFlag = 0;
  //if (eTest == 19115) printFlag = 1;
#endif

  real zero = (real) 0.0;

  // Return value; by default i can go from zero to N - 1
  int2 ret;
  ret.x = 0;
  ret.y = N;

  int3 E, V;         // Edges and vertices of current triangle

  // Find vertices belonging to eTest
  GetEdgeVertices(eTest, pTv, pTe, pEt, tCollapse, V, E, v1, v2);

  // For now: reject edge if stretched across boundary
  if (v1 < 0 || v1 >= nVertex || v2 < 0 || v2 >= nVertex) {
    ret.x = 1;
    ret.y = -1;
  }

  int eCross = eTest;                   // Current edge to cross
  int t = tCollapse.x;                  // Current triangle
  int isSegment = (tCollapse.y == -1);  // Flag if eTest is segment

#ifndef __CUDA_ARCH__
  if (printFlag == 1)
    std::cout << "Getting MinMaxAlpha for collapse of edge " << eTest
              << " with vertices " << v1 << " and " << v2 << std::endl;
#endif

  // Find two segments neighbouring either v1 or v2
  int segment1 = -1;
  int segment2 = -1;
  while (1) {
    // Check if eNext is segment
    int eNext = NextEdgeCounterClockwise(eCross, E);
    if (eNext == eTest)
      eNext = NextEdgeCounterClockwise(eNext, E);
    if (pEt[eNext].x == -1 || pEt[eNext].y == -1) {
      segment1 = eNext;
#ifndef __CUDA_ARCH__
      if (printFlag == 1) {
        std::cout << "  Found segment " << segment1 << std::endl;
      }
#endif
    }

    // Update t, eCross
    WalkAroundEdge(eTest, isSegment, t, eCross, E, pTe, pEt);

    // Check if eCrossed is segment
    if (pEt[eCross].x == -1 || pEt[eCross].y == -1) {
      // Too many segments; can not collapse edge
      if (segment2 != -1) {
        ret.x = 1;
        ret.y = 0;
        return ret;
      }
      segment2 = eCross;
#ifndef __CUDA_ARCH__
      if (printFlag == 1) {
        std::cout << "  Found segment " << segment2 << std::endl;
      }
#endif
    }

    // Done if we reach first triangle
    if (t == tCollapse.x) break;

    E = pTe[t];
  }

  // Coordinates of eTest
  GetTriangleCoordinatesSingle(pVc, v1, nVertex, Px, Py, v1x, v1y);
  GetTriangleCoordinatesSingle(pVc, v2, nVertex, Px, Py, v2x, v2y);

#ifndef __CUDA_ARCH__
  if (printFlag == 1) {
    std::cout << "  Found segments " << segment1 << " " << segment2
              << std::endl;
    std::cout << "  Edge vertex coordinates: " << v1 << " = (" << v1x
              << ", " << v1y << ") and " << v2
              << " = (" << v2x << ", " << v2y << ") "
              << std::endl;
  }
#endif

  // No segments found, nothing further to do
  if (segment1 == -1) return ret;

  // Find vertices belonging to first segment
  int2 T;
  int vs1, vs2;
  GetEdgeVertices(segment1, pTv, pTe, pEt, T, V, E, vs1, vs2);

  // One vertex is on segment
  if (isSegment != 1) {
    int a = NormalizeVertex(v1, nVertex);
    int b = NormalizeVertex(v2, nVertex);
    int A = NormalizeVertex(vs1, nVertex);
    int B = NormalizeVertex(vs2, nVertex);

    // v2 is on segment, v1 is not
    if (A == b || B == b) ret.x = N - 1;
    // v1 is on segment, v2 is not
    if (A == a || B == a) ret.y = 1;

#ifndef __CUDA_ARCH__
    if (printFlag == 1)
      std::cout << "  One vertex on segment " << ret.x << " " << ret.y
                << std::endl;
#endif

    return ret;
  }

  // Now we know we are moving along a segment
  real vs1x, vs1y;
  GetTriangleCoordinatesSingle(pVc, vs1, nVertex, Px, Py, vs1x, vs1y);
  real vs2x, vs2y;
  GetTriangleCoordinatesSingle(pVc, vs2, nVertex, Px, Py, vs2x, vs2y);

  // vs1 corresponds to v2
  TranslateVertexToVertex(v2, vs1, Px, Py, nVertex, vs1x, vs1y);
  TranslateVertexToVertex(v2, vs1, Px, Py, nVertex, vs2x, vs2y);

  // Needed for encroachment checking later
  cSegment1.x = vs2x;
  cSegment1.y = vs2y;

  // Zero if segment1 parallel to eTest
  real det1 = pred->orient2d(v1x, v1y, vs1x, vs1y, vs2x, vs2y, pParam);

#ifndef __CUDA_ARCH__
  if (printFlag == 1)
    std::cout << "  Determinant based on vertices "
              << v1 << " " << vs1 << " " << vs2 << ": " << det1
              << std::endl;
#endif

  // Get vertices second segment
  GetEdgeVertices(segment2, pTv, pTe, pEt, T, V, E, vs1, vs2);

  GetTriangleCoordinatesSingle(pVc, vs1, nVertex, Px, Py, vs1x, vs1y);
  GetTriangleCoordinatesSingle(pVc, vs2, nVertex, Px, Py, vs2x, vs2y);

  // vs2 corresponds to v1
  TranslateVertexToVertex(v1, vs2, Px, Py, nVertex, vs1x, vs1y);
  TranslateVertexToVertex(v1, vs2, Px, Py, nVertex, vs2x, vs2y);

  // Needed for encroachment checking later
  cSegment2.x = vs1x;
  cSegment2.y = vs1y;

  // Zero if segment2 parallel to eTest
  real det2 = pred->orient2d(v2x, v2y, vs1x, vs1y, vs2x, vs2y, pParam);

  // Can not move v1; alpha must be unity
  if (det1 != zero) ret.x = N - 1;
  // Can not move v2; alpha must be zero
  if (det2 != zero) ret.y = 1;
  // Can not move either, no value of alpha will do
  if (det1 != zero && det2 != zero) ret.y = -1;

#ifndef __CUDA_ARCH__
  if (printFlag == 1) {
    std::cout << "  Determinant based on vertices "
              << v1 << " " << vs1 << " " << vs2 << ": " << det2
              << std::endl;
    std::cout << "  Segment vertex coordinates: (" << cSegment1.x
              << ", " << cSegment1.y << "), (" << cSegment2.x << ", "
              << cSegment2.y << ")" << std::endl;
    std::cout << "  Return value: " << ret.x << " " << ret.y << std::endl;
    int qq; std::cin >> qq;
  }
#endif

  return ret;
}


//#########################################################################
//#########################################################################

__host__ __device__
void TestEdgeCollapseSingle(int n, int3 *pTv, int3 *pTe, int2 *pEt,
                            int nVertex, real2 *pVc, real Px, real Py,
                            Predicates *pred, real *pParam,
                            int *pEdgeCollapse, real2 *pEdgeCoordinates,
                            real qualityBound)
{
  real zero = (real) 0.0;

  int eTest = pEdgeCollapse[n];

  int Nmax = 3;      // alpha = 0, 1/2, 1
  int v1, v2;        // Vertices belonging to eTest
  int2 tCollapse;    // Triangles neighbouring eTest

  real v1x, v1y, v2x, v2y;
  real2 cSegment1, cSegment2;
  int2 minMaxAlpha = MinMaxAlpha(Nmax, eTest, pTv, pTe, pEt,
                                 nVertex, pVc, Px, Py,
                                 pred, pParam, tCollapse,
                                 v1, v2, v1x, v1y, v2x, v2y,
                                 cSegment1, cSegment2);

#ifndef __CUDA_ARCH__
  int printFlag = 0;
  if (printFlag == 1) {
    std::cout << "Testing collapse of edge " << eTest
              << " with vertices " << v1 << " = (" << v1x
              << ", " << v1y << ") and " << v2
              << " = (" << v2x << ", " << v2y << ") "
              << std::endl;
    if (tCollapse.y == -1)
      std::cout << "  Segment coordinates: (" << cSegment1.x
                << ", " << cSegment1.y << ") and ("
                << cSegment2.x << ", " << cSegment2.y << ")"
                << std::endl;
    std::cout << "  minMaxAlpha: " << minMaxAlpha.x
              << " " << minMaxAlpha.y << std::endl;
  }
#endif

  int validCollapse = 0;
  real xTarget, yTarget;
  for (int i = minMaxAlpha.x; i < minMaxAlpha.y; i++) {
    validCollapse = 1;

    real alpha = (real) i/((real) Nmax - 1.0);

    // Edge will be collapsed into single point
    xTarget = (1.0 - alpha)*v1x + alpha*v2x;
    yTarget = (1.0 - alpha)*v1y + alpha*v2y;

#ifndef __CUDA_ARCH__
    if (printFlag == 1)
      std::cout << "  Target coordinates: (" << xTarget
                << ", " << yTarget << ")" << std::endl;
#endif

    int eCross = eTest;                   // Current edge to cross
    int t = tCollapse.x;                  // Current triangle
    int isSegment = (tCollapse.y == -1);  // Flag if eTest is segment

    while (1) {
      int3 E = pTe[t];
      int3 V = pTv[t];

      if (t != tCollapse.x && t != tCollapse.y) {
        real ax, bx, cx, ay, by, cy;
        GetTriangleCoordinates(pVc, V.x, V.y, V.z,
                               nVertex, Px, Py, ax, bx, cx, ay, by, cy);

        int a = NormalizeVertex(v1, nVertex);
        int b = NormalizeVertex(v2, nVertex);
        int3 A = NormalizeVertex(V, nVertex);

        // Triangle has either v1 or v2 as a vertex
        int vCommon = v1;
        if (A.x == b || A.y == b || A.z == b) vCommon = v2;

        TranslateTriangleToVertex(vCommon, Px, Py, nVertex, V.x, V.y, V.z,
                                  ax, ay, bx, by, cx, cy);

        // Replace vertices
        if (A.x == a || A.x == b) {
          ax = xTarget;
          ay = yTarget;
        }
        if (A.y == a || A.y == b) {
          bx = xTarget;
          by = yTarget;
        }
        if (A.z == a || A.z == b) {
          cx = xTarget;
          cy = yTarget;
        }

        // Check if new triangle has positive area
        real det = pred->orient2d(ax, ay, bx, by, cx, cy, pParam);
        if (det <= zero) validCollapse = 0;

        // Also check quality of triangle
        int quality = TriangleQuality(ax, ay, bx, by, cx, cy, qualityBound);
        if (quality == 0) validCollapse = 0;

#ifndef __CUDA_ARCH__
        if (printFlag == 1)
          std::cout << "    Triangle " << t
                    << ", area: " << det
                    << ", quality: " << quality;
#endif

        // Check for encroached segments...
        if (tCollapse.y == -1) {
          int a = VertexNotPartOfEdge(eCross, V, E);
          real ax, ay;
          GetTriangleCoordinatesSingle(pVc, a, nVertex, Px, Py, ax, ay);

          TranslateTriangleToVertex(vCommon, Px, Py, nVertex, V.x, V.y, V.z,
                                    ax, ay, bx, by, cx, cy);

          real dot1 =
            (cSegment1.x - ax)*(xTarget - ax) +
            (cSegment1.y - ay)*(yTarget - ay);
          real dot2 =
            (xTarget - ax)*(cSegment2.x - ax) +
            (yTarget - ay)*(cSegment2.y - ay);

          if (dot1 < zero || dot2 < zero) validCollapse = 0;

#ifndef __CUDA_ARCH__
          if (printFlag == 1)
            std::cout << ", dot products encroachment: "
                      << dot1 << " " << dot2;
#endif
        }

        if (tCollapse.y != -1) {
          if (i > 0 && i < Nmax - 1) {
            // Check if new triangle encroaches upon one of its own

            int e = NextEdgeClockwise(eCross, E);
#ifndef __CUDA_ARCH__
            if (printFlag == 1)
              std::cout << ", self-edge: "
                          << e;
#endif
            int2 T = pEt[e];
            if (T.x == -1 || T.y == -1) {
              if (T.x == -1) {
                T.x = T.y;
                T.y = -1;
              }
              int a, b;
              GetEdgeVertices(e, T.x, V, E, a, b);

              real ax, ay, bx, by;
              GetTriangleCoordinatesSingle(pVc, a, nVertex, Px, Py, ax, ay);
              GetTriangleCoordinatesSingle(pVc, b, nVertex, Px, Py, bx, by);

              TranslateTriangleToVertex(vCommon, Px, Py, nVertex, V.x, V.y, V.z,
                                        ax, ay, bx, by, cx, cy);

              real dot =
                (bx - xTarget)*(ax - xTarget) +
                (by - yTarget)*(ay - yTarget);

              if (dot < zero) validCollapse = 0;

#ifndef __CUDA_ARCH__
              if (printFlag == 1)
                std::cout << ", dot product self-encroachment: "
                          << dot;
#endif
            }
          }
        }

#ifndef __CUDA_ARCH__
        if (printFlag == 1)
          std::cout << std::endl;
#endif

      }

      // Update t, eCross
      WalkAroundEdge(eTest, isSegment, t, eCross, E, pTe, pEt);

      // Done if we reach first triangle
      if (t == tCollapse.x) break;

    }

    if (validCollapse == 1) break;
  }

  // We need: validCollapse and (xTarget, yTarget)
  if (validCollapse == 0)
    pEdgeCollapse[n] = -1;

  pEdgeCoordinates[eTest].x = xTarget;
  pEdgeCoordinates[eTest].y = yTarget;

#ifndef __CUDA_ARCH__
  //if (printFlag == 1) {
  //  int qq; std::cin >> qq;
  // }
#endif

}


//#########################################################################
//#########################################################################

void Coarsen::TestEdgeCollapse(Connectivity *connectivity,
                               Predicates *predicates,
                               const MeshParameter *mp)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();
  int nRemove = edgeCollapseList->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  edgeCoordinates->SetSize(nEdge);
  real2 *pEdgeCoordinates = edgeCoordinates->GetPointer();
  int *pEdgeCollapseList = edgeCollapseList->GetPointer();

  real qualityBound = mp->qualityBound;

  if (cudaFlag == 1) {
    /*
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devValidEdgeCollapse,
       (size_t) 0, 0);

    devValidEdgeCollapse<<<nBlocks, nThreads>>>
      (nVertex, pTv, pTe, pEt, pVc, Px, Py,
       predicates, pParam, pVertexTriangle, pVertexMoveFlag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    */
  } else {
    for (int n = 0; n < nRemove; n++)
      TestEdgeCollapseSingle(n, pTv, pTe, pEt,
                             nVertex, pVc, Px, Py,
                             predicates, pParam,
                             pEdgeCollapseList,
                             pEdgeCoordinates,
                             qualityBound);
  }

  nRemove = edgeCollapseList->RemoveValue(-1);
  edgeCollapseList->SetSize(nRemove);
}

}
