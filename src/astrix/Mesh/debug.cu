// -*-c++-*-
/*! \file debug.cu
\brief Functions for debugging mesh generation

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "Predicates/predicates.h"
#include "./mesh.h"
#include "./triangleLow.h"
#include "../Common/inlineMath.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//##############################################################################
/*! Return maximum edge length for triangle \a i
  \param i Triangle number for which maximum edge length is required */
//##############################################################################

real Mesh::MaxEdgeLengthTriangle(int i)
{
  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;

  int nVertex = connectivity->vertexCoordinates->GetSize();

  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  // Three edges
  real l1 = sqrt(Sq(ax - bx) + Sq(ay - by));
  real l2 = sqrt(Sq(ax - cx) + Sq(ay - cy));
  real l3 = sqrt(Sq(cx - bx) + Sq(cy - by));

  return std::max(l1, std::max(l2, l3));
}

//##############################################################################
/*! Return maximum edge length of Mesh.*/
//##############################################################################

real Mesh::MaximumEdgeLength()
{
  real maxEdgeLength = 0.0;
  int nTriangle = connectivity->triangleVertices->GetSize();

  // Copy data to host
  if (cudaFlag == 1) connectivity->CopyToHost();

  for (int i = 0; i < nTriangle; i++) {
    real lMax = MaxEdgeLengthTriangle(i);
    maxEdgeLength = std::max(maxEdgeLength, lMax);
  }

  return maxEdgeLength;
}

//##############################################################################
/*! Check if any edge is larger than \a maxEdgeLength. Used for debugging: inserting new vertices should not increase the maximum edge length. If an edge is detected that is too large the program halts.
\param maxEdgeLength Edge length to compare with*/
//##############################################################################

void Mesh::CheckEdgeLength(real maxEdgeLength)
{
  // Copy data to host
  if (cudaFlag == 1) connectivity->CopyToHost();

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  for (int i = 0; i < nTriangle; i++) {
    real lMax = MaxEdgeLengthTriangle(i);
    if (lMax > maxEdgeLength) {
      int a = pTv[i].x;
      int b = pTv[i].y;
      int c = pTv[i].z;

      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c, nVertex, Px, Py,
                             ax, bx, cx, ay, by, cy);
      std::cout << "Warning: edge of triangle " << i
                <<  " larger than previous maximum: "
                << lMax << " " << maxEdgeLength << std::endl;
      std::cout << "Coordinates: " << ax << " " << ay << " "
                << bx << " " << by << " "
                << cx << " " << cy << std::endl;

      while (a >= nVertex) a -= nVertex;
      while (b >= nVertex) b -= nVertex;
      while (c >= nVertex) c -= nVertex;
      while (a < 0) a += nVertex;
      while (b < 0) b += nVertex;
      while (c < 0) c += nVertex;

      std::cout << "Vertices: " << pTv[i].x << " (" << a << ") "
                << pTv[i].y << " (" << b << ") "
                << pTv[i].z << " (" << c << ")" << std::endl;

      std::cout << "Dumping mesh with save index 999" << std::endl;
      connectivity->Save(999);

      throw std::runtime_error("");
    }
  }
}

//#########################################################################
// Output mesh stats to stdout
//#########################################################################

void Mesh::OutputStat()
{
  // Copy from device
  if (cudaFlag == 1) connectivity->CopyToHost();

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  std::cout << "Number of vertices: "  << nVertex   << std::endl;
  std::cout << "Number of edges: "     << nEdge     << std::endl;
  std::cout << "Number of triangles: " << nTriangle << std::endl;

  real minsin = 1.0;
  for (int i = 0; i < nTriangle; i++) {
    int A = pTv[i].x;
    int B = pTv[i].y;
    int C = pTv[i].z;

    real Ax, Bx, Cx, Ay, By, Cy;
    GetTriangleCoordinates(pVc, A, B, C, nVertex, Px, Py,
                           Ax, Bx, Cx, Ay, By, Cy);

    real area = 0.5*((Ax - Cx)*(By - Cy) - (Ay - Cy)*(Bx - Cx));

    real a = sqrt((Bx - Cx)*(Bx - Cx) + (By - Cy)*(By - Cy));
    real b = sqrt((Ax - Cx)*(Ax - Cx) + (Ay - Cy)*(Ay - Cy));
    real c = sqrt((Bx - Ax)*(Bx - Ax) + (By - Ay)*(By - Ay));

    real sintmin = 2.0*area*std::min(a, std::min(b, c))/(a*b*c);
    if (sintmin < minsin) minsin = sintmin;
  }

  std::cout << "Minimum angle: " << asin(minsin)*180.0/M_PI << std::endl;
  std::cout << "Circumradius to smallest edge ratio: " << 0.5/minsin
            << std::endl;
  std::cout << std::endl;
}

//#########################################################################
/*! Slow but sure way to detect any encroached segments. Used for debugging
purposes; runs only on host*/
//#########################################################################

void Mesh::CheckEncroachSlow()
{
  if (cudaFlag == 1) connectivity->CopyToHost();

  const real zero  = (real) 0.0;

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();
  int3 *pTe = connectivity->triangleEdges->GetHostPointer();
  int2 *pEt = connectivity->edgeTriangles->GetHostPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  for (int v = 0; v < nVertex; v++) {
    real x = pVc[v].x;
    real y = pVc[v].y;

    for (int t = 0; t < nTriangle; t++) {
      int A = pTv[t].x;
      int B = pTv[t].y;
      int C = pTv[t].z;

      real Ax, Bx, Cx, Ay, By, Cy;
      GetTriangleCoordinates(pVc, A, B, C, nVertex, Px, Py,
                             Ax, Bx, Cx, Ay, By, Cy);

      while (A >= nVertex) A -= nVertex;
      while (B >= nVertex) B -= nVertex;
      while (C >= nVertex) C -= nVertex;
      while (A < 0) A += nVertex;
      while (B < 0) B += nVertex;
      while (C < 0) C += nVertex;

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;

      real ux = zero, vx = zero, uy = zero, vy = zero;
      int doFlag = -1;
      if (t11 == -1 || t21 == -1) {
        if (v != A && v != B) {
          ux = Ax;
          uy = Ay;
          vx = Bx;
          vy = By;
          doFlag = 1;
        }
      }
      if (t12 == -1 || t22 == -1) {
        if (v != B && v != C) {
          ux = Bx;
          uy = By;
          vx = Cx;
          vy = Cy;
          doFlag = 1;
        }
      }
      if (t13 == -1 || t23 == -1) {
        if (v != C && v != A) {
          ux = Cx;
          uy = Cy;
          vx = Ax;
          vy = Ay;
          doFlag = 1;
        }
      }

      if (doFlag != -1) {
        real dot = (ux - x)*(vx - x) + (uy - y)*(vy - y);
        if (meshParameter->periodicFlagY == 1) {
          uy += Py;
          vy += Py;
          dot = min(dot, (ux - x)*(vx - x) + (uy - y)*(vy - y));

          uy -= 2.0*Py;
          vy -= 2.0*Py;
          dot = min(dot, (ux - x)*(vx - x) + (uy - y)*(vy - y));

          uy += Py;
          vy += Py;
        }
        if (meshParameter->periodicFlagX == 1) {
          ux += Px;
          vx += Px;
          dot = min(dot, (ux - x)*(vx - x) + (uy - y)*(vy - y));

          ux -= 2.0*Px;
          vx -= 2.0*Px;
          dot = min(dot, (ux - x)*(vx - x) + (uy - y)*(vy - y));

          ux += Px;
          vx += Px;
        }

        if (dot < 0.0f) {
          std::cout << "Error: vertex " << v
                    << " with x = " << x << ", y = " << y
                    << " encroaches upon segment in triangle " << t
                    << std::endl;

          std::cout << "Dumping mesh with save index 999" << std::endl;
          connectivity->Save(999);

          throw std::runtime_error("");
        }
      }
    }
  }
}

}  // namespace astrix
