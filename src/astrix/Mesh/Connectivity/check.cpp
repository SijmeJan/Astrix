// -*-c++-*-
/*! \file check.cpp
\brief Functions for checking validity of Connectivity object

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./connectivity.h"
#include "../Param/meshparameter.h"
#include "../triangleLow.h"
#include "../Predicates/predicates.h"

namespace astrix {

//#########################################################################
/*! Check if the neighbouring triangles of every edge \a e do have \a e as an edge. This is done on the host. An exception is thrown if an invalid edge is detected.*/
//#########################################################################

void Connectivity::CheckEdgeTriangles()
{
  // Copy data to host if necessary
  if (cudaFlag == 1) CopyToHost();

  int2 *pEt = edgeTriangles->GetHostPointer();
  int3 *pTe = triangleEdges->GetHostPointer();

  int nEdge = edgeTriangles->GetSize();

  for (int i = 0; i < nEdge; i++) {
    int t1 = pEt[i].x;
    int t2 = pEt[i].y;

    if (t1 != -1) {
      int e1 = pTe[t1].x;
      int e2 = pTe[t1].y;
      int e3 = pTe[t1].z;

      if (i != e1 && i != e2 && i != e3) {
        std::cout << "Edge " << i << " has neighbouring triangle "
                  << t1 << " but triangle " << t1 << " does not have an edge "
                  << i << std::endl;

        std::cout << "Dumping mesh with save index 999" << std::endl;
        Save(999);

        throw std::runtime_error("");
      }
    }

    if (t2 != -1) {
      int e1 = pTe[t2].x;
      int e2 = pTe[t2].y;
      int e3 = pTe[t2].z;

      if (i != e1 && i != e2 && i != e3) {
        std::cout << "Edge " << i << " has neighbouring triangle "
                  << t2 << " but triangle " << t2 << " does not have an edge "
                  << i << std::endl;

        std::cout << "Dumping mesh with save index 999" << std::endl;
        Save(999);

        throw std::runtime_error("");
      }
    }
  }
}

//#########################################################################
/*! Check all triangles for negative areas. If a negative area is detected an exception is thrown.

\param *predicates Pointer to Predicates object
\param *mp Pointer to MeshParameter object*/
//#########################################################################

void Connectivity::CheckTriangleAreas(const Predicates *predicates,
                                      const MeshParameter *mp)
{
  // Copy data to host if necessary
  if (cudaFlag == 1) CopyToHost();

  int3 *pTv = triangleVertices->GetHostPointer();
  real2 *pVc = vertexCoordinates->GetHostPointer();

  int nVertex = vertexCoordinates->GetSize();
  int nTriangle = triangleVertices->GetSize();

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  real *pParam = predicates->GetParamPointer(0);

  for (int i = 0; i < nTriangle; i++) {
    int a = pTv[i].x;
    int b = pTv[i].y;
    int c = pTv[i].z;

    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    //real area = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
    real area = predicates->orient2d(ax, ay, bx, by, cx, cy, pParam);

    if (area <= 0.0) {
      std::cout << "Triangle " << i << " with vertices "
                << a << " " << b << " " << c << " has negative area "
                << area << std::endl;
      std::cout << "Dumping mesh with save index 999" << std::endl;
      Save(999);

      throw std::runtime_error("");
   }
  }
}

//#########################################################################
/*! Check for encroached segments

\param *mp Pointer to MeshParameter object*/
//#########################################################################

void Connectivity::CheckEncroach(const MeshParameter *mp)
{
  // Copy data to host if necessary
  if (cudaFlag == 1) CopyToHost();

  int3 *pTv = triangleVertices->GetHostPointer();
  real2 *pVc = vertexCoordinates->GetHostPointer();
  int3 *pTe = triangleEdges->GetHostPointer();
  int2 *pEt = edgeTriangles->GetHostPointer();

  int nVertex = vertexCoordinates->GetSize();
  int nEdge = edgeTriangles->GetSize();

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  int error = 0;

  // Check all edges
  for (int i = 0; i < nEdge; i++) {
    // Only proceed if edge has only one neighbouring triangle (= t)
    int t1 = pEt[i].x;
    int t2 = pEt[i].y;

    int t = -1;
    if (t2 == -1) t = t1;
    if (t1 == -1) t = t2;

    if (t != -1) {
      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;

      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c,
                             nVertex, Px, Py,
                             ax, bx, cx, ay, by, cy);

      // Is edge between a and b encroached by c?
      real dot1 = (ax - cx)*(bx - cx) + (ay - cy)*(by - cy);
      // Is edge between b and c encroached by a?
      real dot2 = (bx - ax)*(cx - ax) + (by - ay)*(cy - ay);
      // Is edge between a and c encroached by b?
      real dot3 = (cx - bx)*(ax - bx) + (cy - by)*(ay - by);

      int3 E = pTe[t];

      if ((E.x == i && dot1 < 0.0) ||
          (E.y == i && dot2 < 0.0) ||
          (E.z == i && dot3 < 0.0)) {
        std::cout << "Edge " << i << " is an encroached segment! "
                  << std::endl;
        error = 1;
      }
    }
  }

  if (error == 1) {
    std::cout << "Dumping mesh with save index 999" << std::endl;
    Save(999);

    throw std::runtime_error("");
  }
}

}  // namespace astrix
