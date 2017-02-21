// -*-c++-*-
/*! \file save.cpp
\brief File containing functions to save and restore mesh to and from disk during refining.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../mesh.h"
#include "../Connectivity/connectivity.h"
#include "./refine.h"

namespace astrix {

//#########################################################################
/*! Save vertexCoordinates in a vertex file, triangleVertices and triangleEdges in a triangle file, and edgeTriangles in an edge file.

\param nSave Number of save, used to generate file names*/
//#########################################################################

void Refine::Save(int nSave,
                  Connectivity * const connectivity)
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  // Copy data to host
  if (cudaFlag == 1) connectivity->CopyToHost();

  int3 *pTv = connectivity->triangleVertices->GetHostPointer();

  // Separate into separate arrays for writing
  Array<int> *triangleVerticesSeparate = new Array<int>(3, 0, nTriangle);
  int *tv1 = triangleVerticesSeparate->GetPointer(0);
  int *tv2 = triangleVerticesSeparate->GetPointer(1);
  int *tv3 = triangleVerticesSeparate->GetPointer(2);

  for (int i = 0; i < nTriangle; i++) {
    tv1[i] = pTv[i].x;
    tv2[i] = pTv[i].y;
    tv3[i] = pTv[i].z;
  }

  int3 *pTe = connectivity->triangleEdges->GetHostPointer();

  // Separate into separate arrays for writing
  Array<int> *triangleEdgesSeparate = new Array<int>(3, 0, nTriangle);
  int *te1 = triangleEdgesSeparate->GetHostPointer(0);
  int *te2 = triangleEdgesSeparate->GetHostPointer(1);
  int *te3 = triangleEdgesSeparate->GetHostPointer(2);

  for (int i = 0; i < nTriangle; i++) {
    te1[i] = pTe[i].x;
    te2[i] = pTe[i].y;
    te3[i] = pTe[i].z;
  }

  int2 *pEt = connectivity->edgeTriangles->GetHostPointer();

  // Separate into separate arrays for writing
  Array<int> *edgeTrianglesSeparate = new Array<int>(2, 0, nEdge);
  int *et1 = edgeTrianglesSeparate->GetHostPointer(0);
  int *et2 = edgeTrianglesSeparate->GetHostPointer(1);

  for (int i = 0; i < nEdge; i++) {
    et1[i] = pEt[i].x;
    et2[i] = pEt[i].y;

    if (i == 76100)
      std::cout << i << " " << et1[i] << " " << et2[i] << std::endl;
  }

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();

  // Separate into separate arrays for writing
  Array<real> *vertexCoordinatesSeparate = new Array<real>(2, 0, nVertex);
  real *pVertX = vertexCoordinatesSeparate->GetHostPointer(0);
  real *pVertY = vertexCoordinatesSeparate->GetHostPointer(1);

  for (int i = 0; i < nVertex; i++) {
    pVertX[i] = pVc[i].x;
    pVertY[i] = pVc[i].y;
  }

  int ndim = 2;
  char fname[13];

  // File containing vertices
  snprintf(fname, sizeof(fname), "vert%4.4d.dat", nSave);
  std::ofstream vout(fname, std::ios::binary);

  // Number of dimensions
  vout.write(reinterpret_cast<char*>(&ndim), sizeof(ndim));

  // Size of binary data
  int sizeOfData = sizeof(real);
  vout.write(reinterpret_cast<char*>(&sizeOfData), sizeof(sizeOfData));

  // Total number of vertices
  vout.write(reinterpret_cast<char*>(&nVertex), sizeof(nVertex));

  // Output vertex coordinates
  vout.write(reinterpret_cast<char*>(pVertX), nVertex*sizeof(real));
  vout.write(reinterpret_cast<char*>(pVertY), nVertex*sizeof(real));

  vout.close();

  // File containing triangles
  snprintf(fname, sizeof(fname), "tria%4.4d.dat", nSave);
  std::ofstream tout(fname, std::ios::binary);

  // Number of triangles
  tout.write(reinterpret_cast<char*>(&nTriangle), sizeof(nTriangle));

  // Output vertices and edges belonging to triangle
  tout.write(reinterpret_cast<char*>(tv1), nTriangle*sizeof(int));
  tout.write(reinterpret_cast<char*>(tv2), nTriangle*sizeof(int));
  tout.write(reinterpret_cast<char*>(tv3), nTriangle*sizeof(int));
  tout.write(reinterpret_cast<char*>(te1), nTriangle*sizeof(int));
  tout.write(reinterpret_cast<char*>(te2), nTriangle*sizeof(int));
  tout.write(reinterpret_cast<char*>(te3), nTriangle*sizeof(int));

  tout.close();

  // File containing edges
  snprintf(fname, sizeof(fname), "edge%4.4d.dat", nSave);
  std::ofstream eout(fname, std::ios::binary);

  // Number of edges
  eout.write(reinterpret_cast<char*>(&nEdge), sizeof(nEdge));

  // Output vertices, triangles belonging to edge, and boundary flag
  eout.write(reinterpret_cast<char*>(et1), nEdge*sizeof(int));
  eout.write(reinterpret_cast<char*>(et2), nEdge*sizeof(int));

  eout.close();

  delete triangleVerticesSeparate;
  delete triangleEdgesSeparate;
  delete edgeTrianglesSeparate;
  delete vertexCoordinatesSeparate;
}

}  // namespace astrix
