// -*-c++-*-
/*! \file save.cpp
\brief File containing functions to save and restore mesh to and from disk.

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
#include "./connectivity.h"

namespace astrix {

//#########################################################################
/*! Save vertexCoordinates in a vertex file, triangleVertices and triangleEdges in a triangle file, and edgeTriangles in an edge file.

\param nSave Number of save, used to generate file names*/
//#########################################################################

void Connectivity::Save(int nSave)
{
  int nTriangle = triangleVertices->GetSize();
  int nVertex = vertexCoordinates->GetSize();
  int nEdge = edgeTriangles->GetSize();

  // Copy data to host
  if (cudaFlag == 1) CopyToHost();

  int3 *pTv = triangleVertices->GetHostPointer();

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

  int3 *pTe = triangleEdges->GetHostPointer();

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

  int2 *pEt = edgeTriangles->GetHostPointer();

  // Separate into separate arrays for writing
  Array<int> *edgeTrianglesSeparate = new Array<int>(2, 0, nEdge);
  int *et1 = edgeTrianglesSeparate->GetHostPointer(0);
  int *et2 = edgeTrianglesSeparate->GetHostPointer(1);

  for (int i = 0; i < nEdge; i++) {
    et1[i] = pEt[i].x;
    et2[i] = pEt[i].y;
  }

  real2 *pVc = vertexCoordinates->GetHostPointer();

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

//#########################################################################
/*! Read mesh from disk as it was saved under number \a nSave. In addition, calculate triangle normals, vertex areas and find boundary vertices.

  \param nSave Number of save to restore*/
//#########################################################################

void Connectivity::ReadFromDisk(int nSave)
{
  int nTriangle = triangleVertices->GetSize();
  int nVertex = vertexCoordinates->GetSize();
  int nEdge = edgeTriangles->GetSize();

  std::cout << "Reading mesh from disk..." << std::endl;
  char fname[13];

  if (nSave == -1) {
    std::ifstream inFile;
    inFile.open("lastsave.dat");
    if (!inFile) {
      std::cout << "Could not open lastsave.dat!" << std::endl;
      throw std::runtime_error("");
    }
    inFile >> nSave;
    inFile.close();
  }

  // Open vertex file
  snprintf(fname, sizeof(fname), "vert%4.4d.dat", nSave);
  std::ifstream vertexInFile(fname, std::ios::binary);
  if (!vertexInFile) {
    std::cout << "Could not open vertex file " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Number of dimensions
  int ndim;
  vertexInFile.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

  // Could be floats or doubles
  int sizeOfData;
  vertexInFile.read(reinterpret_cast<char*>(&sizeOfData), sizeof(sizeOfData));

  // Number of vertices in file
  vertexInFile.read(reinterpret_cast<char*>(&nVertex), sizeof(nVertex));

  // Precision wrong: exit
  if (sizeOfData != sizeof(real)) {
    std::cout << "  Vertex coordinates were saved in different precision: "
              << sizeOfData << std::endl;
    vertexInFile.close();
    throw std::runtime_error("");
  }

  std::cout << "  Running in " << ndim << " dimensions" << std::endl;
  std::cout << "  Number of vertices: " << nVertex << std::endl;

  // Read vertex coordinates
  vertexCoordinates->SetSizeHost(nVertex);
  real2 *pVc = vertexCoordinates->GetHostPointer();

  Array<real> *vertexCoordinatesSeparate = new Array<real>(2, 0, nVertex);
  real *pVertX = vertexCoordinatesSeparate->GetHostPointer(0);
  real *pVertY = vertexCoordinatesSeparate->GetHostPointer(1);

  vertexInFile.read(reinterpret_cast<char*>(pVertX), nVertex*sizeOfData);
  vertexInFile.read(reinterpret_cast<char*>(pVertY), nVertex*sizeOfData);

  for (int i = 0; i < nVertex; i++) {
    pVc[i].x = pVertX[i];
    pVc[i].y = pVertY[i];
  }

  delete vertexCoordinatesSeparate;

  vertexInFile.close();

  // Read from triangle file
  snprintf(fname, sizeof(fname), "tria%4.4d.dat", nSave);
  std::ifstream triangleInFile(fname, std::ios::binary);
  if (!triangleInFile) {
    std::cout << "Could not open triangle file " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Number of triangles in file
  triangleInFile.read(reinterpret_cast<char*>(&nTriangle), sizeof(nTriangle));

  std::cout << "  Number of triangles: " << nTriangle << std::endl;

  // Triangle vertices
  triangleVertices->SetSizeHost(nTriangle);
  // Triangle edges
  triangleEdges->SetSizeHost(nTriangle);

  int3 *pTv = triangleVertices->GetHostPointer();

  Array<int> *triangleVerticesSeparate = new Array<int>(3, 0, nTriangle);
  int *tv1 = triangleVerticesSeparate->GetPointer(0);
  int *tv2 = triangleVerticesSeparate->GetPointer(1);
  int *tv3 = triangleVerticesSeparate->GetPointer(2);

  triangleInFile.read(reinterpret_cast<char*>(tv1), nTriangle*sizeof(int));
  triangleInFile.read(reinterpret_cast<char*>(tv2), nTriangle*sizeof(int));
  triangleInFile.read(reinterpret_cast<char*>(tv3), nTriangle*sizeof(int));

  for (int i = 0; i < nTriangle; i++) {
    pTv[i].x = tv1[i];
    pTv[i].y = tv2[i];
    pTv[i].z = tv3[i];
  }

  delete triangleVerticesSeparate;

  int3 *pTe = triangleEdges->GetHostPointer();

  Array<int> *triangleEdgesSeparate = new Array<int>(3, 0, nTriangle);
  int *te1 = triangleEdgesSeparate->GetHostPointer(0);
  int *te2 = triangleEdgesSeparate->GetHostPointer(1);
  int *te3 = triangleEdgesSeparate->GetHostPointer(2);

  triangleInFile.read(reinterpret_cast<char*>(te1), nTriangle*sizeof(int));
  triangleInFile.read(reinterpret_cast<char*>(te2), nTriangle*sizeof(int));
  triangleInFile.read(reinterpret_cast<char*>(te3), nTriangle*sizeof(int));

  for (int i = 0; i < nTriangle; i++) {
    pTe[i].x = te1[i];
    pTe[i].y = te2[i];
    pTe[i].z = te3[i];
  }

  delete triangleEdgesSeparate;

  triangleInFile.close();

  // Read from edge file
  snprintf(fname, sizeof(fname), "edge%4.4d.dat", nSave);
  std::ifstream edgeInFile(fname, std::ios::binary);
  if (!edgeInFile) {
    std::cout << "Could not open edge file " << fname << std::endl;
    throw std::runtime_error("");
  }

  // Number of edges in file
  edgeInFile.read(reinterpret_cast<char*>(&nEdge), sizeof(nEdge));

  std::cout << "  Number of edges: " << nEdge << std::endl;

  edgeTriangles->SetSizeHost(nEdge);
  int2 *pEt = edgeTriangles->GetHostPointer();

  Array<int> *edgeTrianglesSeparate = new Array<int>(3, 0, nEdge);
  int *et1 = edgeTrianglesSeparate->GetPointer(0);
  int *et2 = edgeTrianglesSeparate->GetPointer(1);

  edgeInFile.read(reinterpret_cast<char*>(et1), nEdge*sizeof(int));
  edgeInFile.read(reinterpret_cast<char*>(et2), nEdge*sizeof(int));

  for (int i = 0; i < nEdge; i++) {
    pEt[i].x = et1[i];
    pEt[i].y = et2[i];
  }

  delete edgeTrianglesSeparate;

  edgeInFile.close();

  if (cudaFlag == 1) CopyToDevice();
}

}  // namespace astrix
