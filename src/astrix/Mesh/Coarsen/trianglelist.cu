// -*-c++-*-
/*! \file trianglelist.cu
\brief Functions for creating list of triangles sharing a vertex.

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
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! \brief Create list of triangles sharing vertex \a v = \a pVertRemove[n]

We know that \a pVertexTriangle[n] contains a triangle sharing \a v. Starting from this triangle, we circle around \a v listing all triangles we encounter. Result is stored in \a pVertexTriangleList

\param n Index of vertex in \a pVertexRemove to consider
\param *pVertexTriangle Pointer to array of single triangles that have \a pVertexRemove as a vertex. i.e. pVertexRemove[i] is part of triangle pVertexTriangle[i]
\param *pVertexRemove Pointer to array of vertices to be removed
\param nVertex Total number of vertices in Mesh
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *pVertexTriangleList Pointer to list of triangles sharing vertex (output)*/
//#########################################################################

__host__ __device__
void FillVertexTriangleListSingle(int n, int *pVertexTriangle,
                                  int *pVertexRemove, int nVertex,
                                  int3 *pTv, int3 *pTe, int2 *pEt,
                                  int maxTriPerVert, int *pVertexTriangleList)
{
  int tStart = pVertexTriangle[n];
  int v = pVertexRemove[n];

  int a = pTv[tStart].x;
  int b = pTv[tStart].y;
  int c = pTv[tStart].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = pTe[tStart].x;
  int e2 = pTe[tStart].y;
  int e3 = pTe[tStart].z;

  int eStart = e1;
  if (b == v) eStart = e2;
  if (c == v) eStart = e3;

  int t = tStart;
  int e = eStart;
  int tNext = -1;
  int eNext = -1;
  int i = 0;

  // Move in clockwise direction around v
  int finished = 0;
  while (finished == 0 && i < maxTriPerVert) {
    pVertexTriangleList[i + n*maxTriPerVert] = t;

    // Move across edge e
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    tNext = t1;
    if (tNext == t) tNext = t2;

    if (tNext == -1 || tNext == tStart) {
      finished = 1;
    } else {
      t = tNext;
      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      // Move across *next* edge
      if (e == e1) eNext = e2;
      if (e == e2) eNext = e3;
      if (e == e3) eNext = e1;
      e = eNext;
    }
    i++;
  }

  if (tNext == -1) {
    int e1 = pTe[tStart].x;
    int e2 = pTe[tStart].y;
    int e3 = pTe[tStart].z;
    eStart = e3;
    if (b == v) eStart = e1;
    if (c == v) eStart = e2;

    t = tStart;
    e = eStart;
    eNext = -1;

    // Move across edge e
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    tNext = t1;
    if (tNext == t) tNext = t2;

    if (tNext == -1 || tNext == tStart) {
      i = maxTriPerVert;
    } else {
      t = tNext;
      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      // Move across *previous* edge
      if (e == e1) eNext = e3;
      if (e == e2) eNext = e1;
      if (e == e3) eNext = e2;
      e = eNext;
    }

    // Move in counterclockwise direction around v
    while (i < maxTriPerVert) {
      pVertexTriangleList[i + n*maxTriPerVert] = t;

      // Move across edge e
      int t1 = pEt[e].x;
      int t2 = pEt[e].y;
      tNext = t1;
      if (tNext == t) tNext = t2;

      if (tNext == -1 || tNext == tStart) {
        i = maxTriPerVert;
      } else {
        t = tNext;
        int e1 = pTe[t].x;
        int e2 = pTe[t].y;
        int e3 = pTe[t].z;
        // Move across *previous* edge
        if (e == e1) eNext = e3;
        if (e == e2) eNext = e1;
        if (e == e3) eNext = e2;
        e = eNext;
      }
      i++;
    }
  }
}

//#########################################################################
/*! \brief Kernel creating list of triangles sharing vertex \a v = \a pVertRemove[n]

We know that \a pVertexTriangle[n] contains a triangle sharing \a v. Starting from this triangle, we circle around \a v listing all triangles we encounter. Result is stored in \a pVertexTriangleList

\param nRemove Total number of vertices to be removed
\param *pVertexTriangle Pointer to array of single triangles that have \a pVertexRemove as a vertex. i.e. pVertexRemove[i] is part of triangle pVertexTriangle[i]
\param *pVertexRemove Pointer to array of vertices to be removed
\param nVertex Total number of vertices in Mesh
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *pVertexTriangleList Pointer to list of triangles sharing vertex (output)*/
//#########################################################################

__global__
void devFillVertexTriangleList(int nRemove, int *pVertexTriangle,
                               int *pVertexRemove, int nVertex,
                               int3 *pTv, int3 *pTe, int2 *pEt,
                               int maxTriPerVert, int *pVertexTriangleList)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FillVertexTriangleListSingle(n, pVertexTriangle, pVertexRemove,
                                 nVertex, pTv, pTe, pEt,
                                 maxTriPerVert, pVertexTriangleList);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! We know that \a vertexTriangle[n] contains a triangle sharing vertex \a v = \a vertexRemove[n]. Starting from this triangle, we circle around \a v listing all triangles we encounter. Result is stored in \a vertexTriangleList

\param *vertexTriangleList Pointer to output Array (size nRemove*maxTriPerVert)
\param maxTriPerVert Maximum number of triangles sharing single vertex */
//#########################################################################

void Coarsen::CreateVertexTriangleList(Connectivity *connectivity,
                                       Array<int> *vertexTriangleList,
                                       int maxTriPerVert)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangle->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexTriangle->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();

  int *pVertexTriangle = vertexTriangle->GetPointer();
  int *pVertexTriangleList = vertexTriangleList->GetPointer();
  int *pVertexRemove = vertexRemove->GetPointer();
  int nRemove = vertexRemove->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  // Create list of triangles associated with each vertex
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillVertexTriangleList,
                                       (size_t) 0, 0);

    devFillVertexTriangleList<<<nBlocks, nThreads>>>
      (nRemove, pVertexTriangle, pVertexRemove,
       nVertex, pTv, pTe, pEt,
       maxTriPerVert, pVertexTriangleList);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FillVertexTriangleListSingle(n, pVertexTriangle, pVertexRemove,
                                   nVertex, pTv, pTe, pEt,
                                   maxTriPerVert, pVertexTriangleList);
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangle->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexTriangle->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

}

}
