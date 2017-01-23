// -*-c++-*-
/*! \file findparalleldeletion.cu
\brief File containing function to find parallel deletion set.

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
#include "./coarsen.h"
#include "../../Common/cudaLow.h"
#include "./unique_triangle.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#############################################################################
/*! \brief Find triangles that are affected by removing vertex \a pVertexRemove[n]

\param n Index in \a pVertexRemove to consider
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertexTriangle Pointer to array containing for every vertex a triangle sharing that vertex
\param *pVertexRemove Pointer to array containing vertices to be removed
\param nVertex Total number of vertices in Mesh
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param nRemove Total number of vertices to be removed
\param *pTriangleAffected Pointer to output array*/
//#############################################################################

__host__ __device__
void FillAffectedTrianglesSingle(int n, int3 *pTv, int3 *pTe, int2 *pEt,
                                 int *pVertexTriangle,
                                 int *pVertexRemove, int nVertex,
                                 int maxTriPerVert, int nRemove,
                                 int *pTriangleAffected)
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
    pTriangleAffected[i + n*maxTriPerVert] = t;

    int e1 = pTe[t].x;
    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    // Edge to cross to find neighbouring triangle: *next*
    int eNeighbour = -1;
    if (e == e1) eNeighbour = e2;
    if (e == e2) eNeighbour = e3;
    if (e == e3) eNeighbour = e1;

    int t1 = pEt[eNeighbour].x;
    int t2 = pEt[eNeighbour].y;

    int tNeighbour = t1;
    if (tNeighbour == t) tNeighbour = t2;

    pTriangleAffected[i + n*maxTriPerVert + nRemove*maxTriPerVert] =
      tNeighbour;

    // Move across edge e
    t1 = pEt[e].x;
    t2 = pEt[e].y;
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
      pTriangleAffected[i + n*maxTriPerVert] = t;

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      // Edge to cross to find neighbouring triangle: *previous*
      int eNeighbour = -1;
      if (e == e1) eNeighbour = e3;
      if (e == e2) eNeighbour = e1;
      if (e == e3) eNeighbour = e2;

      int t1 = pEt[eNeighbour].x;
      int t2 = pEt[eNeighbour].y;

      int tNeighbour = t1;
      if (tNeighbour == t) tNeighbour = t2;

      pTriangleAffected[i + n*maxTriPerVert + nRemove*maxTriPerVert] =
        tNeighbour;

      // Move across edge e
      t1 = pEt[e].x;
      t2 = pEt[e].y;
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

  // Check for duplicate triangles
  for (int j = 0; j < maxTriPerVert; j++) {
    int t1 = pTriangleAffected[j + (n + nRemove)*maxTriPerVert];
    int ret = 0;
    for (int k = 0; k < maxTriPerVert; k++) {
      int t2 = pTriangleAffected[k + (n + nRemove)*maxTriPerVert];
      int t3 = pTriangleAffected[k + n*maxTriPerVert];

      if (t1 == t3) ret = 1;
      if (j != k && t1 == t2) ret = 1;
    }
    if (ret == 1)
      pTriangleAffected[j + (n + nRemove)*maxTriPerVert] = -1;
  }
}

//######################################################################
/*! \brief Kernel finding triangles that are affected by removing vertex \a pVertexRemove

\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertexTriangle Pointer to array containing for every vertex a triangle sharing that vertex
\param *pVertexRemove Pointer to array containing vertices to be removed
\param nVertex Total number of vertices in Mesh
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param nRemove Total number of vertices to be removed
\param *pTriangleAffected Pointer to output array*/
//######################################################################

__global__ void
devFillAffectedTriangles(int3 *pTv, int3 *pTe, int2 *pEt,
                         int *pVertexTriangle, int *pVertexRemove, int nVertex,
                         int maxTriPerVert, int nRemove, int *pTriangleAffected)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FillAffectedTrianglesSingle(n, pTv, pTe, pEt,
                                pVertexTriangle, pVertexRemove, nVertex,
                                maxTriPerVert, nRemove, pTriangleAffected);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Kernel filling array \a pTriangleAffectedIndex

Removing a vertex affects the triangles sharing that vertex plus all the neighbours of that triangle. This results in 2*\a maxTriPerVert triangles maximum. The array \a pTriangleAffectedIndex relates these triangles to the vertices to be removed

\param nRemove Total number of vertices to be removed
\param maxTriPerVert Maximum number of traingles sharing any vertex
\param *pTriangleAffectedIndex Pointer to output array*/
//######################################################################

__global__ void
devFillAffectedIndex(int nRemove, int maxTriPerVert,
                     int *pTriangleAffectedIndex)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    for (int i = 0; i < maxTriPerVert; i++) {
      pTriangleAffectedIndex[i + n*maxTriPerVert] = n;
      pTriangleAffectedIndex[i + n*maxTriPerVert + nRemove*maxTriPerVert] = n;
    }

    n += blockDim.x*gridDim.x;
  }
}

//############################################################################
/*! \brief Fill array of triangles affected by vertex removal

Removing a vertex affects the triangles sharing that vertex plus all the neighbours of that triangle. This results in 2*\a maxTriPerVert triangles maximum, which are put in \a triangleAffected. The array \a triangleAffectedIndex relates these triangles to the vertices to be removed

\param *triangleAffected Pointer to output Array containing affected triangles
\param *triangleAffectedIndex Pointer to output Array containing index of the vertex relating to the affected triangles
\param *edgeTriangles Pointer to Array containing triangles neighbouring edges
\param *triangleEdges Pointer to Array containing edges belonging to triangles
\param *triangleVertices Pointer to Array containing vertices belonging to triangles
\param *vertexTriangle Pointer to Array containing for every vertex one triangle sharing that vertex
\param *vertexRemove Pointer to Array containing vertices to be removed
\param maxTriPerVert Maximum number of traingles sharing any vertex
\param nRemove Total number of vertices to be removed
\param cudaFlag Flag whether to do computations on CUDA device. Has to match \a cudaFlag of Arrays involved
\param nVertex Total number of vertices in Mesh*/
//############################################################################

void FillAffectedTriangles(Array<int> *triangleAffected,
                           Array<int> *triangleAffectedIndex,
                           Connectivity *connectivity,
                           Array<int> *vertexTriangle,
                           Array<int> *vertexRemove,
                           int maxTriPerVert,
                           int nRemove, int cudaFlag, int nVertex)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemove->TransformToHost();
      vertexTriangle->TransformToHost();
      triangleAffected->TransformToHost();
      triangleAffectedIndex->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemove->TransformToDevice();
      vertexTriangle->TransformToDevice();
      triangleAffected->TransformToDevice();
      triangleAffectedIndex->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int *pVertexTriangle = vertexTriangle->GetPointer();
  int *pVertexRemove = vertexRemove->GetPointer();

  int *pTriangleAffected = triangleAffected->GetPointer();
  int *pTriangleAffectedIndex = triangleAffectedIndex->GetPointer();

  // Fill neighbouring triangles array
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillAffectedIndex,
                                       (size_t) 0, 0);

    devFillAffectedIndex<<<nBlocks, nThreads>>>
      (nRemove, maxTriPerVert, pTriangleAffectedIndex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++) {
      for (int i = 0; i < maxTriPerVert; i++) {
        pTriangleAffectedIndex[i + n*maxTriPerVert] = n;
        pTriangleAffectedIndex[i + n*maxTriPerVert + nRemove*maxTriPerVert] = n;
      }
    }
  }

  // Fill array of affected triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillAffectedTriangles,
                                       (size_t) 0, 0);

    devFillAffectedTriangles<<<nBlocks, nThreads>>>
      (pTv, pTe, pEt, pVertexTriangle, pVertexRemove, nVertex,
       maxTriPerVert, nRemove, pTriangleAffected);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FillAffectedTrianglesSingle(n, pTv, pTe, pEt,
                                  pVertexTriangle, pVertexRemove, nVertex,
                                  maxTriPerVert, nRemove, pTriangleAffected);
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemove->TransformToHost();
      vertexTriangle->TransformToHost();
      triangleAffected->TransformToHost();
      triangleAffectedIndex->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemove->TransformToDevice();
      vertexTriangle->TransformToDevice();
      triangleAffected->TransformToDevice();
      triangleAffectedIndex->TransformToDevice();

      cudaFlag = 1;
    }
  }
}

//#############################################################################
/*! Find set of vertices that can be removed in parallel. First, we create a list of triangles that will be affected by vertex removal. Then, we find the unique values and compact Arrays \a vertexRemove and \a vertexTriangle

\param maxTriPerVert Maximum number of triangles sharing a vertex*/
//#############################################################################

void Coarsen::FindParallelDeletionSet(Connectivity *connectivity,
                                      int maxTriPerVert)
{
  unsigned int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  // Shuffle points to add to maximise parallelisation
  Array<unsigned int> *randomNumbers =
    new Array<unsigned int>(1, cudaFlag, nRemove);

  randomNumbers->SetEqual(randomVector);
  Array<unsigned int> *randomPermutation =
    new Array<unsigned int>(1, cudaFlag, nRemove);
  randomPermutation->SetToSeries();
  unsigned int *pRandomPermutation = randomPermutation->GetPointer();

  randomNumbers->SortByKey(randomPermutation);

  vertexRemove->Reindex(pRandomPermutation);
  vertexTriangle->Reindex(pRandomPermutation);

  delete randomNumbers;
  delete randomPermutation;

  Array <int> *triangleAffected =
    new Array<int>(1, cudaFlag, (unsigned int) (2*nRemove*maxTriPerVert));
  triangleAffected->SetToValue(-1);

  Array <int> *triangleAffectedIndex =
    new Array<int>(1, cudaFlag, (unsigned int) (2*nRemove*maxTriPerVert));

  FillAffectedTriangles(triangleAffected,
                        triangleAffectedIndex,
                        connectivity,
                        vertexTriangle,
                        vertexRemove,
                        maxTriPerVert,
                        nRemove, cudaFlag, nVertex);

  // Sort triangleAffected, together with triangleAffectedIndex
  // In case of equal, use triangleAffectedIndex
  triangleAffected->Sort(triangleAffectedIndex);

  // Find unique values
  Array <int> *uniqueFlag =
    new Array<int>(1, cudaFlag, nRemove);
  Array <int> *uniqueFlagScan =
    new Array<int>(1, cudaFlag, nRemove);

  // Find unique values
  FindUniqueTriangleAffected(triangleAffected,
                             triangleAffectedIndex,
                             uniqueFlag,
                             2*nRemove*maxTriPerVert, cudaFlag);

  // Compact arrays to new nRefine
  nRemove = uniqueFlag->ExclusiveScan(uniqueFlagScan, nRemove);
  vertexRemove->Compact(nRemove, uniqueFlag, uniqueFlagScan);
  vertexTriangle->Compact(nRemove, uniqueFlag, uniqueFlagScan);

  delete triangleAffected;
  delete triangleAffectedIndex;

  delete uniqueFlag;
  delete uniqueFlagScan;
}

}
