// -*-c++-*-
/*! \file remove.cu
\brief Functions for removing vertices from Mesh.

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
#include "../triangleLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
//#########################################################################

__host__ __device__
void AdjustEdgeNeedsCheckingSingle(int n, int *pEdgeNeedsChecking)
{
  if (pEdgeNeedsChecking[n] != -1)
    pEdgeNeedsChecking[n] = n;
}

//#########################################################################
//#########################################################################

__global__
void devAdjustEdgeNeedsChecking(int neKeep, int *pEdgeNeedsChecking)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < neKeep) {
    AdjustEdgeNeedsCheckingSingle(n, pEdgeNeedsChecking);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! \brief Adjust \a edgeTriangles for removed triangles

\param n Edge index to consider
\param *et1 Pointer to array of first neighbouring triangle
\param *et2 Pointer to array of second neighbouring triangle
\param *pTriangleKeepFlagScan Scanned array of flags whether triangles are to be kept*/
//#########################################################################

__host__ __device__
void AdjustEdgeSingle(int n, int2 *pEt, int *pTriangleKeepFlagScan)
{
  int a = pEt[n].x;
  int b = pEt[n].y;
  if (a != -1) a = pTriangleKeepFlagScan[a];
  if (b != -1) b = pTriangleKeepFlagScan[b];
  pEt[n].x = a;
  pEt[n].y = b;
}

//#########################################################################
/*! \brief Kernel adjusting \a edgeTriangles for removed triangles

\param nEdge Total number of edges in Mesh
\param *et1 Pointer to array of first neighbouring triangle
\param *et2 Pointer to array of second neighbouring triangle
\param *pTriangleKeepFlagScan Scanned array of flags whether triangles are to be kept*/
//#########################################################################

__global__
void devAdjustEdge(int nEdge, int2 *pEt, int *pTriangleKeepFlagScan)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nEdge) {
    AdjustEdgeSingle(n, pEt, pTriangleKeepFlagScan);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! \brief Adjust \a triangleVertices and \a triangleEdges for removed vertices and edges

\param n Index of triangle to consider
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param nVertex Total number of vertices in Mesh
\param nvKeep Number of vertices to keep
\param *pVertexKeepFlagScan Scanned array of flags whether to keep vertices
\param *pEdgeKeepFlagScan Scanned array of flags whether to keep edges*/
//#########################################################################

__host__ __device__
void AdjustTriangleSingle(int n, int3 *pTv, int3 *pTe, int nVertex,
                          int nvKeep, int *pVertexKeepFlagScan,
                          int *pEdgeKeepFlagScan)
{
  int A = 0, B = 0, C = 0;
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) {
    a -= nVertex;
    A++;
  }
  while (b >= nVertex) {
    b -= nVertex;
    B++;
  }
  while (c >= nVertex) {
    c -= nVertex;
    C++;
  }
  while (a < 0) {
    a += nVertex;
    A--;
  }
  while (b < 0) {
    b += nVertex;
    B--;
  }
  while (c < 0) {
    c += nVertex;
    C--;
  }

  pTv[n].x = pVertexKeepFlagScan[a] + A*nvKeep;
  pTv[n].y = pVertexKeepFlagScan[b] + B*nvKeep;
  pTv[n].z = pVertexKeepFlagScan[c] + C*nvKeep;

  int d = pTe[n].x;
  int e = pTe[n].y;
  int f = pTe[n].z;
  pTe[n].x = pEdgeKeepFlagScan[d];
  pTe[n].y = pEdgeKeepFlagScan[e];
  pTe[n].z = pEdgeKeepFlagScan[f];
}

//#########################################################################
/*! \brief Adjust \a triangleVertices and \a triangleEdges for removed vertices and edges

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param nVertex Total number of vertices in Mesh
\param nvKeep Number of vertices to keep
\param *pVertexKeepFlagScan Scanned array of flags whether to keep vertices
\param *pEdgeKeepFlagScan Scanned array of flags whether to keep edges*/
//#########################################################################

__global__
void devAdjustTriangle(int nTriangle, int3 *pTv, int3 *pTe, int nVertex,
                       int nvKeep, int *pVertexKeepFlagScan,
                       int *pEdgeKeepFlagScan)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    AdjustTriangleSingle(n, pTv, pTe, nVertex, nvKeep,
                         pVertexKeepFlagScan, pEdgeKeepFlagScan);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! \brief Remove vertex \a vRemove from Mesh, adjusting surrounding connections

  Remove vertex \a vRemove from Mesh, adjusting surrounding connections

\param vRemove Vertex that will be removed
\param *vTri Pointer to triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *vKeep Pointer to array flagging whether to keep vertex (output)
\param *tKeep Pointer to array flagging whether to keep triangle (output)
\param *eKeep Pointer to array flagging whether to keep edge (output)
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param tTarget Target triangle*/
//#########################################################################

__host__ __device__
void RemoveVertex(int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
                  int *vKeep, int *tKeep, int *eKeep,
                  int nVertex, int eTarget,
                  int *pEdgeNeedsChecking, real2 *pEdgeCoordinates)
{

  int2 tCollapse;
  int3 V, E;
  int v1, v2;
  GetEdgeVertices(eTarget, pTv, pTe, pEt,
                  tCollapse, V, E, v1, v2);

  int t1 = tCollapse.x;
  int t2 = tCollapse.y;

  int vOrigin = v1;
  int vTarget = v2;
  int vRemove = v1;
  while (vRemove >= nVertex) vRemove -= nVertex;
  while (vRemove < 0) vRemove += nVertex;

#ifndef __CUDA_ARCH__
  int printFlag = 0;
  if (printFlag == 1) {
    std::cout << "Removing vertex " << vRemove << " along edge "
              << eTarget << std::endl;
  }
#endif

  E = pTe[t1];

  int a = pTv[t1].x;
  int b = pTv[t1].y;
  int c = pTv[t1].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = eTarget;
  int e2 = NextEdgeClockwise(e1, E);
  int e4 = NextEdgeCounterClockwise(e1, E);

  // pERIODIC
  int translateFlagOrigin = 0;
  while (vOrigin >= nVertex) {
    vOrigin -= nVertex;
    translateFlagOrigin++;
  }
  while (vOrigin < 0) {
    vOrigin += nVertex;
    translateFlagOrigin--;
  }
  int translateFlagTarget = 0;
  while (vTarget >= nVertex) {
    vTarget -= nVertex;
    translateFlagTarget++;
  }
  while (vTarget < 0) {
    vTarget += nVertex;
    translateFlagTarget--;
  }

  int translateFlag = (translateFlagTarget - translateFlagOrigin)*nVertex;

  // NEW
  pVc[vTarget] = pEdgeCoordinates[eTarget];

  // t1 and t2 are neighbours to e1

  // t3 is neighbour of t1 through edge e2
  int2 T = pEt[e2];
  int t3 = OtherNeighbouringTriangle(t1, T);

  int e3 = -1;
  int e5 = -1;
  int t4 = -1;

  if (t2 != -1) {
    int3 E = pTe[t2];
    e3 = NextEdgeCounterClockwise(e1, E);
    e5 = NextEdgeClockwise(e1, E);

    // t4 is neighbour of t2 through edge e3
    int2 T = pEt[e3];
    t4 = OtherNeighbouringTriangle(t2, T);
  }


  int t = t1;
  FindSegmentCounterClock(eTarget, t, E, pTe, pEt);

  int finished = 0;
  // If we try to cross this edge we are done
  int eFinal = -1;
  while (!finished) {
    int3 E = pTe[t];
    int3 V = pTv[t];

    int a = V.x;
    int b = V.y;
    int c = V.z;

    while (a >= nVertex) a -= nVertex;
    while (b >= nVertex) b -= nVertex;
    while (c >= nVertex) c -= nVertex;
    while (a < 0) a += nVertex;
    while (b < 0) b += nVertex;
    while (c < 0) c += nVertex;

    // Determine next triangle to visit
    int eCross = -1;
    if (a == vRemove) {
      eCross = E.x;
      if (eFinal == -1) eFinal = E.z;
    }
    if (b == vRemove) {
      eCross = E.y;
      if (eFinal == -1) eFinal = E.x;
    }
    if (c == vRemove) {
      eCross = E.z;
      if (eFinal == -1) eFinal = E.y;
    }
    int tNext = pEt[eCross].x;
    if (tNext == t) tNext = pEt[eCross].y;

    if (a == vRemove)
      V.x = vTarget + V.x - a + translateFlag;
    if (b == vRemove)
      V.y = vTarget + V.y - b + translateFlag;
    if (c == vRemove)
      V.z = vTarget + V.z - c + translateFlag;

    MakeValidIndices(V.x, V.y, V.z, nVertex);

    // Replace e2 with e4
    if (E.x == e2) E.x = e4;
    if (E.y == e2) E.y = e4;
    if (E.z == e2) E.z = e4;

    // Replace e3 with e5
    if (E.x == e3) E.x = e5;
    if (E.y == e3) E.y = e5;
    if (E.z == e3) E.z = e5;

    int2 T1 = pEt[E.x];
    int2 T2 = pEt[E.y];
    int2 T3 = pEt[E.z];

    // Replace t2 with t4
    if (T1.x == t2) T1.x = t4;
    if (T1.y == t2) T1.y = t4;
    if (T2.x == t2) T2.x = t4;
    if (T2.y == t2) T2.y = t4;
    if (T3.x == t2) T3.x = t4;
    if (T3.y == t2) T3.y = t4;

    // Replace t1 with t3
    if (T1.x == t1) T1.x = t3;
    if (T1.y == t1) T1.y = t3;
    if (T2.x == t1) T2.x = t3;
    if (T2.y == t1) T2.y = t3;
    if (T3.x == t1) T3.x = t3;
    if (T3.y == t1) T3.y = t3;

    pTv[t] = V;
    pTe[t] = E;
    pEt[E.x] = T1;
    pEt[E.y] = T2;
    pEt[E.z] = T3;

    // Move into next triangle
    t = tNext;
    if (eCross == eFinal || t == -1) finished = 1;
  }

  // Flag vertex, triangles and edges for removal
  vKeep[vRemove] = 0;
  tKeep[t1] = 0;
  if (t2 != -1) tKeep[t2] = 0;
  eKeep[e1] = 0;
  eKeep[e2] = 0;
  if (t2 != -1) eKeep[e3] = 0;
}

//#########################################################################
/*! \brief Kernel removing vertices from Mesh, adjusting surrounding connections

\param nRemove Number of vertices to be removed
\param *pVertexRemove Pointer to array of inices of vertices to be removed
\param pVertexTriangleList Pointer to list of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertexKeepFlag Pointer to array flagging whether to keep vertex (output)
\param *pTriangleKeepFlag Pointer to array flagging whether to keep triangle (output)
\param *pEdgeKeepFlag Pointer to array flagging whether to keep edge (output)
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param pTriangleTarget Pointer to array of target triangles*/
//#########################################################################

__global__
void devRemoveVertex(int nRemove,
                     int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
                     int *pVertexKeepFlag, int *pTriangleKeepFlag,
                     int *pEdgeKeepFlag, int nVertex,
                     int *pEdgeTarget,
                     int *pEdgeNeedsChecking,
                     real2 *pEdgeCoordinates)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    RemoveVertex(pTv, pTe, pEt, pVc,
                 pVertexKeepFlag, pTriangleKeepFlag, pEdgeKeepFlag,
                 nVertex, pEdgeTarget[n],
                 pEdgeNeedsChecking, pEdgeCoordinates);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Remove vertices from Mesh. The Array \a vertexRemove contains a list of vertices to be removed, and we have found target triangles in \a triangleTarget. First we remove the vertices and then we adjust all indices.

\param *vertexTriangleList Pointer to Array of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *triangleTarget Pointer to Array of target triangles
\param *vertexState Pointer to Array containing state vector */
//#########################################################################

template<class realNeq>
void Coarsen::Remove(Connectivity *connectivity,
                     Array<int> *triangleWantRefine,
                     Array<realNeq> *vertexState)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int *pEdgeNeedsChecking = edgeNeedsChecking->GetPointer();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();

  int *pEdgeCollapseList = edgeCollapseList->GetPointer();
  real2 *pEdgeCoordinates = edgeCoordinates->GetPointer();

  int nRemove = edgeCollapseList->GetSize();

  Array<int> *vertexKeepFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  Array<int> *triangleKeepFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  Array<int> *edgeKeepFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);
  vertexKeepFlag->SetToValue(1);
  triangleKeepFlag->SetToValue(1);
  edgeKeepFlag->SetToValue(1);
  int *pVertexKeepFlag = vertexKeepFlag->GetPointer();
  int *pTriangleKeepFlag = triangleKeepFlag->GetPointer();
  int *pEdgeKeepFlag = edgeKeepFlag->GetPointer();

  // Remove vertices from mesh
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devRemoveVertex,
                                       (size_t) 0, 0);

    devRemoveVertex<<<nBlocks, nThreads>>>
      (nRemove, pTv, pTe, pEt, pVc,
       pVertexKeepFlag, pTriangleKeepFlag, pEdgeKeepFlag,
       nVertex, pEdgeCollapseList,
       pEdgeNeedsChecking, pEdgeCoordinates);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      RemoveVertex(pTv, pTe, pEt, pVc,
                   pVertexKeepFlag, pTriangleKeepFlag, pEdgeKeepFlag,
                   nVertex, pEdgeCollapseList[n],
                   pEdgeNeedsChecking, pEdgeCoordinates);
  }

  Array<int> *vertexKeepFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  Array<int> *triangleKeepFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  Array<int> *edgeKeepFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);

  int nvKeep = vertexKeepFlag->ExclusiveScan(vertexKeepFlagScan, nVertex);
  int ntKeep = triangleKeepFlag->ExclusiveScan(triangleKeepFlagScan,
                                               nTriangle);
  int neKeep = edgeKeepFlag->ExclusiveScan(edgeKeepFlagScan, nEdge);

  int *pVertexKeepFlagScan = vertexKeepFlagScan->GetPointer();
  int *pTriangleKeepFlagScan = triangleKeepFlagScan->GetPointer();
  int *pEdgeKeepFlagScan = edgeKeepFlagScan->GetPointer();

  // Adjust tv and te for removed vertices and triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustTriangle,
                                       (size_t) 0, 0);

    devAdjustTriangle<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pTe, nVertex, nvKeep,
       pVertexKeepFlagScan, pEdgeKeepFlagScan);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      AdjustTriangleSingle(n, pTv, pTe, nVertex, nvKeep,
                           pVertexKeepFlagScan, pEdgeKeepFlagScan);
  }

  // Adjust et for removed triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustEdge,
                                       (size_t) 0, 0);

    devAdjustEdge<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleKeepFlagScan);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nEdge; n++)
      AdjustEdgeSingle(n, pEt, pTriangleKeepFlagScan);
  }

  connectivity->vertexCoordinates->Compact(nvKeep, vertexKeepFlag,
                                           vertexKeepFlagScan);
  connectivity->vertexArea->Compact(nvKeep, vertexKeepFlag,
                                    vertexKeepFlagScan);
  vertexState->Compact(nvKeep, vertexKeepFlag, vertexKeepFlagScan);
  triangleWantRefine->Compact(ntKeep, triangleKeepFlag, triangleKeepFlagScan);

  connectivity->triangleVertices->Compact(ntKeep, triangleKeepFlag,
                                          triangleKeepFlagScan);
  connectivity->triangleEdges->Compact(ntKeep, triangleKeepFlag,
                                       triangleKeepFlagScan);
  connectivity->edgeTriangles->Compact(neKeep, edgeKeepFlag, edgeKeepFlagScan);

  edgeNeedsChecking->Compact(neKeep, edgeKeepFlag, edgeKeepFlagScan);
  pEdgeNeedsChecking = edgeNeedsChecking->GetPointer();

  // Adjust edgeNeedsChecking
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustEdgeNeedsChecking,
                                       (size_t) 0, 0);

    devAdjustEdgeNeedsChecking<<<nBlocks, nThreads>>>
      (neKeep, pEdgeNeedsChecking);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < neKeep; n++)
      AdjustEdgeNeedsCheckingSingle(n, pEdgeNeedsChecking);
  }

  delete vertexKeepFlag;
  delete triangleKeepFlag;
  delete edgeKeepFlag;

  delete vertexKeepFlagScan;
  delete triangleKeepFlagScan;
  delete edgeKeepFlagScan;
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Coarsen::Remove<real>(Connectivity *connectivity,
                      Array<int> *triangleWantRefine,
                      Array<real> *vertexState);
template void
Coarsen::Remove<real3>(Connectivity *connectivity,
                       Array<int> *triangleWantRefine,
                       Array<real3> *vertexState);
template void
Coarsen::Remove<real4>(Connectivity *connectivity,
                       Array<int> *triangleWantRefine,
                       Array<real4> *vertexState);

}
