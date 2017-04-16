// -*-c++-*-
/*! \file redundant.cu
\brief Function for removing redundant vertices and triangles from initial mesh

\section LICENSE
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
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//######################################################################
/*! \brief Flag vertices for removal

Flag whether vertex \a i needs to be removed. Only the first four inserted vertices are flagged for removal

\param i Index in vertex array
\param pVertexOrder Pointer to vertex insertion order array
\param pVertexRemoveFlag Pointer to output array*/
//######################################################################

__host__ __device__
void FlagVertexRemoveSingle(int i, int *pVertexOrder, int *pVertexRemoveFlag)
{
  int ret = 0;
  // Only flag first four inserted vertices
  if (pVertexOrder[i] < 4) ret = 1;
  pVertexRemoveFlag[i] = ret;
}

//######################################################################
/*! \brief Kernel for flagging vertices for removal.

Only the first four inserted vertices are flagged for removal

\param nVertex Total number of vertices in Mesh
\param pVertexOrder Pointer to vertex insertion order array
\param pVertexRemoveFlag Pointer to output array*/
//######################################################################

__global__ void
devFlagVertexRemove(int nVertex, int *pVertexOrder, int *pVertexRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    FlagVertexRemoveSingle(i, pVertexOrder, pVertexRemoveFlag);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Flag triangles for removal

Flag any triangle located outside the outer boundary or inside the inner boundary for removal.

\param i Triangle to be checked
\param *pTv Pointer to triangle vertices
\param *pVertexRemoveFlag Pointer to flags whether vertices will be removed
\param *pVertexOrder Pointer to vertex insertion order
\param nVertexOuterBoundary Number of vertices making up outer boundary
\param *pTriangleRemoveFlag Pointer to output array
\param nVertex Total number of vertices in current mesh*/
//######################################################################

__host__ __device__
void FlagTriangleRemoveSingle(int i, int3 *pTv, int *pVertexRemoveFlag,
                              int *pVertexOrder, int nVertexOuterBoundary,
                              int *pTriangleRemoveFlag, int nVertex)
{
  int ret = 0;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;

  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;

  // Remove any triangle for which at least one vertex will be removed
  if (pVertexRemoveFlag[a] == 1 ||
      pVertexRemoveFlag[b] == 1 ||
      pVertexRemoveFlag[c] == 1) ret = 1;

  int nBad = 0;

  // Remove any triangle for which all vertices are part of the outer boundary
  // and for which vertices occur in wrong order
  if (pVertexOrder[a] < nVertexOuterBoundary + 4 &&
      pVertexOrder[b] < nVertexOuterBoundary + 4 &&
      pVertexOrder[c] < nVertexOuterBoundary + 4)
    nBad = (pVertexOrder[a] > pVertexOrder[b]) +
                        (pVertexOrder[b] > pVertexOrder[c]) +
      (pVertexOrder[c] > pVertexOrder[a]);

  if (pVertexOrder[a] >= nVertexOuterBoundary + 4 &&
      pVertexOrder[b] >= nVertexOuterBoundary + 4 &&
      pVertexOrder[c] >= nVertexOuterBoundary + 4)
    nBad = (pVertexOrder[a] > pVertexOrder[b]) +
      (pVertexOrder[b] > pVertexOrder[c]) +
      (pVertexOrder[c] > pVertexOrder[a]);

  if (nBad > 1) ret = 1;

  pTriangleRemoveFlag[i] = ret;
}

//######################################################################
/*! \brief Kernel flagging triangles for removal

Flag any triangle for which one of its vertices will be removed for removal.

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVertexRemoveFlag Pointer to flags whether vertices will be removed
\param *pVertexOrder Pointer to vertex insertion order
\param nVertexOuterBoundary Number of vertices making up outer boundary
\param *pTriangleRemoveFlag Pointer to output array
\param nVertex Total number of vertices in current mesh*/
//######################################################################

__global__ void
devFlagTriangleRemove(int nTriangle, int3 *pTv, int *pVertexRemoveFlag,
                      int *pVertexOrder, int nVertexOuterBoundary,
                      int *pTriangleRemoveFlag, int nVertex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FlagTriangleRemoveSingle(i, pTv, pVertexRemoveFlag,
                             pVertexOrder, nVertexOuterBoundary,
                             pTriangleRemoveFlag, nVertex);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Flag unnecessary edges for removal

Flag any edge for which both neighbouring triangles will be removed

\param i Index in edge array
\param *pEt Pointer to array containing edge triangles
\param *pTriangleRemoveFlag Pointer to flags whether triangles will be removed
\param *pEdgeRemoveFlag Pointer to output array*/
//######################################################################

__host__ __device__
void FlagEdgeRemoveSingle(int i, int2 *pEt,
                          int *pTriangleRemoveFlag, int *pEdgeRemoveFlag)
{
  int t1 = pEt[i].x;
  int t2 = pEt[i].y;

  if (t1 != -1)
    if (pTriangleRemoveFlag[t1] == 1) t1 = -1;
  if (t2 != -1)
    if (pTriangleRemoveFlag[t2] == 1) t2 = -1;
  int ret = 0;
  if (t1 == -1 && t2 == -1) ret = 1;
  pEdgeRemoveFlag[i] = ret;

  pEt[i].x = t1;
  pEt[i].y = t2;
}

//######################################################################
/*! \brief Kernel to flag unnecessary edges for removal

Flag any edge for which both neighbouring triangles will be removed

\param nEdge Total number of edges in Mesh
\param *pEt Pointer to array containing edge triangles
\param *pTriangleRemoveFlag Pointer to flags whether triangles will be removed
\param *pEdgeRemoveFlag Pointer to output array*/
//######################################################################

__global__ void
devFlagEdgeRemove(int nEdge, int2 *pEt,
                  int *pTriangleRemoveFlag, int *pEdgeRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    FlagEdgeRemoveSingle(i, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Adjust triangle vertices for vertex removal

\param i Index of triangle
\param *pTv Pointer to triangle vertices
\param *pVertexFlagScan Pointer to scanned array of flags whether vertices will be removed
\param nVertex Total number of vertices in current mesh*/
//######################################################################

__host__ __device__
void AdjustTriangleVerticesSingle(int i, int3 *pTv, int *pVertexFlagScan,
                                  int nVertex)
{
  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;

  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;

  pTv[i].x -= pVertexFlagScan[a];
  pTv[i].y -= pVertexFlagScan[b];
  pTv[i].z -= pVertexFlagScan[c];
}

//######################################################################
/*! \brief Kernel adjusting triangle vertices for vertex removal

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVertexFlagScan Pointer to scanned array of flags whether vertices will be removed
\param nVertex Total number of vertices in current mesh*/
//######################################################################

__global__ void
devAdjustTriangleVertices(int nTriangle, int3 *pTv, int *pVertexFlagScan,
                          int nVertex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    AdjustTriangleVerticesSingle(i, pTv, pVertexFlagScan, nVertex);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Adjust edge triangles for triangle removal

\param i Index of edge
\param *pEt Pointer to edge triangles
\param *pTriangleFlagScan Pointer to scanned array of flags whether triangles will be removed*/
//######################################################################

__host__ __device__
void AdjustEdgeTrianglesSingle(int i, int2 *pEt, int *pTriangleFlagScan)
{
  int t1 = pEt[i].x;
  int t2 = pEt[i].y;

  if (t1 != -1) t1 -= pTriangleFlagScan[t1];
  if (t2 != -1) t2 -= pTriangleFlagScan[t2];

  pEt[i].x = t1;
  pEt[i].y = t2;
}

//######################################################################
/*! \brief Kernel adjusting edge triangles for triangle removal

\param nEdge Total number of edges in Mesh
\param *pEt Pointer to edge triangles
\param *pTriangleFlagScan Pointer to scanned array of flags whether triangles will be removed*/
//######################################################################

__global__ void
devAdjustEdgeTriangles(int nEdge, int2 *pEt, int *pTriangleFlagScan)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    AdjustEdgeTrianglesSingle(i, pEt, pTriangleFlagScan);

    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Adjust triangle edges for edge removal

\param i Index of triangle
\param *pTe Pointer to triangle edges
\param *pEdgeFlagScan Pointer to scanned array of flags whether edges will be removed*/
//######################################################################

__host__ __device__
void AdjustTriangleEdgesSingle(int i, int3 *pTe, int *pEdgeFlagScan)
{
  pTe[i].x -= pEdgeFlagScan[pTe[i].x];
  pTe[i].y -= pEdgeFlagScan[pTe[i].y];
  pTe[i].z -= pEdgeFlagScan[pTe[i].z];
}

//######################################################################
/*! \brief Kernel adjusting triangle edges for edge removal

\param nTriangle Total number of triangles in Mesh
\param *pTe Pointer to triangle edges
\param *pEdgeFlagScan Pointer to scanned array of flags whether edges will be removed*/
//######################################################################

__global__ void
devAdjustTriangleEdges(int nTriangle, int3 *pTe, int *pEdgeFlagScan)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    AdjustTriangleEdgesSingle(i, pTe, pEdgeFlagScan);

    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! The initial mesh was generated within a bounding box large enough to contain all vertices. This function removes the vertices making up this bounding box and all associated triangles. Moreover, all triangles inside the inner boundary are removed as well.

\param *vertexOrder The order in which the boundary vertices were inserted. Entries with \a vertexOrder[i] < 4 belong to the bounding box, entries with 4 <= \a vertexOrder[i] < nVertexOuterBoundary + 4 belong to the outer boundary, and the rest to the inner boundary
\param nVertexOuterBoundary Number of vertices making up the outer boundary
*/
//######################################################################

void Mesh::RemoveRedundant(Array<int> *vertexOrder,
                           int nVertexOuterBoundary)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  Array<int> *vertexRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  int *pVertexRemoveFlag = vertexRemoveFlag->GetPointer();
  Array<int> *vertexFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  int *pVertexFlagScan = vertexFlagScan->GetPointer();
  Array<int> *triangleRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  int *pTriangleRemoveFlag = triangleRemoveFlag->GetPointer();
  Array<int> *triangleFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  int *pTriangleFlagScan = triangleFlagScan->GetPointer();
  Array<int> *edgeRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);
  int *pEdgeRemoveFlag = edgeRemoveFlag->GetPointer();
  Array<int> *edgeFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);
  int *pEdgeFlagScan = edgeFlagScan->GetPointer();

  int *pVertexOrder = vertexOrder->GetPointer();

  // Flag first four vertices to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFlagVertexRemove,
                                       (size_t) 0, 0);

    devFlagVertexRemove<<<nBlocks, nThreads>>>
      (nVertex, pVertexOrder, pVertexRemoveFlag);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      FlagVertexRemoveSingle(i, pVertexOrder, pVertexRemoveFlag);
  }

  // Flag triangles to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFlagTriangleRemove,
                                       (size_t) 0, 0);

    devFlagTriangleRemove<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVertexRemoveFlag,
       pVertexOrder, nVertexOuterBoundary, pTriangleRemoveFlag, nVertex);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      FlagTriangleRemoveSingle(i, pTv, pVertexRemoveFlag,
                               pVertexOrder, nVertexOuterBoundary,
                               pTriangleRemoveFlag, nVertex);
  }

  // Flag edges to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFlagEdgeRemove,
                                       (size_t) 0, 0);

    devFlagEdgeRemove<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++)
      FlagEdgeRemoveSingle(i, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);
  }

  vertexRemoveFlag->ExclusiveScan(vertexFlagScan, nVertex);

  // Adjust tv's for removed vertices
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devAdjustTriangleVertices,
       (size_t) 0, 0);

    devAdjustTriangleVertices<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVertexFlagScan, nVertex);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      AdjustTriangleVerticesSingle(i, pTv, pVertexFlagScan, nVertex);
  }

  triangleRemoveFlag->ExclusiveScan(triangleFlagScan, nTriangle);

  // Adjust et's for removed triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustEdgeTriangles,
                                       (size_t) 0, 0);

    devAdjustEdgeTriangles<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleFlagScan);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++)
      AdjustEdgeTrianglesSingle(i, pEt, pTriangleFlagScan);
  }

  edgeRemoveFlag->ExclusiveScan(edgeFlagScan, nEdge);

  // Adjust te's for removed edges
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustTriangleEdges,
                                       (size_t) 0, 0);

    devAdjustTriangleEdges<<<nBlocks, nThreads>>>
      (nTriangle, pTe, pEdgeFlagScan);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      AdjustTriangleEdgesSingle(i, pTe, pEdgeFlagScan);
  }

  vertexRemoveFlag->Invert();
  int nvKeep = vertexRemoveFlag->ExclusiveScan(vertexFlagScan, nVertex);
  connectivity->vertexCoordinates->Compact(nvKeep, vertexRemoveFlag,
                                           vertexFlagScan);

  triangleRemoveFlag->Invert();
  int ntKeep = triangleRemoveFlag->ExclusiveScan(triangleFlagScan, nTriangle);
  connectivity->triangleVertices->Compact(ntKeep, triangleRemoveFlag,
                                          triangleFlagScan);
  connectivity->triangleEdges->Compact(ntKeep, triangleRemoveFlag,
                                       triangleFlagScan);

  edgeRemoveFlag->Invert();
  int neKeep = edgeRemoveFlag->ExclusiveScan(edgeFlagScan, nEdge);
  connectivity->edgeTriangles->Compact(neKeep, edgeRemoveFlag, edgeFlagScan);

  delete vertexRemoveFlag;
  delete vertexFlagScan;
  delete triangleRemoveFlag;
  delete triangleFlagScan;
  delete edgeRemoveFlag;
  delete edgeFlagScan;
}

}
