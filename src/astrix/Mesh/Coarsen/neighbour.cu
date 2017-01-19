// -*-c++-*-
/*! \file neighbour.cu
\brief Functions for finding vertex neighbours.
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
/*! \brief Find all neighbouring vertices for \a vRemove

Find all vertices sharing an edge with \a vRemove and put the result in \a *vNeighbour.

\param vRemove Vertex under consideration
\param *vTri Pointer to list of triangles sharing vertex \a vRemove
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param tTarget Target triangle containing vertex to move vRemove onto
\param *vNeighbour Pointer to array of neighbouring vertices (output)*/
//#########################################################################

__host__ __device__
void FindVertexNeighbourSingle(int vRemove, int *vTri, const int maxTriPerVert,
                               int3 *pTv, int3 *pTe, int2 *pEt,
                               int nVertex, int tTarget, int *vNeighbour)
{
  // Return if vertex cannot be removed
  if (tTarget == -1) return;

  // Find target vertex (vertex to move vRemove onto)
  int a = pTv[tTarget].x;
  int b = pTv[tTarget].y;
  int c = pTv[tTarget].z;

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = pTe[tTarget].x;
  int e2 = pTe[tTarget].y;
  int e3 = pTe[tTarget].z;

  int t11 = pEt[e1].x;
  int t21 = pEt[e1].y;
  int t12 = pEt[e2].x;
  int t22 = pEt[e2].y;
  int t13 = pEt[e3].x;
  int t23 = pEt[e3].y;

  // Check if vertex is part of segment
  int onSegmentFlag = 0;
  if ((a == vRemove || b == vRemove) && (t11 == -1 || t21 == -1))
    onSegmentFlag = 1;
  if ((b == vRemove || c == vRemove) && (t12 == -1 || t22 == -1))
    onSegmentFlag = 1;
  if ((c == vRemove || a == vRemove) && (t13 == -1 || t23 == -1))
    onSegmentFlag = 1;

  // Check all triangles sharing vRemove
  for (int i = 0; i < maxTriPerVert; i++) {
    int ret = -1;

    int t = vTri[i];

    if (t != -1) {
      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;

      while (a >= nVertex) a -= nVertex;
      while (b >= nVertex) b -= nVertex;
      while (c >= nVertex) c -= nVertex;
      while (a < 0) a += nVertex;
      while (b < 0) b += nVertex;
      while (c < 0) c += nVertex;

      // Select 'next' vertex
      if (a == vRemove) ret = b;
      if (b == vRemove) ret = c;
      if (c == vRemove) ret = a;
    }

    vNeighbour[i] = ret;
  }

  // Must add extra vertex if on segment
  if (onSegmentFlag == 1) {
    for (int i = 0; i < maxTriPerVert; i++) {
      int ret = -1;

      int t = vTri[i];

      if (t != -1) {
        int a = pTv[t].x;
        int b = pTv[t].y;
        int c = pTv[t].z;

        while (a >= nVertex) a -= nVertex;
        while (b >= nVertex) b -= nVertex;
        while (c >= nVertex) c -= nVertex;
        while (a < 0) a += nVertex;
        while (b < 0) b += nVertex;
        while (c < 0) c += nVertex;

        if (a == vRemove) ret = c;
        if (b == vRemove) ret = a;
        if (c == vRemove) ret = b;

        int uniqueFlag = 1;
        for (int j = 0; j < maxTriPerVert; j++)
          uniqueFlag *= (ret != vNeighbour[j]);

        if (uniqueFlag == 1) vNeighbour[maxTriPerVert - 1] = ret;
      }
    }
  }
}

//#########################################################################
/*! \brief Find all neighbouring vertices for all vertices in \a *pVertexRemove

Find all vertices sharing an edge with vertices in  \a *pVertexRemove and put the result in \a *vNeighbour.

\param *pVertexRemove Pointer to list of vertices to consider
\param nRemove Number of vertices in \a *pVertexRemove
\param *pVertexTriangleList Pointer to list of triangles sharing vertices
\param maxTriPerVert Maximum number of triangles sharing any vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param pTriangleTarget Target triangle containing vertex to move vertices onto
\param *pVertexNeighbour Pointer to array of neighbouring vertices (output)*/
//#########################################################################

__global__
void devFindVertexNeighbour(int *pVertexRemove, int nRemove,
                            int *pVertexTriangleList,
                            const int maxTriPerVert,
                            int3 *pTv, int3 *pTe, int2 *pEt,
                            int nVertex, int *pTriangleTarget,
                            int *pVertexNeighbour)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FindVertexNeighbourSingle(pVertexRemove[n],
                              &(pVertexTriangleList[n*maxTriPerVert]),
                              maxTriPerVert, pTv, pTe, pEt,
                              nVertex, pTriangleTarget[n],
                              &(pVertexNeighbour[n*maxTriPerVert]));

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! For every vertex, find its neighbours (vertices sharing an edge).

\param *vertexNeighbour Pointer to Array of neighbouring vertices (output)
\param *triangleTarget Pointer to Array containing target triangles
\param *vertexTriangleList Pointer to Array containing list of triangles sharing a vertex
\param maxTriPerVert Maximum number of triangles sharing any vertex*/
//#########################################################################

void Coarsen::FindVertexNeighbours(Connectivity *connectivity,
                                   Array<int> *vertexNeighbour,
                                   Array<int> *triangleTarget,
                                   Array<int> *vertexTriangleList,
                                   int maxTriPerVert)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexNeighbour->TransformToHost();
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexNeighbour->TransformToDevice();
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();

  int *pVertexNeighbour = vertexNeighbour->GetPointer();

  int *pTriangleTarget = triangleTarget->GetPointer();
  int *pVertexTriangleList = vertexTriangleList->GetPointer();
  int *pVertexRemove = vertexRemove->GetPointer();
  int nRemove = vertexRemove->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  // Find neighbouring vertices
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFindVertexNeighbour,
                                       (size_t) 0, 0);

    devFindVertexNeighbour<<<nBlocks, nThreads>>>
      (pVertexRemove, nRemove, pVertexTriangleList,
       maxTriPerVert, pTv, pTe, pEt,
       nVertex, pTriangleTarget, pVertexNeighbour);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FindVertexNeighbourSingle(pVertexRemove[n],
                                &(pVertexTriangleList[n*maxTriPerVert]),
                                maxTriPerVert, pTv, pTe, pEt,
                                nVertex, pTriangleTarget[n],
                                &(pVertexNeighbour[n*maxTriPerVert]));
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexNeighbour->TransformToHost();
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexNeighbour->TransformToDevice();
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();

      cudaFlag = 1;
    }
  }

}

}
