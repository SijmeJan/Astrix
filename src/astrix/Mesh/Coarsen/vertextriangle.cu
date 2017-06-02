// -*-c++-*-
/*! \file vertextriangle.cu
\brief Functions for finding triangles containing certain vertices.

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
#include "../../Common/atomic.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! \brief Set triangle \a n as the triangle for its vertices

  For every vertex we want to know one triangle sharing it and put the result in \a *pVertexTriangle. Here, we consider triangle \a n and set it as the \a vertexTriangle for its vertices using atomic operations. If the vertex is part of a segment, we make sure we can walk around the vertex in clockwise direction.

\param n Index of triangle to consider
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param *pVertexTriangle Pointer to output array*/
//#########################################################################

__host__ __device__
void FillVertexTriangleSingle(int n, int3 *pTv, int3 *pTe, int2 *pEt,
                              int nVertex, int *pVertexTriangle)
{
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = pTe[n].x;
  int e2 = pTe[n].y;
  int e3 = pTe[n].z;

  if (pEt[e3].x == -1 || pEt[e3].y == -1)
    AtomicExch(&(pVertexTriangle[a]), n);
  if (pEt[e1].x == -1 || pEt[e1].y == -1)
    AtomicExch(&(pVertexTriangle[b]), n);
  if (pEt[e2].x == -1 || pEt[e2].y == -1)
    AtomicExch(&(pVertexTriangle[c]), n);

  AtomicCAS(&(pVertexTriangle[a]), -1, n);
  AtomicCAS(&(pVertexTriangle[b]), -1, n);
  AtomicCAS(&(pVertexTriangle[c]), -1, n);

  /*
  AtomicExch(&(pVertexTriangle[a]), n);
  AtomicExch(&(pVertexTriangle[b]), n);
  AtomicExch(&(pVertexTriangle[c]), n);
  */
}

//#########################################################################
/*! \brief Kernel setting \a vertexTriangle for all vertices

  For every vertex we want to know one triangle sharing it and put the result in \a *pVertexTriangle. Here, we loop through all triangles and set it as the \a vertexTriangle for its vertices using atomic operations. Then \a pVertexTriangle will contain the triangle that has last written to it. If the vertex is part of a segment, we make sure we can walk around the vertex in clockwise direction.

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param *pVertexTriangle Pointer to output array*/
//#########################################################################

__global__
void devFillVertexTriangle(int nTriangle, int3 *pTv, int3 *pTe, int2 *pEt,
                           int nVertex, int *pVertexTriangle)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    FillVertexTriangleSingle(n, pTv, pTe, pEt, nVertex, pVertexTriangle);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! For every vertex we want to know one triangle sharing it, and put the result in \a vertexTriangle. Here, we loop through all triangles and set it as the \a vertexTriangle for its vertices using atomic operations. Then \a vertexTriangle will contain the triangle that has last written to it. If the vertex is part of a segment, we make sure we can walk around the vertex in clockwise direction.

\param *connectivity Pointer to Mesh connectivity data*/
//#########################################################################

void Coarsen::FillVertexTriangle(Connectivity *connectivity)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  vertexTriangle->SetSize(nVertex);
  vertexTriangle->SetToValue(-1);
  int *pVertexTriangle = vertexTriangle->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillVertexTriangle,
                                       (size_t) 0, 0);

    devFillVertexTriangle<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pTe, pEt, nVertex, pVertexTriangle);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      FillVertexTriangleSingle(n, pTv, pTe, pEt, nVertex, pVertexTriangle);
  }


}

}
