// -*-c++-*-
/*! \file checksegment.cu
\brief Check if edges can be flipped to recover segments.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./delaunay.h"
#include "./../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"
#include "../../Common/inlineMath.h"

namespace astrix {

//#########################################################################
/*! \brief Check edge \a i for Delaunay-hood

Check edge \a i and write result in \a eNonDel (i if flippable, -1 otherwise)

\param i Index of edge to check
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pVo Pointer to vertex insertion order*/
//#########################################################################

__host__ __device__
int CheckSegment(int i,
                 real2 *pVc,
                 const int3* __restrict__ pTv,
                 const int3* __restrict__ pTe,
                 int2 *pEt,
                 const Predicates *pred, real *pParam,
                 int nVertex, real Px, real Py,
                 const int *pVo)
{
  // Assume edge is OK
  int ret = -1;

  int t1 = pEt[i].x;
  int t2 = pEt[i].y;

  // If two neighbouring triangles...
  if (t1 != -1 && t2 != -1) {
    int a = pTv[t1].x;
    int b = pTv[t1].y;
    int c = pTv[t1].z;

    int e1 = pTe[t1].x;
    int e2 = pTe[t1].y;
    int e3 = pTe[t1].z;

    // t1 vertex coordinates
    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    // Edge i is between v1 and v2
    int v1 =   (i == e1)*a +  (i == e2)*b +  (i == e3)*c;
    int v2 =   (i == e1)*b +  (i == e2)*c +  (i == e3)*a;
    v1 = NormalizeVertex(v1, nVertex);
    v2 = NormalizeVertex(v2, nVertex);

    int dv =
      min(abs(pVo[v1] - pVo[v2]),
          min(abs(pVo[v1] - pVo[v2] + nVertex - 4),
              abs(pVo[v1] - pVo[v2] - nVertex + 4)));

    // Vertex of t1 not part of edge i
    int d = (i == e1)*c + (i == e2)*a + (i == e3)*b;

    a = pTv[t2].x;
    b = pTv[t2].y;
    c = pTv[t2].z;

    e1 = pTe[t2].x;
    e2 = pTe[t2].y;
    e3 = pTe[t2].z;

    // Vertex of t2 not part of edge i
    int f = (i == e1)*c + (i == e2)*a + (i == e3)*b;
    real fx, fy;
    GetTriangleCoordinatesSingle(pVc, f, nVertex, Px, Py, fx, fy);

    d = NormalizeVertex(d, nVertex);
    f = NormalizeVertex(f, nVertex);

    int df =
      min(abs(pVo[d] - pVo[f]),
          min(abs(pVo[d] - pVo[f] + nVertex - 4),
              abs(pVo[d] - pVo[f] - nVertex + 4)));

    // Edge flippable if nBad == 1
    int nBad =
      (pred->orient2d(ax, ay, bx, by, fx, fy, pParam) < 0) +
      (pred->orient2d(bx, by, cx, cy, fx, fy, pParam) < 0) +
      (pred->orient2d(cx, cy, ax, ay, fx, fy, pParam) < 0);

    // If vertices d and f were inserted in order, they form a segment
    // that needs to be recovered: flip needed
    if (df <  dv && nBad == 1) ret = i;
  }

  return ret;
}

//######################################################################
/*! \brief Kernel checking if edges acn be flipped to recover segments

Check edges and write result in \a pEnd (i if flippable, -1 otherwise)

\param nEdge Total number of edges in Mesh
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pEnd Pointer to list of edges should be flipped (output)
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pVo Pointer to vertex insertion order*/
//######################################################################

__global__ void
devCheckSegment(int nEdge,
                real2 *pVc,
                const int3* __restrict__ pTv,
                const int3* __restrict__ pTe,
                int2 *pEt,
                int *pEnd,
                const Predicates *pred, real *pParam,
                int nVertex,
                real Px, real Py,
                const int *pVo)
{
  // i = edge number
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    pEnd[i] =
      CheckSegment(i, pVc, pTv, pTe, pEt, pred, pParam,
                   nVertex, Px, Py, pVo);

    // Next edge
    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Check which edges need to be flipped to recover segments. Result is written in \a edgeNonDelaunay (-1 if no need to flip)

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to exact geometric predicates
\param *meshParameter Pointer to Mesh parameters
\param *vertexOrder Pointer to Array containing vertex insertion order.
*/
//#########################################################################

void Delaunay::CheckSegments(Connectivity * const connectivity,
                             const Predicates *predicates,
                             const MeshParameter *meshParameter,
                             Array<int> * const vertexOrder)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int *pEnd = edgeNonDelaunay->GetPointer();

  int *pVo = vertexOrder->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCheckSegment,
                                       (size_t) 0, 0);

    devCheckSegment<<<nBlocks, nThreads>>>
      (nEdge, pVc, pTv, pTe, pEt, pEnd, predicates, pParam,
       nVertex, Px, Py, pVo);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++)
      pEnd[i] =
        CheckSegment(i, pVc, pTv, pTe, pEt, predicates, pParam,
                     nVertex, Px, Py, pVo);
  }
}

}  // namespace astrix
