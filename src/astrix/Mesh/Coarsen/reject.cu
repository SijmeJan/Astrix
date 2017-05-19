// -*-c++-*-
/*! \file coarsen_reject.cu
\brief Functions for rejecting large triangles for coarsening.

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
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "./coarsen.h"

namespace astrix {

//#########################################################################
/*! \brief Reject triangles that are too large for coarsening

We do not want to coarsen too much: reject any triangle for coarsening that has an edge that is larger than half the domain size

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param Px Domain size x
\param Py Domain size y
\param *pTriangleWantRefine Pointer to array indicating whether triangle wants refining (=1) or coarsening (=-1). If equal to -1, set to 0 if triangle too large*/
//#########################################################################

__host__ __device__
void RejectSingle(int n, int3 *pTv, int nVertex,
                  real2 *pVc, real Px, real Py,
                  int *pTriangleWantRefine)
{
  real half = (real) 0.5;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  real l1 = (ax - bx)*(ax - bx) + (ay - by)*(ay - by);
  real l2 = (ax - cx)*(ax - cx) + (ay - cy)*(ay - cy);
  real l3 = (cx - bx)*(cx - bx) + (cy - by)*(cy - by);

  real maxEdgeLength = sqrt(max(l1, max(l2, l3)));
  real minDomainSize = min(Px, Py);

  if (maxEdgeLength > half*minDomainSize) {
    pTriangleWantRefine[n] = 0;
  }
}

//#########################################################################
/*! \brief Kernel rejecting triangles that are too large for coarsening

We do not want to coarsen too much: reject any triangle for coarsening that has an edge that is larger than half the domain size

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param Px Domain size x
\param Py Domain size y
\param *pTriangleWantRefine Pointer to array indicating whether triangle wants refining (=1) or coarsening (=-1). If equal to -1, set to 0 if triangle too large*/
//#########################################################################

__global__
void devReject(int nTriangle, int3 *pTv, int nVertex,
               real2 *pVc, real Px, real Py,
               int *pTriangleWantRefine)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    RejectSingle(n, pTv, nVertex, pVc, Px, Py, pTriangleWantRefine);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! We do not want to coarsen too much: reject any triangle for coarsening that has an edge that is larger than half the domain size. The corresponding entry in \a triangleWantRefine is set to 0.

\param *connectivity Pointer to Mesh connectivity data
\param *mp Pointer to Mesh parameter object
\param *triangleWantRefine Pointer to flags whether triangles can be coarsened. Will be modified. */
//#########################################################################

void Coarsen::RejectLargeTriangles(Connectivity *connectivity,
                                   const MeshParameter *mp,
                                   Array<int> *triangleWantRefine)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();

  int *pTriangleWantRefine = triangleWantRefine->GetPointer();

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  // Flag vertices for removal
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devReject,
                                       (size_t) 0, 0);

    devReject<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex, pVc, Px, Py, pTriangleWantRefine);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      RejectSingle(n, pTv, nVertex, pVc, Px, Py, pTriangleWantRefine);
  }
}

}
