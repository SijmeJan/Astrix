// -*-c++-*-
/*! \file edgenormal.cu
\brief Functions for ccalculating triangle normals and triangle and vertex areas

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "./triangleLow.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//######################################################################
/*! \brief Calculate normals and edge lengths for triangle \a n

\param n Index of triangle to consider
\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVc Pointer to vertex coordinates
\param *pTn1 Pointer to triangle normals first edge (output)
\param *pTn2 Pointer to triangle normals second edge (output)
\param *pTn3 Pointer to triangle normals third edge (output)
\param *triL Pointer to array of triangle edge lengths (output)
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pTrad Pointer to average x coordinate (output)*/
//######################################################################

__host__ __device__
void CalcNormalEdgeSingle(int n, int nTriangle, int3 *pTv, real2 *pVc,
                          real2 *pTn1, real2 *pTn2, real2 *pTn3,
                          real3 *triL, int nVertex, real Px, real Py,
                          real *pTrad)
{
  const real zero  = (real) 0.0;
  const real half  = (real) 0.5;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  // Vector along face
  real facedx = bx - cx;
  real facedy = by - cy;

  // Vector orthogonal to face
  real nx = facedy;
  real ny = -facedx;

  // Check if pointing inward
  real innerprod = nx*ax + ny*ay - half*(nx*bx + ny*by + nx*cx + ny*cy);

  // If pointing outward, reverse
  if (innerprod < zero) {
    nx = -nx;
    ny = -ny;
  }

  // Scale to length unity
  real inverselength = pow(nx*nx + ny*ny, -half);

  pTn1[n].x = nx*inverselength;
  pTn1[n].y = ny*inverselength;

  // Vector along face
  facedx = cx - ax;
  facedy = cy - ay;

  // Vector orthogonal to face
  nx = facedy;
  ny = -facedx;

  // Check if pointing inward
  innerprod = nx*bx + ny*by - half*(nx*cx + ny*cy + nx*ax + ny*ay);

  // If pointing outward, reverse
  if (innerprod < zero) {
    nx = -nx;
    ny = -ny;
  }

  // Scale to length unity
  inverselength = pow(nx*nx + ny*ny, -half);

  pTn2[n].x = nx*inverselength;
  pTn2[n].y = ny*inverselength;

  // Vector along face
  facedx = ax - bx;
  facedy = ay - by;

  // Vector orthogonal to face
  nx = facedy;
  ny = -facedx;

  // Check if pointing inward
  innerprod = nx*cx + ny*cy - half*(nx*ax + ny*ay + nx*bx + ny*by);

  // If pointing outward, reverse
  if (innerprod < zero) {
    nx = -nx;
    ny = -ny;
  }

  // Scale to length unity
  inverselength = pow(nx*nx + ny*ny, -half);

  pTn3[n].x = nx*inverselength;
  pTn3[n].y = ny*inverselength;

  triL[n].x = sqrt(Sq(bx - cx) + Sq(by - cy));
  triL[n].y = sqrt(Sq(ax - cx) + Sq(ay - cy));
  triL[n].z = sqrt(Sq(bx - ax) + Sq(by - ay));

  // Average radius (needed in cylindrical geometry). Note logarithmic x!
  pTrad[n] = (exp(ax) + exp(bx) + exp(cx))/(real) 3.0;
}

//######################################################################
/*! \brief Kernel calculating normals and edge lengths for all triangles

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVc Pointer to vertex coordinates
\param *pTn1 Pointer to triangle normals first edge (output)
\param *pTn2 Pointer to triangle normals second edge (output)
\param *pTn3 Pointer to triangle normals third edge (output)
\param *triL Pointer to array of triangle edge lengths (output)
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pTrad Pointer to average x coordinate (output)*/
//######################################################################

__global__ void
devCalcNormalEdge(int nTriangle, int3 *pTv, real2 *pVc,
                  real2 *pTn1, real2 *pTn2, real2 *pTn3,
                  real3 *triL, int nVertex,
                  real Px, real Py, real *pTrad)
{
  // n = triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcNormalEdgeSingle(n, nTriangle, pTv, pVc,
                         pTn1, pTn2, pTn3,
                         triL, nVertex, Px, Py, pTrad);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate inward-pointing normals (length unity) and edge lengths for all
triangles in Mesh*/
//######################################################################

void Mesh::CalcNormalEdge()
{
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  triangleEdgeNormals->SetSize(nTriangle);
  real2 *pTn1 = triangleEdgeNormals->GetPointer(0);
  real2 *pTn2 = triangleEdgeNormals->GetPointer(1);
  real2 *pTn3 = triangleEdgeNormals->GetPointer(2);

  triangleEdgeLength->SetSize(nTriangle);
  real3 *triL = triangleEdgeLength->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  triangleAverageX->SetSize(nTriangle);
  real *pTrad = triangleAverageX->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcNormalEdge,
                                       (size_t) 0, 0);

    devCalcNormalEdge<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVc, pTn1, pTn2, pTn3, triL, nVertex, Px, Py, pTrad);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      CalcNormalEdgeSingle(n, nTriangle, pTv, pVc,
                           pTn1, pTn2, pTn3, triL,
                           nVertex, Px, Py, pTrad);
  }
}

}  // namespace astrix
