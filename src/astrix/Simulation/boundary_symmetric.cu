// -*-c++-*-
/*! \file boundary_symmetric.cu
\brief Functions for setting symmetric boundary conditions

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./simulation.h"
#include "../Mesh/mesh.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Set symmetry boundary conditions on vertex \a a

Vertex \a a is located on a boundary with inward pointing normal \a n. Set the velocity parallel to \a n (perpendicular to the boundary) to zero, leaving alone the velocity perpendicular to \a n (parallel to the boundary).

\param a Vertex to consider
\param n Inward pointing normal to symmetry boundary
\param *pState Pointer to state
\param nVertex Total number of vertices in mesh*/
//#########################################################################

__host__ __device__
void SetSymmetricVertex(int a, real2 n, real4 *pState, int nVertex)
{
  // Set u*nx + v*ny = 0
  // Leave alone u*ny - v*nx = vpara
  //
  // u = vpara*ny
  // v = -vpara*nx

  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;

  real d = pState[a].x;
  real u = pState[a].y/d;
  real v = pState[a].z/d;

  real vPara = u*n.y - v*n.x;
  pState[a].y = d*vPara*n.y;
  pState[a].z = -d*vPara*n.x;
}

__host__ __device__
void SetSymmetricVertex(int a, real2 n, real3 *pState, int nVertex)
{
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;

  real d = pState[a].x;
  real u = pState[a].y/d;
  real v = pState[a].z/d;

  real vPara = u*n.y - v*n.x;
  pState[a].y = d*vPara*n.y;
  pState[a].z = -d*vPara*n.x;
}

__host__ __device__
void SetSymmetricVertex(int a, real2 n, real *pState, int nVertex)
{
  // Dummy: no symmetry boundaries for scalar equation
}

//#########################################################################
/*! Set symmetry boundary conditions on any boundary segment of triangle \a n

\param n Triangle to consider
\param *pState Pointer to state vector
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pTn1 Pointer to first edge normal of triangle
\param *pTn2 Pointer to first edge normal of triangle
\param *pTn3 Pointer to first edge normal of triangle
\param nVertex Total number of vertices in Mesh*/
//#########################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void SetSymmetricSingle(int n, realNeq *pState, const int3 *pTv,
                        const int3* __restrict__ pTe,
                        const int2* __restrict__ pEt,
                        const real2 *pTn1, const real2 *pTn2,
                        const real2 *pTn3, int nVertex)
{
  int e1 = pTe[n].x;
  int e2 = pTe[n].y;
  int e3 = pTe[n].z;

  int t11 = pEt[e1].x;
  int t21 = pEt[e1].y;


  if (t11 == -1 || t21 == -1) {
    SetSymmetricVertex(pTv[n].x, pTn3[n], pState, nVertex);
    SetSymmetricVertex(pTv[n].y, pTn3[n], pState, nVertex);
  }

  int t12 = pEt[e2].x;
  int t22 = pEt[e2].y;

  if (t12 == -1 || t22 == -1) {
    SetSymmetricVertex(pTv[n].y, pTn1[n], pState, nVertex);
    SetSymmetricVertex(pTv[n].z, pTn1[n], pState, nVertex);
  }


  int t13 = pEt[e3].x;
  int t23 = pEt[e3].y;

  if (t13 == -1 || t23 == -1) {
    SetSymmetricVertex(pTv[n].z, pTn2[n], pState, nVertex);
    SetSymmetricVertex(pTv[n].x, pTn2[n], pState, nVertex);
  }
}

//############################################################################
/*! Kernel setting symmetry boundary conditions at every segment in the mesh.

\param *pState Pointer to state vector
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pTn1 Pointer to first edge normal of triangle
\param *pTn2 Pointer to first edge normal of triangle
\param *pTn3 Pointer to first edge normal of triangle
\param nTriangle Total number of triangles in Mesh
\param nVertex Total number of vertices in Mesh*/
//############################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devSetSymmetry(realNeq *pState, const int3 *pTv,
               const int3* __restrict__ pTe,
               const int2* __restrict__ pEt,
               const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
               int nTriangle, int nVertex)
{
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    SetSymmetricSingle<realNeq, CL>(n, pState, pTv, pTe,
                                    pEt, pTn1, pTn2, pTn3, nVertex);

    n += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/* Set symmetry boundary conditions on any segment found in the mesh. For any segment found, the velocity perpendicular to the segment is set to zero, leaving alone the velocity parallel to the segment.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::SetSymmetricBoundaries()
{
  realNeq *pState = vertexState->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const int3 *pTe = mesh->TriangleEdgesData();
  const int2 *pEt = mesh->EdgeTrianglesData();
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetSymmetry<realNeq, CL>,
                                       (size_t) 0, 0);

    devSetSymmetry<realNeq, CL><<<nBlocks, nThreads>>>
      (pState, pTv, pTe, pEt,
       pTn1, pTn2, pTn3, nTriangle, nVertex);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      SetSymmetricSingle<realNeq, CL>(n, pState,
                                      pTv, pTe, pEt,
                                      pTn1, pTn2, pTn3,
                                      nVertex);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::SetSymmetricBoundaries();
template void Simulation<real, CL_BURGERS>::SetSymmetricBoundaries();
template void Simulation<real3, CL_CART_ISO>::SetSymmetricBoundaries();
template void Simulation<real3, CL_CYL_ISO>::SetSymmetricBoundaries();
template void Simulation<real4, CL_CART_EULER>::SetSymmetricBoundaries();

}  // namespace astrix
