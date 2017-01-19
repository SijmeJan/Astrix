// -*-c++-*-
/*! \file boundary_reflect.cu
\brief Functions for setting reflecting boundary conditions

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

namespace astrix {

//#########################################################################
/*! \brief Reflecting boundaries for triangle \a n, edge \a e

Reflecting boundary conditions are implemented "weakly" by adding a corrective flux that counteracts any flow through the boundary.

\param n Triangle to consider
\param e Edge to consider; e must be equal to te1[n] or te2[n] or te3[n]
\param dt Time step
\param *pState Pointer to state vector
\param *pTv Pointer to triangle vertices
\param e1 First edge of triangle
\param e2 Second edge of triangle
\param e3 Third edge of triangle
\param *pVarea Pointer to array of vertex areas
\param *pTl Pointer to array of triangle edge lengths
\param *pTn1 Pointer to first edge normal of triangle
\param *pTn2 Pointer to first edge normal of triangle
\param *pTn3 Pointer to first edge normal of triangle
\param nVertex Total number of vertices in Mesh
\param G1 Ratio of specific heats - 1*/
//#########################################################################

__host__ __device__
void SetReflectingEdge(int n, int e, real dt, real4 *pState,
                       const int3 *pTv, int e1, int e2, int e3,
                       const real *pVarea, const real3 *pTl,
                       const real2 *pTn1, const real2 *pTn2,
                       const real2 *pTn3, int nVertex, real G1)
{
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  // Edge vertices
  int v1 = a;
  int v2 = b;
  int v3 = c;
  if (e == e2) {
    v1 = b;
    v2 = c;
    v3 = a;
  }
  if (e == e3) {
    v1 = c;
    v2 = a;
    v3 = b;
  }
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real nx = pTn1[n].x;
  real ny = pTn1[n].y;
  real edge_length = tl1;
  if (e == e3) {
    nx = pTn2[n].x;
    ny = pTn2[n].y;
    edge_length = tl2;
  }
  if (e == e1) {
    nx = pTn3[n].x;
    ny = pTn3[n].y;
    edge_length = tl3;
  }

  // Vertices belonging to edge e
  int vj = v1;
  int vk = v2;

  real Dj = pState[vj].x;
  real Uj = pState[vj].y/Dj;
  real Vj = pState[vj].z/Dj;
  real Ej = pState[vj].w;

  real Dk = pState[vk].x;
  real Uk = pState[vk].y/Dk;
  real Vk = pState[vk].z/Dk;
  real Ek = pState[vk].w;

  // Velocities normal to edge
  real vnj = Uj*nx + Vj*ny;
  real vnk = Uk*nx + Vk*ny;

  // Correction fluxes
  real Fcorrj0 = Dj*vnj;
  real Fcorrj1 = Dj*Uj*vnj;
  real Fcorrj2 = Dj*Vj*vnj;
  real Pj = G1*(Ej - half*Dj*(Sq(Uj) + Sq(Vj)));
  real Fcorrj3 = (Ej + Pj)*vnj;

  real Fcorrk0 = Dk*vnk;
  real Fcorrk1 = Dk*Uk*vnk;
  real Fcorrk2 = Dk*Vk*vnk;
  real Pk = G1*(Ek - half*Dk*(Sq(Uk) + Sq(Vk)));
  real Fcorrk3 = (Ek + Pk)*vnk;

  real A = (real) 0.75;
  real dtdx = dt*edge_length/pVarea[vj];
  pState[vj].x -= half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0);
  pState[vj].y -= half*dtdx*(A*Fcorrj1 + (one - A)*Fcorrk1);
  pState[vj].z -= half*dtdx*(A*Fcorrj2 + (one - A)*Fcorrk2);
  pState[vj].w -= half*dtdx*(A*Fcorrj3 + (one - A)*Fcorrk3);

  dtdx = dt*edge_length/pVarea[vk];
  pState[vk].x -= half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0);
  pState[vk].y -= half*dtdx*(A*Fcorrk1 + (one - A)*Fcorrj1);
  pState[vk].z -= half*dtdx*(A*Fcorrk2 + (one - A)*Fcorrj2);
  pState[vk].w -= half*dtdx*(A*Fcorrk3 + (one - A)*Fcorrj3);

  // Check for physical state
  real presj =
    G1*(pState[vj].w - half*(Sq(pState[vj].y) + Sq(pState[vj].z))/pState[vj].x);
  real presk =
    G1*(pState[vk].w - half*(Sq(pState[vk].y) + Sq(pState[vk].z))/pState[vk].x);

  if (presj < 0.0 || presk < 0.0 ||
      pState[vj].x < 0.0 || pState[vk].x < 0.0) {
    // Try A = 1.0
    real dtdx = dt*edge_length/pVarea[vj];
    pState[vj].x += half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0 - Fcorrj0);
    pState[vj].y += half*dtdx*(A*Fcorrj1 + (one - A)*Fcorrk1 - Fcorrj1);
    pState[vj].z += half*dtdx*(A*Fcorrj2 + (one - A)*Fcorrk2 - Fcorrj2);
    pState[vj].w += half*dtdx*(A*Fcorrj3 + (one - A)*Fcorrk3 - Fcorrj3);

    dtdx = dt*edge_length/pVarea[vk];
    pState[vk].x += half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0 - Fcorrk0);
    pState[vk].y += half*dtdx*(A*Fcorrk1 + (one - A)*Fcorrj1 - Fcorrk1);
    pState[vk].z += half*dtdx*(A*Fcorrk2 + (one - A)*Fcorrj2 - Fcorrk2);
    pState[vk].w += half*dtdx*(A*Fcorrk3 + (one - A)*Fcorrj3 - Fcorrk3);

#ifndef __CUDA_ARCH__
    presj = G1*(pState[vj].w - half*(Sq(pState[vj].y) +
                                     Sq(pState[vj].z))/pState[vj].x);
    presk = G1*(pState[vk].w - half*(Sq(pState[vk].y) +
                                     Sq(pState[vk].z))/pState[vk].x);

    if (presj < 0.0 || presk < 0.0 ||
        pState[vj].x < 0.0 || pState[vk].x < 0.0) {
      std::cout << "Negative pressure in reflection "
                << presj << " " << presk << std::endl;
      std::cout << Fcorrj0 << " " << Fcorrj1 << " "
                << Fcorrj2 << " " << Fcorrj3 << std::endl;
      std::cout << Fcorrk0 << " " << Fcorrk1 << " "
                << Fcorrk2 << " " << Fcorrk3 << std::endl;
      std::cout << vnj << " " << vnk << std::endl;
      std::cout << nx << " " << ny << std::endl;

      // Go back to old state
      real A = (real) 1.0;
      real dtdx = dt*edge_length/pVarea[vj];
      pState[vj].x += half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0);
      pState[vj].y += half*dtdx*(A*Fcorrj1 + (one - A)*Fcorrk1);
      pState[vj].z += half*dtdx*(A*Fcorrj2 + (one - A)*Fcorrk2);
      pState[vj].w += half*dtdx*(A*Fcorrj3 + (one - A)*Fcorrk3);

      dtdx = dt*edge_length/pVarea[vk];
      pState[vk].x += half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0);
      pState[vk].y += half*dtdx*(A*Fcorrk1 + (one - A)*Fcorrj1);
      pState[vk].z += half*dtdx*(A*Fcorrk2 + (one - A)*Fcorrj2);
      pState[vk].w += half*dtdx*(A*Fcorrk3 + (one - A)*Fcorrj3);

      presj = G1*(pState[vj].w - half*(Sq(pState[vj].y) +
                                       Sq(pState[vj].z))/pState[vj].x);
      presk = G1*(pState[vk].w - half*(Sq(pState[vk].y) +
                                       Sq(pState[vk].z))/pState[vk].x);

      std::cout << "Old pressures: " << presj << " " << presk << std::endl;

      int N = 10;
      for (int i = 0; i < N; i++) {
        real A = (real)i/(real) (N - 1);

        // Try update with A
        real dtdx = dt*edge_length/pVarea[vj];
        pState[vj].x -= half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0);
        pState[vj].y -= half*dtdx*(A*Fcorrj1 + (one - A)*Fcorrk1);
        pState[vj].z -= half*dtdx*(A*Fcorrj2 + (one - A)*Fcorrk2);
        pState[vj].w -= half*dtdx*(A*Fcorrj3 + (one - A)*Fcorrk3);

        dtdx = dt*edge_length/pVarea[vk];
        pState[vk].x -= half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0);
        pState[vk].y -= half*dtdx*(A*Fcorrk1 + (one - A)*Fcorrj1);
        pState[vk].z -= half*dtdx*(A*Fcorrk2 + (one - A)*Fcorrj2);
        pState[vk].w -= half*dtdx*(A*Fcorrk3 + (one - A)*Fcorrj3);

        presj = G1*(pState[vj].w - half*(Sq(pState[vj].y) +
                                         Sq(pState[vj].z))/pState[vj].x);
        presk = G1*(pState[vk].w - half*(Sq(pState[vk].y) +
                                         Sq(pState[vk].z))/pState[vk].x);

        // Go back to old state
        dtdx = dt*edge_length/pVarea[vj];
        pState[vj].x += half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0);
        pState[vj].y += half*dtdx*(A*Fcorrj1 + (one - A)*Fcorrk1);
        pState[vj].z += half*dtdx*(A*Fcorrj2 + (one - A)*Fcorrk2);
        pState[vj].w += half*dtdx*(A*Fcorrj3 + (one - A)*Fcorrk3);

        dtdx = dt*edge_length/pVarea[vk];
        pState[vk].x += half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0);
        pState[vk].y += half*dtdx*(A*Fcorrk1 + (one - A)*Fcorrj1);
        pState[vk].z += half*dtdx*(A*Fcorrk2 + (one - A)*Fcorrj2);
        pState[vk].w += half*dtdx*(A*Fcorrk3 + (one - A)*Fcorrj3);

        std::cout << A << " " << presj << " " << presk << std::endl;
      }
    }
#endif
  }
}

__host__ __device__
void SetReflectingEdge(int n, int e, real dt, real *pState,
                       const int3 *pTv, int e1, int e2, int e3,
                       const real *pVarea, const real3 *pTl,
                       const real2 *pTn1, const real2 *pTn2,
                       const real2 *pTn3, int nVertex, real G1)
{
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  // Edge vertices
  int v1 = a;
  int v2 = b;
  int v3 = c;
  if (e == e2) {
    v1 = b;
    v2 = c;
    v3 = a;
  }
  if (e == e3) {
    v1 = c;
    v2 = a;
    v3 = b;
  }
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real nx = pTn1[n].x;
  // real ny = pTn1[n].y;
  real edge_length = tl1;
  if (e == e3) {
    nx = pTn2[n].x;
    // ny = pTn2[n].y;
    edge_length = tl2;
  }
  if (e == e1) {
    nx = pTn3[n].x;
    // ny = pTn3[n].y;
    edge_length = tl3;
  }

  // Vertices belonging to edge e
  int vj = v1;
  int vk = v2;

  real Dj = pState[vj];
  real Dk = pState[vk];

  // Velocities normal to edge (EQ_SOLVE)
  real vnj = nx;
  real vnk = nx;

  // Correction fluxes
  real Fcorrj0 = Dj*vnj;
  real Fcorrk0 = Dk*vnk;

  real A = (real) 0.75;
  real dtdx = dt*edge_length/pVarea[vj];
  pState[vj] -= half*dtdx*(A*Fcorrj0 + (one - A)*Fcorrk0);

  dtdx = dt*edge_length/pVarea[vk];
  pState[vk] -= half*dtdx*(A*Fcorrk0 + (one - A)*Fcorrj0);
}

//#########################################################################
/*! \brief Reflecting boundaries for triangle \a n

Reflecting boundary conditions are implemented "weakly" by adding a corrective flux that counteracts any flow through the boundary.

\param n Triangle to consider
\param dt Time step
\param *pState Pointer to state vector
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVarea Pointer to array of vertex areas
\param *pTl Pointer to array of triangle edge lengths
\param *pTn1 Pointer to first edge normal of triangle
\param *pTn2 Pointer to first edge normal of triangle
\param *pTn3 Pointer to first edge normal of triangle
\param nVertex Total number of vertices in Mesh
\param G1 Ratio of specific heats - 1*/
//#########################################################################

__host__ __device__
void SetReflectingSingle(int n, real dt, realNeq *pState, const int3 *pTv,
                         const int3* __restrict__ pTe,
                         const int2* __restrict__ pEt,
                         const real *pVarea, const real3 *pTl,
                         const real2 *pTn1, const real2 *pTn2,
                         const real2 *pTn3, int nVertex, real G1)
{
  int e1 = pTe[n].x;
  int e2 = pTe[n].y;
  int e3 = pTe[n].z;

  int t11 = pEt[e1].x;
  int t21 = pEt[e1].y;

  if (t11 == -1 || t21 == -1)
    SetReflectingEdge(n, e1, dt, pState,
                      pTv, e1, e2, e3,
                      pVarea, pTl,
                      pTn1, pTn2, pTn3,
                      nVertex, G1);

  int t12 = pEt[e2].x;
  int t22 = pEt[e2].y;

  if (t12 == -1 || t22 == -1)
    SetReflectingEdge(n, e2, dt, pState,
                      pTv, e1, e2, e3,
                      pVarea, pTl,
                      pTn1, pTn2, pTn3,
                      nVertex, G1);

  int t13 = pEt[e3].x;
  int t23 = pEt[e3].y;

  if (t13 == -1 || t23 == -1)
    SetReflectingEdge(n, e3, dt, pState,
                      pTv, e1, e2, e3,
                      pVarea, pTl,
                      pTn1, pTn2, pTn3,
                      nVertex, G1);
}

//############################################################################
/*! \brief Kernel setting reflecting boundaries

Reflecting boundary conditions are implemented "weakly" by adding a corrective flux that counteracts any flow through the boundary.

\param dt Time step
\param *pState Pointer to state vector
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVarea Pointer to array of vertex areas
\param *pTl Pointer to array of triangle edge lengths
\param *pTn1 Pointer to first edge normal of triangle
\param *pTn2 Pointer to first edge normal of triangle
\param *pTn3 Pointer to first edge normal of triangle
\param nTriangle Total number of triangles in Mesh
\param nVertex Total number of vertices in Mesh
\param G1 Ratio of specific heats - 1*/
//############################################################################

__global__ void
devSetReflecting(real dt, realNeq *pState, const int3 *pTv,
                 const int3* __restrict__ pTe,
                 const int2* __restrict__ pEt,
                 const real *pVarea, const real3 *pTl,
                 const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                 int nTriangle, int nVertex, real G1)
{
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    SetReflectingSingle(n, dt, pState,
                        pTv, pTe, pEt,
                        pVarea, pTl,
                        pTn1, pTn2, pTn3,
                        nVertex, G1);

    n += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! Reflecting boundary conditions are implemented "weakly" by adding a corrective flux that counteracts any flow through the boundary.

  \param dt Time step*/
//#########################################################################

void Simulation::ReflectingBoundaries(real dt)
{
  realNeq *pState = vertexState->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const int3 *pTe = mesh->TriangleEdgesData();
  const int2 *pEt = mesh->EdgeTrianglesData();
  const real *pVarea = mesh->VertexAreaData();
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetReflecting,
                                       (size_t) 0, 0);

    devSetReflecting<<<nBlocks, nThreads>>>
      (dt, pState, pTv, pTe, pEt, pVarea, pTl,
       pTn1, pTn2, pTn3, nTriangle, nVertex, specificHeatRatio - 1.0);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      SetReflectingSingle(n, dt, pState,
                          pTv, pTe, pEt, pVarea, pTl,
                          pTn1, pTn2, pTn3,
                          nVertex, specificHeatRatio - 1.0);
  }
}

}  // namespace astrix
