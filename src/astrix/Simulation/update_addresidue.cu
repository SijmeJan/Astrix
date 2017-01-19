// -*-c++-*-
/*! \file update_addresidue.cu
\brief File containing functions for distributing residue over vertices

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"
#include "../Common/profile.h"

namespace astrix {

//######################################################################
/*! \brief Distribute residue of triangle \a n to its vertices

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pTl Pointer to triangle edge lengths
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pBlend Pointer to blend parameter
\param *pShock Pointer to shock sensor
\param *pState Pointer to state vector
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param dt Time step
\param nVertex Total number of vertices in Mesh
\param intScheme Integration scheme*/
//######################################################################

__host__ __device__
void AddResidueSingle(int n, const int3* __restrict__ pTv,
                      const real3 *pTl, const real *pVarea,
                      real4 *pBlend, real *pShock, real4 *pState,
                      real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                      real4 *pTresLDA0, real4 *pTresLDA1, real4 *pTresLDA2,
                      real dt, int nVertex, IntegrationScheme intScheme)
{
  const real one = (real) 1.0;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  real lb0 = one;
  real lb1 = one;
  real lb2 = one;
  real lb3 = one;

  if (intScheme == SCHEME_B) {
    lb0 = pBlend[n].x;
    lb1 = pBlend[n].y;
    lb2 = pBlend[n].z;
    lb3 = pBlend[n].w;
  }
  if (intScheme == SCHEME_BX) {
    lb0 = pShock[n];
    lb1 = lb0;
    lb2 = lb0;
    lb3 = lb0;
  }

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real res0 = one;
  real res1 = one;
  real res2 = one;
  real res3 = one;

  real dtdx = dt*tl1/pVarea[a];
  real dW;

  if (intScheme == SCHEME_N) {
    res0 = pTresN0[n].x;
    res1 = pTresN0[n].y;
    res2 = pTresN0[n].z;
    res3 = pTresN0[n].w;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA0[n].x;
    res1 = pTresLDA0[n].y;
    res2 = pTresLDA0[n].z;
    res3 = pTresLDA0[n].w;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN0[n].x;
    real resN1 = pTresN0[n].y;
    real resN2 = pTresN0[n].z;
    real resN3 = pTresN0[n].w;
    real resLDA0 = pTresLDA0[n].x;
    real resLDA1 = pTresLDA0[n].y;
    real resLDA2 = pTresLDA0[n].z;
    real resLDA3 = pTresLDA0[n].w;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
    res3 = lb3*resN3 + (one - lb3)*resLDA3;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[a].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[a].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[a].z), dW);
  dW = -dtdx*res3;
  AtomicAdd(&(pState[a].w), dW);

  dtdx = dt*tl2/pVarea[b];

  if (intScheme == SCHEME_N) {
    res0 = pTresN1[n].x;
    res1 = pTresN1[n].y;
    res2 = pTresN1[n].z;
    res3 = pTresN1[n].w;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA1[n].x;
    res1 = pTresLDA1[n].y;
    res2 = pTresLDA1[n].z;
    res3 = pTresLDA1[n].w;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN1[n].x;
    real resN1 = pTresN1[n].y;
    real resN2 = pTresN1[n].z;
    real resN3 = pTresN1[n].w;
    real resLDA0 = pTresLDA1[n].x;
    real resLDA1 = pTresLDA1[n].y;
    real resLDA2 = pTresLDA1[n].z;
    real resLDA3 = pTresLDA1[n].w;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
    res3 = lb3*resN3 + (one - lb3)*resLDA3;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[b].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[b].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[b].z), dW);
  dW = -dtdx*res3;
  AtomicAdd(&(pState[b].w), dW);

  dtdx = dt*tl3/pVarea[c];

  if (intScheme == SCHEME_N) {
    res0 = pTresN2[n].x;
    res1 = pTresN2[n].y;
    res2 = pTresN2[n].z;
    res3 = pTresN2[n].w;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA2[n].x;
    res1 = pTresLDA2[n].y;
    res2 = pTresLDA2[n].z;
    res3 = pTresLDA2[n].w;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN2[n].x;
    real resN1 = pTresN2[n].y;
    real resN2 = pTresN2[n].z;
    real resN3 = pTresN2[n].w;
    real resLDA0 = pTresLDA2[n].x;
    real resLDA1 = pTresLDA2[n].y;
    real resLDA2 = pTresLDA2[n].z;
    real resLDA3 = pTresLDA2[n].w;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
    res3 = lb3*resN3 + (one - lb3)*resLDA3;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[c].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[c].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[c].z), dW);
  dW = -dtdx*res3;
  AtomicAdd(&(pState[c].w), dW);
}

__host__ __device__
void AddResidueSingle(int n, const int3* __restrict__ pTv,
                      const real3 *pTl, const real *pVarea,
                      real *pBlend, real *pShock, real *pState,
                      real *pTresN0, real *pTresN1, real *pTresN2,
                      real *pTresLDA0, real *pTresLDA1, real *pTresLDA2,
                      real dt, int nVertex, IntegrationScheme intScheme)
{
  const real one = (real) 1.0;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  real lb0 = one;

  if (intScheme == SCHEME_B) lb0 = pBlend[n];
  if (intScheme == SCHEME_BX) lb0 = pShock[n];

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real res0 = one;

  real dtdx = dt*tl1/pVarea[a];
  real dW;

  if (intScheme == SCHEME_N) res0 = pTresN0[n];
  if (intScheme == SCHEME_LDA) res0 = pTresLDA0[n];
  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN0[n];
    real resLDA0 = pTresLDA0[n];
    res0 = lb0*resN0 + (one - lb0)*resLDA0;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[a]), dW);

  dtdx = dt*tl2/pVarea[b];

  if (intScheme == SCHEME_N) res0 = pTresN1[n];
  if (intScheme == SCHEME_LDA) res0 = pTresLDA1[n];
  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN1[n];
    real resLDA0 = pTresLDA1[n];
    res0 = lb0*resN0 + (one - lb0)*resLDA0;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[b]), dW);

  dtdx = dt*tl3/pVarea[c];

  if (intScheme == SCHEME_N) res0 = pTresN2[n];
  if (intScheme == SCHEME_LDA) res0 = pTresLDA2[n];
  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN2[n];
    real resLDA0 = pTresLDA2[n];
    res0 = lb0*resN0 + (one - lb0)*resLDA0;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[c]), dW);
}

//######################################################################
/*! \brief Distribute residue of triangles to their vertices

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pTl Pointer to triangle edge lengths
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pBlend Pointer to blend parameter
\param *pShock Pointer to shock sensor
\param *pState Pointer to state vector
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param dt Time step
\param nVertex Total number of vertices in Mesh
\param intScheme Integration scheme*/
//######################################################################

__global__ void
devAddResidue(int nTriangle, const int3* __restrict__ pTv, const real3 *pTl,
              const real *pVarea, realNeq *pBlend, real *pShock,
              realNeq *pState,
              realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
              realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
              real dt, int nVertex, IntegrationScheme intScheme)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    AddResidueSingle(n, pTv, pTl, pVarea, pBlend, pShock, pState,
                     pTresN0, pTresN1, pTresN2,
                     pTresLDA0, pTresLDA1, pTresLDA2,
                     dt, nVertex, intScheme);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Distribute triangle residuals over their vertices

\param dt Time step*/
//######################################################################

void Simulation::AddResidue(real dt)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *state    = vertexState->GetPointer();

  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);

  realNeq *pBlend = triangleBlendFactor->GetPointer();
  real *pShock = triangleShockSensor->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *triL = mesh->TriangleEdgeLengthData();
  const real *vertArea = mesh->VertexAreaData();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAddResidue,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    // Execute kernel...
    devAddResidue<<<nBlocks, nThreads>>>
      (nTriangle, pTv, triL, vertArea, pBlend, pShock, state,
       pTresN0, pTresN1, pTresN2, pTresLDA0, pTresLDA1, pTresLDA2,
       dt, nVertex, intScheme);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int n = 0; n < nTriangle; n++)
      AddResidueSingle(n, pTv, triL, vertArea, pBlend, pShock, state,
                       pTresN0, pTresN1, pTresN2,
                       pTresLDA0, pTresLDA1, pTresLDA2,
                       dt, nVertex, intScheme);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("AddResidual.prof2", nTriangle, elapsedTime, cudaFlag);
#endif
}

}  // namespace astrix
