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
#include "./Param/simulationparameter.h"

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
void AddResidueSingle(int n,
                      const int3* __restrict__ pTv,
                      const real3 *pTl,
                      const real *pVarea,
                      real *pShock, real4 *pState, real4 *pTresTot,
                      real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                      real4 *pTresLDA0, real4 *pTresLDA1, real4 *pTresLDA2,
                      real dt, int nVertex, IntegrationScheme intScheme,
                      int setToMinMaxFlag)
{
  const real one = (real) 1.0;
  const real small = (real) 1.0e-10;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real lb0 = one;
  real lb1 = one;
  real lb2 = one;
  real lb3 = one;

  if (intScheme == SCHEME_B) {
    real resN0 = pTresN0[n].x;
    real resN1 = pTresN0[n].y;
    real resN2 = pTresN0[n].z;
    real resN3 = pTresN0[n].w;

    real blend0 = fabs(resN0)*tl1;
    real blend1 = fabs(resN1)*tl1;
    real blend2 = fabs(resN2)*tl1;
    real blend3 = fabs(resN3)*tl1;

    resN0 = pTresN1[n].x;
    resN1 = pTresN1[n].y;
    resN2 = pTresN1[n].z;
    resN3 = pTresN1[n].w;

    blend0 += fabs(resN0)*tl1;
    blend1 += fabs(resN1)*tl1;
    blend2 += fabs(resN2)*tl1;
    blend3 += fabs(resN3)*tl1;

    resN0 = pTresN2[n].x;
    resN1 = pTresN2[n].y;
    resN2 = pTresN2[n].z;
    resN3 = pTresN2[n].w;

    blend0 += fabs(resN0)*tl1;
    blend1 += fabs(resN1)*tl1;
    blend2 += fabs(resN2)*tl1;
    blend3 += fabs(resN3)*tl1;

    real resTot0 = pTresTot[n].x;
    real resTot1 = pTresTot[n].y;
    real resTot2 = pTresTot[n].z;
    real resTot3 = pTresTot[n].w;

    blend0 = fabs(resTot0)/(blend0 + small);
    blend1 = fabs(resTot1)/(blend1 + small);
    blend2 = fabs(resTot2)/(blend2 + small);
    blend3 = fabs(resTot3)/(blend3 + small);

    // Set all to minimum
    if (setToMinMaxFlag == -1) {
      real blendMin = min(blend0, min(blend1, min(blend2, blend3)));
      blend0 = blendMin;
      blend1 = blendMin;
      blend2 = blendMin;
      blend3 = blendMin;
    }

    // Set all to maximum
    if (setToMinMaxFlag == 1) {
      real blendMax = max(blend0, max(blend1, max(blend2, blend3)));
      blend0 = blendMax;
      blend1 = blendMax;
      blend2 = blendMax;
      blend3 = blendMax;
    }

    lb0 = blend0;
    lb1 = blend1;
    lb2 = blend2;
    lb3 = blend3;
  }

  if (intScheme == SCHEME_BX) {
    lb0 = pShock[n];
    lb1 = lb0;
    lb2 = lb0;
    lb3 = lb0;
  }

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
void AddResidueSingle(int n,
                      const int3* __restrict__ pTv,
                      const real3 *pTl,
                      const real *pVarea,
                      real *pShock, real3 *pState, real3 *pTresTot,
                      real3 *pTresN0, real3 *pTresN1, real3 *pTresN2,
                      real3 *pTresLDA0, real3 *pTresLDA1, real3 *pTresLDA2,
                      real dt, int nVertex, IntegrationScheme intScheme,
                      int setToMinMaxFlag)
{
  const real one = (real) 1.0;
  const real small = (real) 1.0e-10;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real lb0 = one;
  real lb1 = one;
  real lb2 = one;

  if (intScheme == SCHEME_B) {
    real resN0 = pTresN0[n].x;
    real resN1 = pTresN0[n].y;
    real resN2 = pTresN0[n].z;

    real blend0 = fabs(resN0)*tl1;
    real blend1 = fabs(resN1)*tl1;
    real blend2 = fabs(resN2)*tl1;

    resN0 = pTresN1[n].x;
    resN1 = pTresN1[n].y;
    resN2 = pTresN1[n].z;

    blend0 += fabs(resN0)*tl1;
    blend1 += fabs(resN1)*tl1;
    blend2 += fabs(resN2)*tl1;

    resN0 = pTresN2[n].x;
    resN1 = pTresN2[n].y;
    resN2 = pTresN2[n].z;

    blend0 += fabs(resN0)*tl1;
    blend1 += fabs(resN1)*tl1;
    blend2 += fabs(resN2)*tl1;

    real resTot0 = pTresTot[n].x;
    real resTot1 = pTresTot[n].y;
    real resTot2 = pTresTot[n].z;

    blend0 = fabs(resTot0)/(blend0 + small);
    blend1 = fabs(resTot1)/(blend1 + small);
    blend2 = fabs(resTot2)/(blend2 + small);

    // Set all to minimum
    if (setToMinMaxFlag == -1) {
      real blendMin = min(blend0, min(blend1, blend2));
      blend0 = blendMin;
      blend1 = blendMin;
      blend2 = blendMin;
    }

    // Set all to maximum
    if (setToMinMaxFlag == 1) {
      real blendMax = max(blend0, max(blend1, blend2));
      blend0 = blendMax;
      blend1 = blendMax;
      blend2 = blendMax;
    }

    lb0 = blend0;
    lb1 = blend1;
    lb2 = blend2;
  }

  if (intScheme == SCHEME_BX) {
    lb0 = pShock[n];
    lb1 = lb0;
    lb2 = lb0;
  }

  real res0 = one;
  real res1 = one;
  real res2 = one;

  real dtdx = dt*tl1/pVarea[a];
  real dW;

  if (intScheme == SCHEME_N) {
    res0 = pTresN0[n].x;
    res1 = pTresN0[n].y;
    res2 = pTresN0[n].z;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA0[n].x;
    res1 = pTresLDA0[n].y;
    res2 = pTresLDA0[n].z;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN0[n].x;
    real resN1 = pTresN0[n].y;
    real resN2 = pTresN0[n].z;
    real resLDA0 = pTresLDA0[n].x;
    real resLDA1 = pTresLDA0[n].y;
    real resLDA2 = pTresLDA0[n].z;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[a].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[a].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[a].z), dW);

  dtdx = dt*tl2/pVarea[b];

  if (intScheme == SCHEME_N) {
    res0 = pTresN1[n].x;
    res1 = pTresN1[n].y;
    res2 = pTresN1[n].z;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA1[n].x;
    res1 = pTresLDA1[n].y;
    res2 = pTresLDA1[n].z;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN1[n].x;
    real resN1 = pTresN1[n].y;
    real resN2 = pTresN1[n].z;
    real resLDA0 = pTresLDA1[n].x;
    real resLDA1 = pTresLDA1[n].y;
    real resLDA2 = pTresLDA1[n].z;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[b].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[b].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[b].z), dW);

  dtdx = dt*tl3/pVarea[c];

  if (intScheme == SCHEME_N) {
    res0 = pTresN2[n].x;
    res1 = pTresN2[n].y;
    res2 = pTresN2[n].z;
  }

  if (intScheme == SCHEME_LDA) {
    res0 = pTresLDA2[n].x;
    res1 = pTresLDA2[n].y;
    res2 = pTresLDA2[n].z;
  }

  if (intScheme == SCHEME_B || intScheme == SCHEME_BX) {
    real resN0 = pTresN2[n].x;
    real resN1 = pTresN2[n].y;
    real resN2 = pTresN2[n].z;
    real resLDA0 = pTresLDA2[n].x;
    real resLDA1 = pTresLDA2[n].y;
    real resLDA2 = pTresLDA2[n].z;

    res0 = lb0*resN0 + (one - lb0)*resLDA0;
    res1 = lb1*resN1 + (one - lb1)*resLDA1;
    res2 = lb2*resN2 + (one - lb2)*resLDA2;
  }

  dW = -dtdx*res0;
  AtomicAdd(&(pState[c].x), dW);
  dW = -dtdx*res1;
  AtomicAdd(&(pState[c].y), dW);
  dW = -dtdx*res2;
  AtomicAdd(&(pState[c].z), dW);
}

__host__ __device__
void AddResidueSingle(int n, const int3* __restrict__ pTv,
                      const real3 *pTl, const real *pVarea,
                      real *pShock, real *pState, real *pTresTot,
                      real *pTresN0, real *pTresN1, real *pTresN2,
                      real *pTresLDA0, real *pTresLDA1, real *pTresLDA2,
                      real dt, int nVertex, IntegrationScheme intScheme,
                      int setToMinMaxFlag)
{
  const real one = (real) 1.0;
  const real small = (real) 1.0e-10;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real lb0 = one;

  if (intScheme == SCHEME_B) {
    real resN = pTresN0[n];
    real blend = fabs(resN)*tl1;
    resN = pTresN1[n];
    blend += fabs(resN)*tl1;
    resN = pTresN2[n];
    blend += fabs(resN)*tl1;

    real resTot = pTresTot[n];
    blend = fabs(resTot)/(blend + small);

    lb0 = blend;
  }

  if (intScheme == SCHEME_BX) lb0 = pShock[n];


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
              const real *pVarea, real *pShock,
              realNeq *pState, realNeq *pTresTot,
              realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
              realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
              real dt, int nVertex, IntegrationScheme intScheme,
              int setToMinMaxFlag)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    AddResidueSingle(n, pTv, pTl, pVarea, pShock, pState,
                     pTresTot, pTresN0, pTresN1, pTresN2,
                     pTresLDA0, pTresLDA1, pTresLDA2,
                     dt, nVertex, intScheme, setToMinMaxFlag);

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

  realNeq *pTresTot = triangleResidueTotal->GetPointer();

  real *pShock = triangleShockSensor->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *triL = mesh->TriangleEdgeLengthData();
  const real *vertArea = mesh->VertexAreaData();

  IntegrationScheme intScheme = simulationParameter->intScheme;
  int preferMinMaxBlend = simulationParameter->preferMinMaxBlend;

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
      (nTriangle, pTv, triL, vertArea, pShock, state, pTresTot,
       pTresN0, pTresN1, pTresN2, pTresLDA0, pTresLDA1, pTresLDA2,
       dt, nVertex, intScheme, preferMinMaxBlend);
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
      AddResidueSingle(n, pTv, triL, vertArea, pShock, state, pTresTot,
                       pTresN0, pTresN1, pTresN2,
                       pTresLDA0, pTresLDA1, pTresLDA2,
                       dt, nVertex, intScheme, preferMinMaxBlend);
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
