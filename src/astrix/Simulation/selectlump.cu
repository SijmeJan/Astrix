// -*-c++-*-
/*! \file selectlump.cu

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
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "../Common/helper_math.h"

namespace astrix {

//######################################################################
/*! \brief Computing contribution of second mass matrix and selective lumping for single triangle

\param n Triangle to consider
\param dt Time step
\param massMatrix Mass matrix used
\param selectLumpFlag Flag whether to use selective lumping
\param *pTv Pointer to triangle vertices
\param *pDstate Pointer to state difference at vertices
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void SelectLumpSingle(int n, real dt, int massMatrix, int selectLumpFlag,
                      const int3* __restrict__ pTv, realNeq *pDstate,
                      realNeq *pTresLDA0, realNeq *pTresLDA1,
                      realNeq *pTresLDA2, realNeq *pTresN0,
                      realNeq *pTresN1, realNeq *pTresN2,
                      const real3 *pTl, int nVertex)
{
  real half = (real) 0.5;
  real two = (real) 2.0;

  real f = (real) 0.0;
  if (selectLumpFlag == 1) f += (real) 1.0/(real) 12.0;
  if (massMatrix == 2) f += (real) 1.0/ (real) 36.0;

  // Vertices belonging to triangle: 3 coalesced reads
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  realNeq dW0 = pDstate[v1];
  realNeq dW1 = pDstate[v2];
  realNeq dW2 = pDstate[v3];

  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  // |T|/(12*dt)
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*f/dt;

  realNeq ResLDA0 = (-two*dW0 + dW1 + dW2)*Adt;
  realNeq ResLDA1 = (dW0 - two*dW1 + dW2)*Adt;
  realNeq ResLDA2 = (dW0 + dW1 - two*dW2)*Adt;

  pTresLDA0[n] -= ResLDA0/Tl1;
  pTresLDA1[n] -= ResLDA1/Tl2;
  pTresLDA2[n] -= ResLDA2/Tl3;

  if (selectLumpFlag == 1) {
    pTresN0[n] -= ResLDA0/Tl1;
    pTresN1[n] -= ResLDA1/Tl2;
    pTresN2[n] -= ResLDA2/Tl3;
  }
}

//######################################################################
/*! \brief Kernel computing contribution of second mass matrix and selective lumping for all triangles

\param nTriangle Total number of triangles in Mesh
\param dt Time step
\param massMatrix Mass matrix used
\param selectLumpFlag Flag whether to use selective lumping
\param *pTv Pointer to triangle vertices
\param *pDstate Pointer to state difference at vertices
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devSelectLump(int nTriangle, real dt, int massMatrix, int selectLumpFlag,
              const int3* __restrict__ pTv, realNeq *pDstate,
              realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
              realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
              const real3 *pTl, int nVertex)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    SelectLumpSingle<realNeq, CL>(n, dt, massMatrix, selectLumpFlag,
                                  pTv, pDstate,
                                  pTresLDA0, pTresLDA1, pTresLDA2,
                                  pTresN0, pTresN1, pTresN2,
                                  pTl, nVertex);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Add contribution of selective lumping and the second mass matrix to residuals. If \a massMatrix is not equal to 2 and \a selectLumpFlag is not equal to 1 nothing happens.

\param dt Time step
\param massMatrix Mass matrix used
\param selectLumpFlag Flag whether to use selective lumping*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::SelectLump(real dt, int massMatrix,
                                         int selectLumpFlag)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pDstate = vertexStateDiff->GetPointer();
  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);
  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *pTl  = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSelectLump<realNeq, CL>,
                                       (size_t) 0, 0);

    devSelectLump<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, dt, massMatrix, selectLumpFlag,
       pTv, pDstate,
       pTresLDA0, pTresLDA1, pTresLDA2,
       pTresN0, pTresN1, pTresN2,
       pTl, nVertex);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      SelectLumpSingle<realNeq, CL>(n, dt, massMatrix, selectLumpFlag,
                                    pTv, pDstate,
                                    pTresLDA0, pTresLDA1, pTresLDA2,
                                    pTresN0, pTresN1, pTresN2,
                                    pTl, nVertex);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::SelectLump(real dt,
                                                      int massMatrix,
                                                      int selectLumpFlag);
template void Simulation<real, CL_BURGERS>::SelectLump(real dt,
                                                       int massMatrix,
                                                       int selectLumpFlag);
template void Simulation<real3, CL_CART_ISO>::SelectLump(real dt,
                                                         int massMatrix,
                                                         int selectLumpFlag);
template void Simulation<real4, CL_CART_EULER>::SelectLump(real dt,
                                                           int massMatrix,
                                                           int selectLumpFlag);

}  // namespace astrix
