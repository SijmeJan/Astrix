// -*-c++-*-
/*! \file replaceLDA.cu
\brief File containing functions to replace LDA residue with N in case of unphysical state.

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
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
/*! \brief Replace LDA with N if unphysical state at triangle n

If any of the vertices of \a n has an unphysical state, replace the triangle residue with N only. If we are at the second stage of the Runge Kutta integration, just set all residues to zero, forcing a first-order update

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pVuf Pointer to array of flags indicating whether vertex has an unphysical state
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param RKStep Stage of Runge-Kutta integration
\param nVertex Total number of vertices in Mesh*/
//######################################################################

__host__ __device__
void SingleReplaceLDA(int n, const int3* __restrict__ pTv,
                      const int* __restrict__ pVuf,
                      real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                      real4 *pTresLDA0, real4 *pTresLDA1, real4 *pTresLDA2,
                      const int RKStep, const int nVertex)
{
  //const real zero = (real) 0.0;

  // Triangle vertices
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Check whether any vertex state is flagged as unphysical
  int correct_flag = 0;
  correct_flag += pVuf[a];
  correct_flag += pVuf[b];
  correct_flag += pVuf[c];

  // If any vertex has unphysical state, replace LDA with N
  if (correct_flag > 0) {
    /*
    // If we are in the second-order update, just ignore residual
    if (RKStep == 1) {
      pTresN0[n].x = zero;
      pTresN0[n].y = zero;
      pTresN0[n].z = zero;
      pTresN0[n].w = zero;
      pTresN1[n].x = zero;
      pTresN1[n].y = zero;
      pTresN1[n].z = zero;
      pTresN1[n].w = zero;
      pTresN2[n].x = zero;
      pTresN2[n].y = zero;
      pTresN2[n].z = zero;
      pTresN2[n].w = zero;
    }
    */
    // Replace LDA
    pTresLDA0[n] = pTresN0[n];
    pTresLDA1[n] = pTresN1[n];
    pTresLDA2[n] = pTresN2[n];
  }
}

//! Version for three equations
__host__ __device__
void SingleReplaceLDA(int n, const int3* __restrict__ pTv,
                      const int* __restrict__ pVuf,
                      real3 *pTresN0, real3 *pTresN1, real3 *pTresN2,
                      real3 *pTresLDA0, real3 *pTresLDA1, real3 *pTresLDA2,
                      const int RKStep, const int nVertex)
{
  // Triangle vertices
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Check whether any vertex state is flagged as unphysical
  int correct_flag = 0;
  correct_flag += pVuf[a];
  correct_flag += pVuf[b];
  correct_flag += pVuf[c];

  // If any vertex has unphysical state, replace LDA with N
  if (correct_flag > 0) {
    // Replace LDA
    pTresLDA0[n] = pTresN0[n];
    pTresLDA1[n] = pTresN1[n];
    pTresLDA2[n] = pTresN2[n];
  }
}

//! Version for single equation
__host__ __device__
void SingleReplaceLDA(int n, const int3* __restrict__ pTv,
                      const int* __restrict__ pVuf,
                      real *pTresN0, real *pTresN1, real *pTresN2,
                      real *pTresLDA0, real *pTresLDA1, real *pTresLDA2,
                      const int RKStep, const int nVertex)
{
  // Dummy function; nothing to do if solving only one equation
}

//######################################################################
/*! \brief Kernel replacing LDA with N if unphysical state at triangle

If any of the vertices of a triangle has an unphysical state, replace the triangle residue with N only. If we are at the second stage of the Runge Kutta integration, just set all residues to zero, forcing a first-order update

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVuf Pointer to array of flags indicating whether vertex has an unphysical state
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param RKStep Stage of Runge-Kutta integration
\param nVertex Total number of vertices in Mesh*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devReplaceLDA(int nTriangle,
              const int3* __restrict__ pTv,
              const int* __restrict__ pVuf,
              realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
              realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
              int RKStep, int nVertex)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    SingleReplaceLDA(n, pTv, pVuf,
                     pTresN0, pTresN1, pTresN2,
                     pTresLDA0, pTresLDA1, pTresLDA2,
                     RKStep, nVertex);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! If any of the vertices of a triangle has an unphysical state, replace the triangle residue with N only. If we are at the second stage of the Runge Kutta integration, just set all residues to zero, forcing a first-order update.

\param *vertexUnphysicalFlag Pointer to array of flags indicating whether vertex has an unphysical state
\param RKStep Stage of Runge-Kutta integration*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                         int RKStep)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  const int3 *pTv = mesh->TriangleVerticesData();

  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);
  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);

  int *pVuf = vertexUnphysicalFlag->GetPointer();

  // Replace LDA with N where necessary
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devReplaceLDA<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devReplaceLDA<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVuf,
       pTresN0, pTresN1, pTresN2,
       pTresLDA0, pTresLDA1, pTresLDA2,
       RKStep, nVertex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      SingleReplaceLDA(n, pTv, pVuf,
                       pTresN0, pTresN1, pTresN2,
                       pTresLDA0, pTresLDA1, pTresLDA2,
                       RKStep, nVertex);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template
void Simulation<real,
                CL_ADVECT>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                       int RKStep);
template
void Simulation<real,
                CL_BURGERS>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                        int RKStep);
template
void Simulation<real3,
                CL_CART_ISO>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                         int RKStep);
template
void Simulation<real3,
                CL_CYL_ISO>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                        int RKStep);
template
void Simulation<real4,
                CL_CART_EULER>::ReplaceLDA(Array<int> *vertexUnphysicalFlag,
                                           int RKStep);

}  // namespace astrix
