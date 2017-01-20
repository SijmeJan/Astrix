// -*-c++-*-
/*! \file blend.cu
\brief File containing function calculating blend factor

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
#include "../Common/profile.h"

namespace astrix {

//######################################################################
/*! \brief Calculate blend parameter for triangle \a n

\param n Triangle to consider
\param *pTl Pointer to triangle edge lengths
\param *pTresN0 Pointer to triangle residue N direction 0
\param *pTresN1 Pointer to triangle residue N direction 1
\param *pTresN2 Pointer to triangle residue N direction 2
\param *pTres Pointer to triangle total residue
\param *pBlend Pointer to blend parameter (output)*/
//######################################################################
/*
__host__ __device__
void CalcBlendSingle(int n, const real3* __restrict__ pTl,
                     real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                     real4 *pTres, real4 *pBlend, int setToMinMaxFlag)
{
  real small = (real) 1.0e-10;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Total residual
  real tTot0 = pTres[n].x;
  real tTot1 = pTres[n].y;
  real tTot2 = pTres[n].z;
  real tTot3 = pTres[n].w;

  // N residuals
  real tN00 = pTresN0[n].x;
  real tN01 = pTresN0[n].y;
  real tN02 = pTresN0[n].z;
  real tN03 = pTresN0[n].w;

  real tN10 = pTresN1[n].x;
  real tN11 = pTresN1[n].y;
  real tN12 = pTresN1[n].z;
  real tN13 = pTresN1[n].w;

  real tN20 = pTresN2[n].x;
  real tN21 = pTresN2[n].y;
  real tN22 = pTresN2[n].z;
  real tN23 = pTresN2[n].w;

  real blend0 = fabs(tTot0)/
    (fabs(tN00)*tl1 + fabs(tN10)*tl2 + fabs(tN20)*tl3 + small);
  real blend1 = fabs(tTot1)/
    (fabs(tN01)*tl1 + fabs(tN11)*tl2 + fabs(tN21)*tl3 + small);
  real blend2 = fabs(tTot2)/
    (fabs(tN02)*tl1 + fabs(tN12)*tl2 + fabs(tN22)*tl3 + small);
  real blend3 = fabs(tTot3)/
    (fabs(tN03)*tl1 + fabs(tN13)*tl2 + fabs(tN23)*tl3 + small);

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

  pBlend[n].x = blend0;
  pBlend[n].y = blend1;
  pBlend[n].z = blend2;
  pBlend[n].w = blend3;
}

__host__ __device__
void CalcBlendSingle(int n, const real3* __restrict__ pTl,
                     real *pTresN0, real *pTresN1, real *pTresN2,
                     real *pTres, real *pBlend, int setToMinMaxFlag)
{
  real small = (real) 1.0e-10;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Total residual
  real tTot = pTres[n];

  // N residuals
  real tN0 = pTresN0[n];
  real tN1 = pTresN1[n];
  real tN2 = pTresN2[n];

  // Calculate blend parameter
  pBlend[n] = fabs(tTot)/
    (fabs(tN0)*tl1 + fabs(tN1)*tl2 + fabs(tN2)*tl3 + small);
}
*/
//######################################################################
/*! \brief Kernel calculating blend parameter for all triangles

\param nTriangle Total number of triangles in Mesh
\param *pTl Pointer to triangle edge lengths
\param *pTresN0 Pointer to triangle residue N direction 0
\param *pTresN1 Pointer to triangle residue N direction 1
\param *pTresN2 Pointer to triangle residue N direction 2
\param *pTres Pointer to triangle total residue
\param *pBlend Pointer to blend parameter (output)*/

//######################################################################
/*
__global__ void
devCalcBlendFactor(int nTriangle, const real3* __restrict__ pTl,
                   realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
                   realNeq *pTres, realNeq *pBlend, int setToMinMaxFlag)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcBlendSingle(n, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend,
                    setToMinMaxFlag);

    n += blockDim.x*gridDim.x;
  }
}
*/
//######################################################################
/*! Calculate blend parameter for all triangles. When using the B scheme,
we need to blend the N and LDA residuals. This function calculates the
blend parameter \a triangleBlendFactor*/
//######################################################################
/*
void Simulation::CalcBlend()
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int nTriangle = mesh->GetNTriangle();

  // N residuals
  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  // Total residual
  realNeq *pTres = triangleResidueTotal->GetPointer();

  // Blend parameter (output)
  realNeq *pBlend = triangleBlendFactor->GetPointer();

  // Triangle edge lengths
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcBlendFactor,
                                       (size_t) 0, 0);

    // Execute kernel...
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcBlendFactor<<<nBlocks, nThreads>>>
      (nTriangle, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend,
       preferMinMaxBlend);
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
      CalcBlendSingle(n, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend,
                      preferMinMaxBlend);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("Blend.prof2", nTriangle, elapsedTime, cudaFlag);
#endif
}
*/
}  // namespace astrix
