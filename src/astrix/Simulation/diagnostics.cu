// -*-c++-*-
/*! \file diagnostics.cu
\brief Calculating various diagnostics for Simulations

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <fstream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//##############################################################################
//##############################################################################

__host__ __device__
void KineticEnergySingle(unsigned int i, const real *pVarea,
                         real4 *pState, real *Ex, real *Ey)
{
  real half = (real) 0.5;

  real d = pState[i].x;
  real m = pState[i].y;
  real n = pState[i].z;

  Ex[i] = pVarea[i]*half*m*m/d;
  Ey[i] = pVarea[i]*half*n*n/d;
}

__host__ __device__
void KineticEnergySingle(unsigned int i, const real *pVarea,
                         real3 *pState, real *Ex, real *Ey)
{
  real half = (real) 0.5;

  real d = pState[i].x;
  real m = pState[i].y;
  real n = pState[i].z;

  Ex[i] = pVarea[i]*half*m*m/d;
  Ey[i] = pVarea[i]*half*n*n/d;
}

  __host__ __device__
void KineticEnergySingle(unsigned int i, const real *pVarea,
                         real *pState, real *Ex, real *Ey)
{
  Ex[i] = (real) 0.0;
  Ey[i] = (real) 0.0;
}

//######################################################################
//######################################################################

__global__ void
devKineticEnergy(unsigned int nVertex, const real *pVarea,
                 realNeq *pState, real *Ex, real *Ey)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    KineticEnergySingle(n, pVarea, pState, Ex, Ey);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

real2 Simulation::KineticEnergy()
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  Array<real> *Ex = new Array<real>(1, cudaFlag, nVertex);
  Array<real> *Ey = new Array<real>(1, cudaFlag, nVertex);
  real *pEx = Ex->GetPointer();
  real *pEy = Ey->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devKineticEnergy,
                                       (size_t) 0, 0);

    devKineticEnergy<<<nBlocks, nThreads>>>
      (nVertex, pVarea, pState, pEx, pEy);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      KineticEnergySingle(n, pVarea, pState, pEx, pEy);
  }

  real ex = Ex->Sum();
  real ey = Ey->Sum();

  delete Ex;
  delete Ey;

  real2 ret;
  ret.x = ex;
  ret.y = ey;

  return ret;
}

//##############################################################################
//##############################################################################

__host__ __device__
void ThermalEnergySingle(unsigned int i, const real *pVarea,
                         real4 *pState, real *E)
{
  real half = (real) 0.5;

  real d = pState[i].x;
  real m = pState[i].y;
  real n = pState[i].z;
  real e = pState[i].w;

  E[i] = pVarea[i]*(e - half*(m*m + n*n)/d);
}

__host__ __device__
void ThermalEnergySingle(unsigned int i, const real *pVarea,
                         real3 *pState, real *E)
{
  E[i] = (real) 0.0;
}

__host__ __device__
void ThermalEnergySingle(unsigned int i, const real *pVarea,
                         real *pState, real *E)
{
  E[i] = (real) 0.0;
}

//######################################################################
//######################################################################

__global__ void
devThermalEnergy(unsigned int nVertex, const real *pVarea,
                 realNeq *pState, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    ThermalEnergySingle(n, pVarea, pState, E);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

real Simulation::ThermalEnergy()
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devKineticEnergy,
                                       (size_t) 0, 0);

    devThermalEnergy<<<nBlocks, nThreads>>>
      (nVertex, pVarea, pState, pE);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      ThermalEnergySingle(n, pVarea, pState, pE);
  }

  real e = E->Sum();

  delete E;

  return e;
}

//##############################################################################
//##############################################################################

__host__ __device__
void PotentialEnergySingle(unsigned int i, const real2 *pVc, const real *pVarea,
                           real4 *pState, real *E, ProblemDefinition problemDef)
{
  real d = pState[i].x;
  real vertY = pVc[i].y;

  real pot = 0.0;
  if (problemDef == PROBLEM_SOURCE)
    pot = 0.1*vertY;

  E[i] = pVarea[i]*d*pot;
}

//######################################################################
//######################################################################

__global__ void
devPotentialEnergy(unsigned int nVertex, const real2 *pVc, const real *pVarea,
                   realNeq *pState, real *E, ProblemDefinition problemDef)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    PotentialEnergySingle(n, pVc, pVarea, pState, E, problemDef);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

real Simulation::PotentialEnergy()
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  const real *pVarea = mesh->VertexAreaData();
  const real2 *pVc = mesh->VertexCoordinatesData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  ProblemDefinition p = simulationParameter->problemDef;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devPotentialEnergy,
                                       (size_t) 0, 0);

    devPotentialEnergy<<<nBlocks, nThreads>>>
      (nVertex, pVc, pVarea, pState, pE, p);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      PotentialEnergySingle(n, pVc, pVarea, pState, pE, p);
  }

  real e = E->Sum();

  delete E;

  return e;
}

//##############################################################################
//##############################################################################

__host__ __device__
void DensityErrorSingle(unsigned int i, const real *pVarea,
                        const real2 *pVc,
                        real4 *pState, real4 *pStateOld, real *E)
{
  real d = pState[i].x;
  real d0 = pStateOld[i].x;

  E[i] = pVarea[i]*std::abs(d - d0);
}

__host__ __device__
void DensityErrorSingle(unsigned int i, const real *pVarea,
                        const real2 *pVc,
                        real3 *pState, real3 *pStateOld, real *E)
{
  real d = pState[i].x;
  real d0 = pStateOld[i].x;

  E[i] = pVarea[i]*std::abs(d - d0);
}

__host__ __device__
void DensityErrorSingle(unsigned int i, const real *pVarea,
                        const real2 *pVc,
                        real *pState, real *pStateOld, real *E)
{
  real d = pState[i];
  real d0 = pStateOld[i];

  E[i] = pVarea[i]*std::abs(d - d0);
}

//######################################################################
//######################################################################

__global__ void
devDensityError(unsigned int nVertex, const real *pVarea,
                const real2 *pVc,
                realNeq *pState, realNeq *pStateOld, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    DensityErrorSingle(n, pVarea, pVc, pState, pStateOld, E);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

real Simulation::DensityError()
{
  real2 Ekin = KineticEnergy();
  real Eth = ThermalEnergy();

  vertexStateOld->SetEqual(vertexState);

  // Assume this gets the correct solution into vertexState
  SetInitial(simulationTime);

  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  realNeq *pStateOld = vertexStateOld->GetPointer();

  const real *pVarea = mesh->VertexAreaData();
  const real2 *pVc = mesh->VertexCoordinatesData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devDensityError,
                                       (size_t) 0, 0);

    devDensityError<<<nBlocks, nThreads>>>
      (nVertex, pVarea, pVc, pState, pStateOld, pE);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      DensityErrorSingle(n, pVarea, pVc, pState, pStateOld, pE);
  }

  real e = E->Sum()/mesh->GetTotalArea();

  delete E;

  vertexState->SetEqual(vertexStateOld);

  return e;
}

}  // namespace astrix
