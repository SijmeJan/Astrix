// -*-c++-*-
/*! \file limitupdate.cu
\brief File containing functions flag a vertex as unphysical is the change in one timestep is deemed too large.

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
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Check if vertex \a v experienced too large an update

\param v Vertex to consider
\param *pState Pointer to state at vertices
\param *pStateOld Pointer to old state at vertices
\param *pVertexLimitFlag Pointer to array of flags indicating whether change in state is small (0) or too big (1) (output)
\param G1 Ratio of specific heats - 1*/
//######################################################################

__host__ __device__
void FlagLimitVertex(const int v, real4 *pState, real4 *pStateOld,
                     int *pVertexLimitFlag, const real G1)
{
  const real half = (real) 0.5;

  // Assume everything is fine
  int ret = 0;

  real dens = pState[v].x;
  real momx = pState[v].y;
  real momy = pState[v].z;
  real ener = pState[v].w;

  // Pressure
  real p = G1*(ener - half*(Sq(momx) + Sq(momy))/dens);

  real densOld = pStateOld[v].x;
  real momxOld = pStateOld[v].y;
  real momyOld = pStateOld[v].z;
  real enerOld = pStateOld[v].w;

  // Pressure
  real pOld = G1*(enerOld - half*(Sq(momxOld) + Sq(momyOld))/densOld);

  if (abs(dens - densOld)/densOld > 0.1 ||
      abs(p - pOld)/pOld > 0.1)
    ret = 1;

  // Output flag
  if (ret == 1)
    pVertexLimitFlag[v] = ret;
}

__host__ __device__
void FlagLimitVertex(const int v, real3 *pState, real3 *pStateOld,
                     int *pVertexLimitFlag, const real G1)
{
  // Assume everything is fine
  int ret = 0;

  real dens = pState[v].x;
  real densOld = pStateOld[v].x;

  if (abs(dens - densOld)/densOld > 0.1)
    ret = 1;

  // Output flag
  if (ret == 1)
    pVertexLimitFlag[v] = ret;
}

__host__ __device__
void FlagLimitVertex(const int v, real *pState, real *pStateOld,
                     int *pVertexLimitFlag, const real G1)
{
  // Output flag
  pVertexLimitFlag[v] = 0;
}

//######################################################################
/*! \brief Kernel checking if vertices experienced too large an update

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state at vertices
\param *pStateOld Pointer to old state at vertices
\param *pVertexLimitFlag Pointer to array of flags indicating whether change in state is small (0) or too big (1) (output)
\param G1 Ratio of specific heats - 1*/
//######################################################################

__global__ void
devFlagLimit(const int nVertex, realNeq *pState, realNeq *pStateOld,
             int *pVertexLimitFlag, const real G1)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    FlagLimitVertex(n, pState, pStateOld, pVertexLimitFlag, G1);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Check all vertices for too large updates.

  \param *vertexLimitFlag Pointer to Array of flags indicating whether change in state is small (0) or too big (1) (output)*/
//######################################################################

void Simulation::FlagLimit(Array<int> *vertexLimitFlag)
{
  // Total number of vertices in Mesh
  int nVertex = mesh->GetNVertex();

  // State vector at vertices
  realNeq *state = vertexState->GetPointer();
  realNeq *stateOld = vertexStateOld->GetPointer();

  // Ratio of specific heats
  real G = simulationParameter->specificHeatRatio;

  // Pointer to output
  int *pVertexLimitFlag = vertexLimitFlag->GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFlagLimit,
                                       (size_t) 0, 0);

    // Execute kernel...
    devFlagLimit<<<nBlocks, nThreads>>>
      (nVertex, state, stateOld, pVertexLimitFlag, G - 1.0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int v = 0; v < nVertex; v++)
      FlagLimitVertex(v, state, stateOld, pVertexLimitFlag, G - 1.0);
  }
}

}  // namespace astrix
