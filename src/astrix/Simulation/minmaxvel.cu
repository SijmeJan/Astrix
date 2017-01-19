// -*-c++-*-
/*! \file minmaxvel.cu
\brief File containing functions to find minimum and maximum velocity in the Simulation

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {

//#########################################################################
/*! \brief Find minimum/maximum velocity at vertex \a i

\param i Vertex to consider
\param *pState Pointer to vertex state
\param *pMinVel Pointer to minimum velocities (output)
\param *pMinVel Pointer to maximum velocities (output)*/
//#########################################################################

__host__ __device__
void FillMinMaxVelocitySingle(unsigned int i, real4 *pState,
                              real *pMinVel, real *pMaxVel)
{
  real dens = pState[i].x;
  real momx = pState[i].y;
  real momy = pState[i].z;

  // Assume maximum is x-velocity
  real vMax = momx/dens;
  real vMin = momy/dens;

  // Swap if necessary
  if (vMin > vMax) {
    real vTemp = vMax;
    vMax = vMin;
    vMin = vTemp;
  }

  // Output
  pMinVel[i] = vMin;
  pMaxVel[i] = vMax;
}

__host__ __device__
void FillMinMaxVelocitySingle(unsigned int i, real *pState,
                              real *pMinVel, real *pMaxVel)
{
#if BURGERS == 1
  pMinVel[i] = pState[i];
  pMaxVel[i] = pState[i];
#else
  pMinVel[i] = (real) 1.0;
  pMaxVel[i] = (real) 1.0;
#endif
}

//#########################################################################
/*! \brief Fill minimum/maximum velocity arrays

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to vertex state
\param *pMinVel Pointer to minimum velocities (output)
\param *pMinVel Pointer to maximum velocities (output)*/
//#########################################################################

__global__ void
devFillMinMaxVelocity(unsigned int nVertex, realNeq *pState,
                      real *pMinVel, real *pMaxVel)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    FillMinMaxVelocitySingle(i, pState, pMinVel, pMaxVel);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! \brief Find minimum and maximum velocity on Mesh

\param minVel Will contain minimum velocity (output)
\param maxVel Will contain maximum velocity (output)*/
//#########################################################################

void Simulation::FindMinMaxVelocity(real& minVel, real& maxVel)
{
  unsigned int nVertex = mesh->GetNVertex();

  // State at vertices
  realNeq *pState = vertexState->GetPointer();

  // Arrays containing minimum/maximum velocity for each vertex
  Array<real> *minVelocity = new Array<real>(1, cudaFlag, nVertex);
  Array<real> *maxVelocity = new Array<real>(1, cudaFlag, nVertex);

  real *pMinVel = minVelocity->GetPointer();
  real *pMaxVel = maxVelocity->GetPointer();

  // Fill minimum/maximum velocity arrays
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillMinMaxVelocity,
                                       (size_t) 0, 0);

    devFillMinMaxVelocity<<<nBlocks, nThreads>>>
      (nVertex, pState, pMinVel, pMaxVel);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int i = 0; i < nVertex; i++)
      FillMinMaxVelocitySingle(i, pState, pMinVel, pMaxVel);
  }

  // Find minimum/maximum velocity
  minVel = minVelocity->Minimum();
  maxVel = maxVelocity->Maximum();

  delete minVelocity;
  delete maxVelocity;
}

}  // namespace astrix
