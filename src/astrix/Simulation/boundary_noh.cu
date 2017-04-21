// -*-c++-*-
/*! \file boundary_noh.cu
\brief Functions for setting boundary conditions for 2D Noh problem

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

//######################################################################
/*! \brief Set Noh boundaries single vertex \a n

At the boundaries of the 2D Noh problem the pressure and velocities are the same as the initial condition, while the density evolves as 1 + t/r

\param n Vertex to consider
\param *pState Pointer to state vector
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param simulationTime Current simulation time
\param iG1 1/(Ratio of specific heats - 1)*/
//######################################################################

__host__ __device__
void SetBoundaryNohVertex(int n, real4 *pState,
                          const real2 *pVc, const int *pVbf,
                          real simulationTime, real iG1)
{
  real one = (real) 1.0;
  real half = (real) 0.5;

  if (pVbf[n] > 0) {
    real x = pVc[n].x;
    real y = pVc[n].y;

    if (fabs(x - one) < 1.0e-6 || fabs(y - one) < 1.0e-6 ||
        fabs(x + one) < 1.0e-6 || fabs(y + one) < 1.0e-6) {
      real r = sqrt(x*x + y*y);

      real dens = one + simulationTime/r;
      real pres = 1.0e-6;
      real momx = -dens*x/r;
      real momy = -dens*y/r;
      real ener = half*(Sq(momx) + Sq(momy))/dens + pres*iG1;

      pState[n].x = dens;
      pState[n].y = momx;
      pState[n].z = momy;
      pState[n].w = ener;
    }
  }
}

__host__ __device__
void SetBoundaryNohVertex(int n, real3 *pState,
                          const real2 *pVc, const int *pVbf,
                          real simulationTime, real iG1)
{
  // Dummy function: Noh boundaries are specific to Euler
}

__host__ __device__
void SetBoundaryNohVertex(int n, real *pState,
                          const real2 *pVc, const int *pVbf,
                          real simulationTime, real iG1)
{
  // Dummy function: Noh boundaries are specific to Euler
}

//######################################################################
/*! \brief Kernel setting Noh boundaries

At the outer boundaries the state is set to the analytic solution

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state vector
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param simulationTime Current simulation time
\param iG1 1/(Ratio of specific heats - 1)*/
//######################################################################

__global__ void
devSetNohBoundaries(int nVertex, realNeq *pState,
                    const real2 *pVc, const int *pVbf,
                    real simulationTime, real iG1)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    SetBoundaryNohVertex(n, pState, pVc, pVbf,
                         simulationTime, iG1);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! At the outer boundaries the state is set to the analytic solution*/
//######################################################################

void Simulation::SetNohBoundaries()
{
  int nVertex = mesh->GetNVertex();

  const real2 *pVc = mesh->VertexCoordinatesData();
  const int *pVbf = mesh->VertexBoundaryFlagData();

  realNeq *pState = vertexState->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetNohBoundaries,
                                       (size_t) 0, 0);

    // Execute kernel...
    devSetNohBoundaries<<<nBlocks, nThreads>>>
      (nVertex, pState, pVc, pVbf,
       simulationTime, 1.0/(G - 1.0));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int n = 0; n < nVertex; n++)
      SetBoundaryNohVertex(n, pState, pVc, pVbf,
                               simulationTime,
                               1.0/(G - 1.0));
  }
}

}  // namespace astrix
