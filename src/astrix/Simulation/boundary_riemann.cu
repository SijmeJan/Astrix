// -*-c++-*-
/*! \file boundary_riemann.cu
\brief Functions for setting boundary conditions for 2D Riemann problem

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
/*! \brief Set Riemann boundaries single vertex \a n

At the boundaries of the 2D Riemann problem the solution is simply a moving shock, with a speed we can compute. This makes it easy to set appropriate boundary conditions. Note that this is specific for the 2D Riemann problem implemented!

\param n Vertex to consider
\param *pState Pointer to state vector
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param simulationTime Current simulation time
\param iG1 1/(Ratio of specific heats - 1)*/
//######################################################################

__host__ __device__
void SetBoundaryRiemannVertex(int n, real4 *pState,
                              const real2 *pVc, const int *pVbf,
                              real simulationTime, real iG1)
{
  if (pVbf[n] > 0) {
    real densRB = 1.5f;
    real momxRB = 1.0e-10f;
    real momyRB = 1.0e-10f;
    real enerRB = 0.5f*(Sq(momxRB) + Sq(momyRB))/densRB + 1.5f*iG1;

    real densLB = 0.5322581;
    real momxLB = 1.2060454f*densLB;
    real momyLB = 0.0f;
    real enerLB = 0.5f*(Sq(momxLB) + Sq(momyLB))/densLB + 0.3f*iG1;

    real densRO = 0.5322581f;
    real momxRO = 0.0f;
    real momyRO = 1.2060454f*densRO;
    real enerRO = 0.5f*(Sq(momxRO) + Sq(momyRO))/densRO + 0.3f*iG1;

    real densLO = 0.1379928f;
    real momxLO = 1.2060454f*densLO;
    real momyLO = 1.2060454f*densLO;
    real enerLO = 0.5f*(Sq(momxLO) + Sq(momyLO))/densLO + 0.0290323f*iG1;

    if ((pVbf[n] % 4) != 0) {
      if (pVc[n].x < 0.5f) {
        // Left x boundary
        real shock_y = -0.422f*simulationTime + 0.8f;
        real g = 1.0f - (pVc[n].y > shock_y);
        pState[n].x = densLB*(1.0f - g) + densLO*g;
        pState[n].y = momxLB*(1.0f - g) + momxLO*g;
        pState[n].z = momyLB*(1.0f - g) + momyLO*g;
        pState[n].w = enerLB*(1.0f - g) + enerLO*g;
      } else {
        // Right x boundary
        // real shock_y = -0.663*simulationTime + 0.8;
        real shock_y = -0.6375f*simulationTime + 0.8f;
        // real g = 0.5 - atan((vy[n]-shock_y)/d)/M_PI;
        real g = 1.0f - (pVc[n].y > shock_y);
        pState[n].x = densRB*(1.0f - g) + densRO*g;
        pState[n].y = momxRB*(1.0f - g) + momxRO*g;
        pState[n].z = momyRB*(1.0f - g) + momyRO*g;
        pState[n].w = enerRB*(1.0f - g) + enerRO*g;
      }
    }

    if (pVbf[n] > 3) {
      if (pVc[n].y < 0.5f) {
        // Left y boundary
        real shock_x = -0.422f*simulationTime + 0.8f;
        // real f = 0.5 - atan((vx[n]-shock_x)/d)/M_PI;
        real f = 1.0f - (pVc[n].x > shock_x);
        pState[n].x = densRO*(1.0f - f) + densLO*f;
        pState[n].y = momxRO*(1.0f - f) + momxLO*f;
        pState[n].z = momyRO*(1.0f - f) + momyLO*f;
        pState[n].w = enerRO*(1.0f - f) + enerLO*f;
      } else {
        // Right y boundary
        // real shock_x = -0.663*simulationTime + 0.8;
        real shock_x = -0.6375f*simulationTime + 0.8f;
        // real f = 0.5 - atan((vx[n]-shock_x)/d)/M_PI;
        real f = 1.0f - (pVc[n].x > shock_x);
        pState[n].x = densRB*(1.0f - f) + densLB*f;
        pState[n].y = momxRB*(1.0f - f) + momxLB*f;
        pState[n].z = momyRB*(1.0f - f) + momyLB*f;
        pState[n].w = enerRB*(1.0f - f) + enerLB*f;
      }
    }
  }
}

__host__ __device__
void SetBoundaryRiemannVertex(int n, real3 *pState,
                              const real2 *pVc, const int *pVbf,
                              real simulationTime, real iG1)
{
  // Dummy: no Riemann setup for 3 equations
}

__host__ __device__
void SetBoundaryRiemannVertex(int n, real *pState,
                              const real2 *pVc, const int *pVbf,
                              real simulationTime, real iG1)
{
  // Non-reflecting in the case of Burgers Riemann
  if (pVbf[n] != 0) pState[n] = 1.0e-10;
}

//######################################################################
/*! \brief Kernel setting Riemann boundaries

At the boundaries of the 2D Riemann problem the solution is simply a moving
shock, with a speed we can compute. This makes it easy to set appropriate
boundary conditions. Note that this is specific for the 2D Riemann problem
implemented!

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state vector
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param simulationTime Current simulation time
\param iG1 1/(Ratio of specific heats - 1)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devSetRiemannBoundaries(int nVertex, realNeq *pState,
                        const real2 *pVc, const int *pVbf,
                        real simulationTime, real iG1)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    SetBoundaryRiemannVertex(n, pState, pVc, pVbf,
                             simulationTime, iG1);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! At the boundaries of the 2D Riemann problem the solution is simply a moving
shock, with a speed we can compute. This makes it easy to set appropriate
boundary conditions. Note that this is specific for the 2D Riemann problem
implemented!*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::SetRiemannBoundaries()
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
                                       devSetRiemannBoundaries<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devSetRiemannBoundaries<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pState, pVc, pVbf,
       simulationTime, 1.0/(G - 1.0));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int n = 0; n < nVertex; n++)
      SetBoundaryRiemannVertex(n, pState, pVc, pVbf,
                               simulationTime,
                               1.0/(G - 1.0));
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::SetRiemannBoundaries();
template void Simulation<real, CL_BURGERS>::SetRiemannBoundaries();
template void Simulation<real3, CL_CART_ISO>::SetRiemannBoundaries();
template void Simulation<real3, CL_CYL_ISO>::SetRiemannBoundaries();
template void Simulation<real4, CL_CART_EULER>::SetRiemannBoundaries();

}  // namespace astrix
