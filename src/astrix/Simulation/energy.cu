// -*-c++-*-
/*! \file Simulation/energy.cu
\brief Functions swapping energy with pressure in state vector

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
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Replace energy with pressure at vertex \a i

\param i Index of vertex
\param *pState Pointer to state vector at vertices
\param G1 Ratio of specific heats - 1
\param *pVp Pointer to external potential at vertices*/
//#########################################################################

__host__ __device__
void ReplaceEnergyWithPressureSingle(int i, real4 *pState, real G1, real *pVp)
{
  real half = (real) 0.5;

  pState[i].w = G1*(pState[i].w -
                    half*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x -
                    pState[i].x*pVp[i]);
}

//! Version for three equations
__host__ __device__
void ReplaceEnergyWithPressureSingle(int i, real3 *pState, real G1, real *pVp)
{
  // Dummy function, nothing to be done if solving only three equations
}

//! Version for single equation
__host__ __device__
void ReplaceEnergyWithPressureSingle(int i, real *pState, real G1, real *pVp)
{
  // Dummy function, nothing to be done if solving only one equation
}

//#########################################################################
/*! Replace pressure with energy at vertex \a i

\param i Index of vertex
\param *pState Pointer to state vector at vertices
\param iG1 1/(Ratio of specific heats - 1)
\param *pVp Pointer to external potential at vertices*/
//#########################################################################

__host__ __device__
void ReplacePressureWithEnergySingle(int i, real4 *pState, real iG1, real *pVp)
{
  real half = (real) 0.5;

  pState[i].w = half*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x +
    pState[i].x*pVp[i] + pState[i].w*iG1;
}

//! Version for three equations
__host__ __device__
void ReplacePressureWithEnergySingle(int i, real3 *pState, real iG1, real *pVp)
{
  // Dummy function, nothing to be done if solving only three equations
}

//! Version for single equation
__host__ __device__
void ReplacePressureWithEnergySingle(int i, real *pState, real iG1, real *pVp)
{
  // Dummy function, nothing to be done if solving only one equation
}

//######################################################################
/*! Kernel replacing energy with pressure at all vertices

\param nVertex Total number of vertices
\param *pState Pointer to state vector at vertices
\param G1 Ratio of specific heats - 1
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devReplaceEnergyWithPressure(int nVertex, realNeq *pState, real G1, real *pVp)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    ReplaceEnergyWithPressureSingle(i, pState, G1, pVp);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Kernel replacing pressure with energy at all vertices

\param nVertex Total number of vertices
\param *pState Pointer to state vector at vertices
\param iG1 1/(Ratio of specific heats - 1)
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devReplacePressureWithEnergy(int nVertex, realNeq *pState, real iG1, real *pVp)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    ReplacePressureWithEnergySingle(i, pState, iG1, pVp);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! It can be useful, especially when interpolating, to use the pressure
rather than the total energy. This function replaces the total energy with
the pressure in the state vector for all vertices.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::ReplaceEnergyWithPressure()
{
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devReplaceEnergyWithPressure<realNeq, CL>,
       (size_t) 0, 0);

    devReplaceEnergyWithPressure<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pState, G - 1.0, pVp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      ReplaceEnergyWithPressureSingle(i, pState, G - 1.0, pVp);
  }
}

//#########################################################################
/*! It can be useful, especially when interpolating, to use the pressure rather
than the total energy. However, before doing any hydro, we have to swap them
back. This function replaces the pressure with the total energy in the state
vector for all vertices.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::ReplacePressureWithEnergy()
{
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devReplacePressureWithEnergy<realNeq, CL>,
       (size_t) 0, 0);

    devReplacePressureWithEnergy<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pState, 1.0/(G - 1.0), pVp);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      ReplacePressureWithEnergySingle(i, pState, 1.0/(G - 1.0), pVp);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::ReplacePressureWithEnergy();
template void Simulation<real, CL_BURGERS>::ReplacePressureWithEnergy();
template void Simulation<real3, CL_CART_ISO>::ReplacePressureWithEnergy();
template void Simulation<real3, CL_CYL_ISO>::ReplacePressureWithEnergy();
template void Simulation<real4, CL_CART_EULER>::ReplacePressureWithEnergy();

//##############################################################################

template void Simulation<real, CL_ADVECT>::ReplaceEnergyWithPressure();
template void Simulation<real, CL_BURGERS>::ReplaceEnergyWithPressure();
template void Simulation<real3, CL_CART_ISO>::ReplaceEnergyWithPressure();
template void Simulation<real3, CL_CYL_ISO>::ReplaceEnergyWithPressure();
template void Simulation<real4, CL_CART_EULER>::ReplaceEnergyWithPressure();

}  // namespace astrix
