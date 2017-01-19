// -*-c++-*-
/*! \file energy.cu
\brief Functions swapping energy with pressure in state vector*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

namespace astrix {

//#########################################################################
/*! Replace energy with pressure at vertex \a i

\param i Index of vertex
\param *pState Pointer to state vector at vertices
\param G1 Ratio of specific heats - 1*/
//#########################################################################

__host__ __device__
void ReplaceEnergyWithPressureSingle(int i, real4 *pState, real G1)
{
  real half = (real) 0.5;

  pState[i].w = G1*(pState[i].w - half*(Sq(pState[i].y) + Sq(pState[i].z))/
                   pState[i].x);
}

__host__ __device__
void ReplaceEnergyWithPressureSingle(int i, real *pState, real G1)
{
  // Dummy function, nothing to be done if solving only one equation
}

//#########################################################################
/*! Replace pressure with energy at vertex \a i

\param i Index of vertex
\param *pState Pointer to state vector at vertices
\param iG1 1/(Ratio of specific heats - 1)*/
//#########################################################################

__host__ __device__
void ReplacePressureWithEnergySingle(int i, real4 *pState, real iG1)
{
  real half = (real) 0.5;

  pState[i].w =
    half*(Sq(pState[i].y) + Sq(pState[i].z))/pState[i].x + pState[i].w*iG1;
}

__host__ __device__
void ReplacePressureWithEnergySingle(int i, real *pState, real iG1)
{
  // Dummy function, nothing to be done if solving only one equation
}

//######################################################################
/*! Kernel replacing energy with pressure at all vertices

\param nVertex Total number of vertices
\param *pState Pointer to state vector at vertices
\param G1 Ratio of specific heats - 1*/
//######################################################################

__global__ void
devReplaceEnergyWithPressure(int nVertex, realNeq *pState, real G1)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    ReplaceEnergyWithPressureSingle(i, pState, G1);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Kernel replacing pressure with energy at all vertices

\param nVertex Total number of vertices
\param *pState Pointer to state vector at vertices
\param iG1 1/(Ratio of specific heats - 1)*/
//######################################################################

__global__ void
devReplacePressureWithEnergy(int nVertex, realNeq *pState, real iG1)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    ReplacePressureWithEnergySingle(i, pState, iG1);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! It can be useful, especially when interpolating, to use the pressure
rather than the total energy. This function replaces the total energy with
the pressure in the state vector for all vertices.*/
//#########################################################################

void Simulation::ReplaceEnergyWithPressure()
{
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devReplaceEnergyWithPressure,
       (size_t) 0, 0);

    devReplaceEnergyWithPressure<<<nBlocks, nThreads>>>
      (nVertex, pState, specificHeatRatio - 1.0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      ReplaceEnergyWithPressureSingle(i, pState, specificHeatRatio - 1.0);
  }
}

//#########################################################################
/*! It can be useful, especially when interpolating, to use the pressure rather
than the total energy. However, before doing any hydro, we have to swap them
back. This function replaces the pressure with the total energy in the state
vector for all vertices.*/
//#########################################################################

void Simulation::ReplacePressureWithEnergy()
{
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devReplacePressureWithEnergy,
       (size_t) 0, 0);

    devReplacePressureWithEnergy<<<nBlocks, nThreads>>>
      (nVertex, pState, 1.0/(specificHeatRatio - 1.0));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      ReplacePressureWithEnergySingle(i, pState, 1.0/(specificHeatRatio - 1.0));
  }
}

}  // namespace astrix
