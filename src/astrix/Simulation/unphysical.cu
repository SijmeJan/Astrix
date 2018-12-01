// -*-c++-*-
/*! \file unphysical.cu
\brief File containing functions detect unphysical states at vertices.

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

//######################################################################
/*! \brief Check unphysical state at vertex \a v

\param v Vertex to consider
\param *pState Pointer to state at vertices
\param *pVp Pointer to external potential at vertices
\param *pVertexUnphysicalFlag Pointer to array of flags indicating whether state is physical (0) or unphysical (1) (output)
\param G1 Ratio of specific heats - 1*/
//######################################################################

__host__ __device__
void FlagUnphysicalVertex(const int v, real4 *pState, real *pVp,
                          int *pVertexUnphysicalFlag, const real G1)
{
  const real zero = (real) 0.0;
  const real half = (real) 0.5;

  // Assume everything is fine
  int ret = 0;

  real dens = pState[v].x;
  real momx = pState[v].y;
  real momy = pState[v].z;
  real ener = pState[v].w;

  // Pressure
  real p = G1*(ener - half*(Sq(momx) + Sq(momy))/dens - dens*pVp[v]);

  // Flag if negative density or pressure
  if (dens < zero || p < zero || isnan(p)) ret = 1;

  // Output flag
  pVertexUnphysicalFlag[v] = ret;
}

__host__ __device__
void FlagUnphysicalVertex(const int v, real3 *pState, real *pVp,
                          int *pVertexUnphysicalFlag, const real G1)
{
  const real zero = (real) 0.0;

  // Assume everything is fine
  int ret = 0;

  real dens = pState[v].x;

  // Flag if negative density or pressure
  if (dens < zero) ret = 1;

  // Output flag
  pVertexUnphysicalFlag[v] = ret;
}

__host__ __device__
void FlagUnphysicalVertex(const int v, real *pState, real *pVp,
                          int *pVertexUnphysicalFlag, const real G1)
{
  // Output flag
  pVertexUnphysicalFlag[v] = 0;
}

//######################################################################
/*! \brief Kernel checking vertices for unphysical state

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state at vertices
\param *pVp Pointer to external potential at vertices
\param *pVertexUnphysicalFlag Pointer to array of flags indicating whether state is physical (0) or unphysical (1) (output)
\param G1 Ratio of specific heats - 1*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devFlagUnphysical(const int nVertex, realNeq *pState, real *pVp,
                  int *pVertexUnphysicalFlag, const real G1)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    FlagUnphysicalVertex(n, pState, pVp, pVertexUnphysicalFlag, G1);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Check all vertices for unphysical state.

  \param *vertexUnphysicalFlag Pointer to Array of flags indicating whether state is physical (0) or unphysical (1) (output)*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::FlagUnphysical(Array<int> *vertexUnphysicalFlag)
{
  // Total number of vertices in Mesh
  int nVertex = mesh->GetNVertex();

  // State vector at vertices
  realNeq *state = vertexState->GetPointer();
  real *pVp = vertexPotential->GetPointer();

  // Ratio of specific heats
  real G = simulationParameter->specificHeatRatio;

  // Pointer to output
  int *pVertexUnphysicalFlag = vertexUnphysicalFlag->GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFlagUnphysical<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devFlagUnphysical<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, state, pVp, pVertexUnphysicalFlag, G - 1.0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int v = 0; v < nVertex; v++)
      FlagUnphysicalVertex(v, state, pVp, pVertexUnphysicalFlag, G - 1.0);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::
FlagUnphysical(Array<int> *vertexUnphysicalFlag);
template void Simulation<real, CL_BURGERS>::
FlagUnphysical(Array<int> *vertexUnphysicalFlag);
template void Simulation<real3, CL_CART_ISO>::
FlagUnphysical(Array<int> *vertexUnphysicalFlag);
template void Simulation<real3, CL_CYL_ISO>::
FlagUnphysical(Array<int> *vertexUnphysicalFlag);
template void Simulation<real4, CL_CART_EULER>::
FlagUnphysical(Array<int> *vertexUnphysicalFlag);

}  // namespace astrix
