// -*-c++-*-
/*! \file diagnostics.cu
\brief Calculating various diagnostics for Simulations

*/ /* \section LICENSE
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
#include "../Common/state.h"

namespace astrix {

//##############################################################################
/*! \brief L1 density error at vertex \a i

\param i Vertex to consider
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pState Pointer to current state at vertices
\param *pStateExact Pointer to exact solution at vertices
\param *E Pointer to L1 error (output)*/
//##############################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void DensityErrorSingle(unsigned int i, const real *pVarea,
                        realNeq *pState, realNeq *pStateExact, real *E)
{
  real d = state::GetDensity<realNeq, CL>(pState[i]);
  real d0 = state::GetDensity<realNeq, CL>(pStateExact[i]);

  E[i] = pVarea[i]*std::abs(d - d0);
}

//######################################################################
/*! \brief Kernel calculating L1 density errors

\param nVertex Total number of vertices in Mesh
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pState Pointer to current state at vertices
\param *pStateExact Pointer to exact solution at vertices
\param *E Pointer to L1 error (output)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devDensityError(unsigned int nVertex, const real *pVarea,
                realNeq *pState, realNeq *pStateExact, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    DensityErrorSingle<realNeq, CL>(n, pVarea, pState, pStateExact, E);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Returns L1 density error. */
//######################################################################

template <class realNeq, ConservationLaw CL>
real Simulation<realNeq, CL>::DensityError()
{
  vertexStateOld->SetEqual(vertexState);

  // Assume this gets the correct solution into vertexState
  SetInitial(simulationTime, 0);

  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pStateExact = vertexState->GetPointer();
  realNeq *pState = vertexStateOld->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devDensityError<realNeq, CL>,
                                       (size_t) 0, 0);

    devDensityError<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pVarea, pState, pStateExact, pE);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      DensityErrorSingle<realNeq, CL>(n, pVarea, pState, pStateExact, pE);
  }

  real e = E->Sum()/mesh->GetTotalArea();

  delete E;

  // Put current state back where it belongs
  vertexState->SetEqual(vertexStateOld);

  return e;
}

//##############################################################################
// Instantiate
//##############################################################################

template real Simulation<real, CL_ADVECT>::DensityError();
template real Simulation<real, CL_BURGERS>::DensityError();
template real Simulation<real3, CL_CART_ISO>::DensityError();
template real Simulation<real3, CL_CYL_ISO>::DensityError();
template real Simulation<real4, CL_CART_EULER>::DensityError();

}  // namespace astrix
