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

template<class realNeq, ConservationLaw CL>
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

template <class realNeq, ConservationLaw CL>
real Simulation<realNeq, CL>::DensityError()
{
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
                                       devDensityError<realNeq, CL>,
                                       (size_t) 0, 0);

    devDensityError<realNeq, CL><<<nBlocks, nThreads>>>
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

//##############################################################################
// Instantiate
//##############################################################################

template real Simulation<real, CL_ADVECT>::DensityError();
template real Simulation<real, CL_BURGERS>::DensityError();
template real Simulation<real3, CL_CART_ISO>::DensityError();
template real Simulation<real4, CL_CART_EULER>::DensityError();

}  // namespace astrix
