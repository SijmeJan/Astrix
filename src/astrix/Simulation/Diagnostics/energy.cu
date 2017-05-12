// -*-c++-*-
/*! \file energy.cu
\brief Functions to calculate total energy

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../../Common/cudaLow.h"
#include "../../Mesh/mesh.h"
#include "./diagnostics.h"

namespace astrix {

//##############################################################################
//##############################################################################

template<class T, ConservationLaw CL>
__host__ __device__
void TotalEnergySingle(unsigned int i, const real *pVarea,
                       T *pState, real *E)
{
  E[i] = (real) 0.0;
}

template<>
__host__ __device__
void TotalEnergySingle<real4, CL_CART_EULER>(unsigned int i,
                                             const real *pVarea,
                                             real4 *pState, real *E)
{
  E[i] = pVarea[i]*pState[i].w;
}

//######################################################################
//######################################################################

template<class T, ConservationLaw CL>
__global__ void
devTotalEnergy(unsigned int nVertex, const real *pVarea,
               T *pState, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    TotalEnergySingle<T, CL>(n, pVarea, pState, E);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

template <class T, ConservationLaw CL>
real Diagnostics<T, CL>::TotalEnergy(Array<T> *state, Mesh *mesh)
{
  int cudaFlag = state->GetCudaFlag();
  unsigned int nVertex = mesh->GetNVertex();

  T *pState = state->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devTotalEnergy<T, CL>,
                                       (size_t) 0, 0);

    devTotalEnergy<T, CL><<<nBlocks, nThreads>>>
      (nVertex, pVarea, pState, pE);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      TotalEnergySingle<T, CL>(n, pVarea, pState, pE);
  }

  real eTotal = E->Sum();

  delete E;

  return eTotal;
}

//###################################################
// Instantiate
//###################################################

template
real Diagnostics<real, CL_ADVECT>::TotalEnergy(Array<real> *state,
                                               Mesh *mesh);
template
real Diagnostics<real, CL_BURGERS>::TotalEnergy(Array<real> *state,
                                                Mesh *mesh);
template
real Diagnostics<real3, CL_CART_ISO>::TotalEnergy(Array<real3> *state,
                                                  Mesh *mesh);
template
real Diagnostics<real4, CL_CART_EULER>::TotalEnergy(Array<real4> *state,
                                                    Mesh *mesh);

}  // namespace astrix
