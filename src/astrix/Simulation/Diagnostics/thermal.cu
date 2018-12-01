// -*-c++-*-
/*! \file thermal.cu
\brief Functions to calculate total thermal energy

*/ /* \section LICENSE
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
void ThermalEnergySingle(unsigned int i, const real *pVarea,
                         T *pState, real *pVp, real *E)
{
  E[i] = (real) 0.0;
}

template<>
__host__ __device__
void ThermalEnergySingle<real4, CL_CART_EULER>(unsigned int i,
                                               const real *pVarea,
                                               real4 *pState,
                                               real *pVp,
                                               real *E)
{
  real half = (real) 0.5;

  real d = pState[i].x;
  real m = pState[i].y;
  real n = pState[i].z;
  real e = pState[i].w;

  E[i] = pVarea[i]*(e - pVp[i]*d - half*(m*m + n*n)/d);
}

//######################################################################
//######################################################################

template<class T, ConservationLaw CL>
__global__ void
devThermalEnergy(unsigned int nVertex, const real *pVarea,
                 T *pState, real *pVp, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    ThermalEnergySingle<T, CL>(n, pVarea, pState, pVp, E);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
//######################################################################

template <class T, ConservationLaw CL>
real Diagnostics<T, CL>::ThermalEnergy(Array<T> *state,
                                       Array<real> *pot,
                                       Mesh *mesh)
{
  int cudaFlag = state->GetCudaFlag();
  unsigned int nVertex = mesh->GetNVertex();

  T *pState = state->GetPointer();
  real *pVp = pot->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);
  real *pE = E->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devThermalEnergy<T, CL>,
                                       (size_t) 0, 0);

    devThermalEnergy<T, CL><<<nBlocks, nThreads>>>
      (nVertex, pVarea, pState, pVp, pE);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      ThermalEnergySingle<T, CL>(n, pVarea, pState, pVp, pE);
  }

  real eTotal = E->Sum();

  delete E;

  return eTotal;
}

//###################################################
// Instantiate
//###################################################

template
real Diagnostics<real, CL_ADVECT>::ThermalEnergy(Array<real> *state,
                                                 Array<real> *pot,
                                                 Mesh *mesh);
template
real Diagnostics<real, CL_BURGERS>::ThermalEnergy(Array<real> *state,
                                                  Array<real> *pot,
                                                  Mesh *mesh);
template
real Diagnostics<real3, CL_CART_ISO>::ThermalEnergy(Array<real3> *state,
                                                    Array<real> *pot,
                                                    Mesh *mesh);
template
real Diagnostics<real3, CL_CYL_ISO>::ThermalEnergy(Array<real3> *state,
                                                   Array<real> *pot,
                                                   Mesh *mesh);
template
real Diagnostics<real4, CL_CART_EULER>::ThermalEnergy(Array<real4> *state,
                                                      Array<real> *pot,
                                                      Mesh *mesh);

}  // namespace astrix
