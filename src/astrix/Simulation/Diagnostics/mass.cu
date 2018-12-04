// -*-c++-*-
/*! \file mass.cu
\brief Functions to calculate total mass

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

//######################################################################
/*! \brief Compute mass associated with vertex \a n

\param n Vertex to consider
\param *pState Pointer to state vector
\param *pVarea Pointer to vertex area
\param *pVm Pointer to vertex mass (output)*/
//######################################################################


template<class T, ConservationLaw CL>
__host__ __device__
void FillMassArraySingle(unsigned int n, T *pState,
                         const real *pVarea, real *pVm)
{
  // General case, make zero
  pVm[n] = 0.0;
}

//! Version for linear advection
template<>
__host__ __device__
void FillMassArraySingle<real, CL_ADVECT>(unsigned int n, real *pState,
                                          const real *pVarea, real *pVm)
{
  // Specialization for linear advection
  pVm[n] = pVarea[n]*pState[n];
}

//! Version for Burgers equation
template<>
__host__ __device__
void FillMassArraySingle<real, CL_BURGERS>(unsigned int n, real *pState,
                                           const real *pVarea, real *pVm)
{
  // Specialization for Burgers equation
  pVm[n] = pVarea[n]*pState[n];
}

//! Version for isothermal hydrodynamics
template<>
__host__ __device__
void FillMassArraySingle<real3, CL_CART_ISO>(unsigned int n, real3 *pState,
                                             const real *pVarea, real *pVm)
{
  // Specialization for isothermal Cartesian hydro
  pVm[n] = pVarea[n]*pState[n].x;
}

//! Version for Euler equations
template<>
__host__ __device__
void FillMassArraySingle<real4, CL_CART_EULER>(unsigned int n, real4 *pState,
                                               const real *pVarea, real *pVm)
{
  // Specialization for Cartesian hydro
  pVm[n] = pVarea[n]*pState[n].x;
}

//######################################################################
/*! \brief Compute mass associated with vertices

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state vector
\param *pVarea Pointer to vertex area
\param *pVm Pointer to vertex mass (output)*/
//######################################################################

template<class T, ConservationLaw CL>
__global__ void
devFillMassArray(unsigned int nVertex, T *pState,
                 const real *pVarea, real *pVm)
{
  // n=vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    FillMassArraySingle<T, CL>(n, pState, pVarea, pVm);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Compute total mass in simulation.

\param *state Pointer to state vector
\param *mesh Pointer to Mesh object*/
//######################################################################

template <class T, ConservationLaw CL>
real Diagnostics<T, CL>::TotalMass(Array<T> *state, Mesh *mesh)
{
  unsigned int nVertex = mesh->GetNVertex();
  int cudaFlag = state->GetCudaFlag();

  // Mass in every cell
  Array<real> *vertexMass = new Array<real>(1, cudaFlag, nVertex);
  real *pVm = vertexMass->GetPointer();
  T* pState = state->GetPointer();

  const real *pVarea = mesh->VertexAreaData();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillMassArray<T, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devFillMassArray<T, CL><<<nBlocks, nThreads>>>
      (nVertex, pState, pVarea, pVm);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      FillMassArraySingle<T, CL>(n, pState, pVarea, pVm);
  }

  real mass = vertexMass->Sum();

  delete vertexMass;

  return mass;
}

//###################################################
// Instantiate
//###################################################

template
real Diagnostics<real, CL_ADVECT>::TotalMass(Array<real> *state,
                                             Mesh *mesh);
template
real Diagnostics<real, CL_BURGERS>::TotalMass(Array<real> *state,
                                              Mesh *mesh);
template
real Diagnostics<real3, CL_CART_ISO>::TotalMass(Array<real3> *state,
                                                Mesh *mesh);
template
real Diagnostics<real3, CL_CYL_ISO>::TotalMass(Array<real3> *state,
                                               Mesh *mesh);
template
real Diagnostics<real4, CL_CART_EULER>::TotalMass(Array<real4> *state,
                                                  Mesh *mesh);

}  // namespace astrix
