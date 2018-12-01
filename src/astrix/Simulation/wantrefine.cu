// -*-c++-*-
/*! \file wantrefine.cu
\brief File containing functions to determine which triangles need refining based on an estimate of the local truncation error.

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
#include "../Common/cudaLow.h"
#include "./simulation.h"
#include "./Param/simulationparameter.h"

namespace astrix {

// #########################################################################
/*! \brief Check if triangle i needs refining based on ErrorEstimate

\param i Index of triangle to consider
\param *pErrorEstimate Pointer to array with estimates of local truncation error
 (LTE)
\param maxError Limit of LTE above which to flag triangle for refinement
\param minError Limit of LTE below which to flag triangle for refinement
\param *pWantRefine Pointer to output array: 1 if triangle needs refining, -1 if it can be coarsened, 0 if nothing needs to happen*/
// #########################################################################

__host__ __device__
void FillWantRefineSingle(int i, real *pErrorEstimate,
                          real maxError, real minError,
                          int *pWantRefine)
{
  int ret = 0;

  if (pErrorEstimate[i] > maxError) ret = 1;
  if (pErrorEstimate[i] < minError) ret = -1;

  pWantRefine[i] = ret;
}

//######################################################################
/*! \brief Kernel checking if triangles need refining based on ErrorEstimate

\param nTriangle Total number of triangles in Mesh
\param *pErrorEstimate Pointer to array with estimates of local truncation error
 (LTE)
\param maxError Limit of LTE above which to flag triangle for refinement
\param minError Limit of LTE below which to flag triangle for refinement
\param *pWantRefine Pointer to output array: 1 if triangle needs refining, -1 if it can be coarsened, 0 if nothing needs to happen*/
//######################################################################

__global__ void
devFillWantRefine(int nTriangle, real *pErrorEstimate,
                  real maxError, real minError, int *pWantRefine)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FillWantRefineSingle(i, pErrorEstimate, maxError, minError, pWantRefine);

    i += blockDim.x*gridDim.x;
  }
}

// #########################################################################
/*! Flag triangles for refinement or coarsening based on an estimate of the local truncation error (LTE). First the LTE is computed; then we fill the Array triangleWantRefine with either 1 (triangle needs refining), -1 (triangle can be coarsened) or 0 (nothing needs to happen).*/
// #########################################################################

template<class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::FillWantRefine()
{
  int nTriangle = mesh->GetNTriangle();

  CalcErrorEstimate();
  triangleWantRefine->SetSize(nTriangle);

  real *pErrorEstimate = triangleErrorEstimate->GetPointer();
  int *pWantRefine = triangleWantRefine->GetPointer();

  real minError = simulationParameter->minError;
  real maxError = simulationParameter->maxError;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillWantRefine,
                                       (size_t) 0, 0);

    devFillWantRefine<<<nBlocks, nThreads>>>
      (nTriangle, pErrorEstimate, maxError, minError, pWantRefine);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      FillWantRefineSingle(i, pErrorEstimate, maxError, minError, pWantRefine);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::FillWantRefine();
template void Simulation<real, CL_BURGERS>::FillWantRefine();
template void Simulation<real3, CL_CART_ISO>::FillWantRefine();
template void Simulation<real3, CL_CYL_ISO>::FillWantRefine();
template void Simulation<real4, CL_CART_EULER>::FillWantRefine();

}  // namespace astrix
