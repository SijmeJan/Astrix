// -*-c++-*-
/*! \file potential.cu
\brief File containing function to calculate external gravitational potential at vertices.

\section LICENSE
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
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Calculate external gravitational potential and isothermal sound speed at vertex \a i

\param i Vertex to calculate gravitational potential at
\param problemDef Problem definition
\param *pVc Pointer to coordinates of vertices
\param *pVpot Pointer to gravitational potential (output)
\param *pVcs Pointer to isothermal sound speed (output)*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void CalcPotentialSingle(int i, ProblemDefinition problemDef,
                         const real2 *pVc, real *pVpot, real *pCs)
{
  real zero = (real) 0.0;
  real tenth = (real) 0.1;

  pVpot[i] = zero;
  pCs[i] = (real) 1.0;

  if (problemDef == PROBLEM_SOURCE) pVpot[i] = tenth*pVc[i].y;

  if (CL == CL_CYL_ISO) {
    // Constant H/r: c prop to r^{-1.5}
    //pCs[i] = (real) 0.05;//*exp(-(real) 1.5*pVc[i].x);
    pCs[i] = (real) 0.05*exp(-pVc[i].x);
  }
}

//######################################################################
/*! \brief Kernel calculating external gravitational potential at vertices

\param nVertex Total number of vertices in Mesh
\param problemDef Problem definition
\param *pVc Pointer to coordinates of vertices
\param *pVpot Pointer to gravitational potential (output)
\param *pVcs Pointer to isothermal sound speed (output)*/
//######################################################################

template<ConservationLaw CL>
__global__ void
devCalcPotential(int nVertex, ProblemDefinition problemDef,
                 const real2 *pVc, real *pVpot, real *pVcs)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    CalcPotentialSingle<CL>(n, problemDef, pVc, pVpot, pVcs);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Calculate external gravitational potential at vertices, based on vertex
coordinates and problem definition.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalcPotential()
{
  int nVertex = mesh->GetNVertex();
  real *vertPot = vertexPotential->GetPointer();
  real *vertCs = vertexSoundSpeed->GetPointer();

  const real2 *pVc = mesh->VertexCoordinatesData();
  ProblemDefinition p = simulationParameter->problemDef;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcPotential<CL>,
                                       (size_t) 0, 0);

    devCalcPotential<CL><<<nBlocks, nThreads>>>
      (nVertex, p, pVc, vertPot, vertCs);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      CalcPotentialSingle<CL>(i, p, pVc, vertPot, vertCs);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::CalcPotential();
template void Simulation<real, CL_BURGERS>::CalcPotential();
template void Simulation<real3, CL_CART_ISO>::CalcPotential();
template void Simulation<real3, CL_CYL_ISO>::CalcPotential();
template void Simulation<real4, CL_CART_EULER>::CalcPotential();

}  // namespace astrix
