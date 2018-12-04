// -*-c++-*-
/*! \file param.cu
\brief File containing function to calculate Roe's parameter vector at vertices.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/profile.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Calculate Roe's parameter vector at vertex \a n.

This function calculates Roe's parameter vector Z at vertex \a n.

 \param n index of vertex.
 \param *pState Pointer to state vector at vertices
 \param *pVz Pointer to parameter vector at vertices (output)
 \param G1 Ratio of specific heats - 1
\param *pVp Pointer to external potential at vertices*/
//######################################################################

__host__ __device__
void CalcParamVecSingle(int n, real4 *pState, real4 *pVz, real G1, real *pVp)
{
  real half = (real) 0.5;

  real dens = pState[n].x;
  real momx = pState[n].y;
  real momy = pState[n].z;
  real ener = pState[n].w;

  real d = sqrt(dens);
  real u = momx/dens;
  real v = momy/dens;

  // Pressure
  real p = G1*(ener - half*dens*(u*u + v*v) - dens*pVp[n]);

  // Roe parameter vector
  pVz[n].x = d;
  pVz[n].y = d*u;
  pVz[n].z = d*v;
  pVz[n].w = d*(ener + p)/dens;
}

//! Version for three equations
__host__ __device__
void CalcParamVecSingle(int n, real3 *pState, real3 *pVz, real G1, real *pVp)
{
  real dens = pState[n].x;
  real momx = pState[n].y;
  real momy = pState[n].z;

  real d = sqrt(dens);
  real u = momx/dens;
  real v = momy/dens;

  // Roe parameter vector
  pVz[n].x = d;
  pVz[n].y = d*u;
  pVz[n].z = d*v;
}

//! Version for single equation
__host__ __device__
void CalcParamVecSingle(int n, real *pState, real *pVz, real G1, real *pVp)
{
  pVz[n] = pState[n];
}

//######################################################################
/*! \brief Kernel to calculate Roe's parameter vector at all vertices.

This kernel function calculates Roe's parameter vector Z for all vertices in the mesh.

 \param nVertex total number of vertices.
 \param *pState Pointer to state vector at vertices
 \param *pVz Pointer to parameter vector at vertices (output)
 \param G1 Ratio of specific heats - 1
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcParamVec(int nVertex, realNeq *pState, realNeq *pVz, real G1, real *pVp)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    CalcParamVecSingle(n, pState, pVz, G1, pVp);

    n += blockDim.x*gridDim.x;
  }
}

//##############################################################################
/*! This function calculates Roe's parameter vector Z for all vertices in the mesh, based on either \a vertexState or \a vertexStateOld.

  \param useOldFlag flag indicating whether to use \a vertexStateOld (1) or \a vertexState (any other value).*/
//##############################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalculateParameterVector(int useOldFlag)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  // Use old state
  if (useOldFlag == 1)
    pState = vertexStateOld->GetPointer();

  realNeq *pVz = vertexParameterVector->GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcParamVec<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcParamVec<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pState, pVz, G - 1.0, pVp);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int n = 0; n < nVertex; n++)
      CalcParamVecSingle(n, pState, pVz, G - 1.0, pVp);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("Param.prof2", nVertex, elapsedTime, cudaFlag);
#endif
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Simulation<real, CL_ADVECT>::CalculateParameterVector(int useOldFlag);
template void
Simulation<real, CL_BURGERS>::CalculateParameterVector(int useOldFlag);
template void
Simulation<real3, CL_CART_ISO>::CalculateParameterVector(int useOldFlag);
template void
Simulation<real3, CL_CYL_ISO>::CalculateParameterVector(int useOldFlag);
template void
Simulation<real4, CL_CART_EULER>::CalculateParameterVector(int useOldFlag);

}  // namespace astrix
