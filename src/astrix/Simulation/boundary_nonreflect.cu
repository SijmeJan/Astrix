// -*-c++-*-
/*! \file boundary_nonreflect.cu
\brief Functions for setting non-reflecting boundary conditions

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./simulation.h"
#include "../Mesh/mesh.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

namespace astrix {

//######################################################################
/*! \brief Set non-reflecting boundaries single vertex on host or device.

  Non-reflecting boundaries are implemented by setting the currect state equal to the old state, so that the state at the boundary never changes.

\param n Vertex to consider
\param *pState Pointer to state vector
\param *pStateOld Pointer to old state vector
\param *pVbf Pointer to array of boundary flags*/
//######################################################################

__host__ __device__
void SetNonReflectingVertex(int n, realNeq *pState, realNeq *pStateOld,
                            const int *pVbf)
{
  if (pVbf[n] != 0) pState[n] = pStateOld[n];
}

//######################################################################
/*! \brief Kernel setting non-reflecting boundaries

  Non-reflecting boundaries are implemented by setting the currect state equal
to the old state, so that the state at the boundary never changes.

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state vector
\param *pStateOld Pointer to old state vector
\param *pVbf Pointer to array of boundary flags*/
//######################################################################

__global__ void
devSetNonReflectingBoundaries(int nVertex, realNeq *pState, realNeq *pStateOld,
                              const int *pVbf)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    SetNonReflectingVertex(n, pState, pStateOld, pVbf);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Non-reflecting boundaries are implemented by setting the currect state
equal to the old state, so that the state at the boundary never changes.*/
//######################################################################

void Simulation::SetNonReflectingBoundaries()
{
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();
  realNeq *pStateOld = vertexStateOld->GetPointer();

  const int *pVbf = mesh->VertexBoundaryFlagData();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devSetNonReflectingBoundaries,
       (size_t) 0, 0);

    // Execute kernel...
    devSetNonReflectingBoundaries<<<nBlocks, nThreads>>>
      (nVertex, pState, pStateOld, pVbf);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    // Set boundary state to previous state (= initial state)
    for (int n = 0; n < nVertex; n++)
      SetNonReflectingVertex(n, pState, pStateOld, pVbf);
  }
}

}  // namespace astrix
