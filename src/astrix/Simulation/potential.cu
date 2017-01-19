// -*-c++-*-
/*! \file potential.cu
\brief File containing function to calculate external gravitational potential at vertices.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
/*! \brief Calculate external gravitational potential at vertex \a i

\param i Vertex to calculate gravitational potential at
\param problemDef Problem definition
\param *pVc Pointer to coordinates of vertices
\param *pVpot Pointer to gravitational potential (output)*/
//######################################################################

__host__ __device__
void CalcPotentialSingle(int i, ProblemDefinition problemDef,
                         const real2 *pVc, real *pVpot)
{
  real zero = (real) 0.0;
  // real tenth = (real) 0.1;

  pVpot[i] = zero;

  // if (problemDef == PROBLEM_RT) pVpot[i] = tenth*pVc[i].y;
}

//######################################################################
/*! \brief Kernel calculating external gravitational potential at vertices

\param nVertex Total number of vertices in Mesh
\param problemDef Problem definition
\param *pVc Pointer to coordinates of vertices
\param *pVpot Pointer to gravitational potential (output)*/
//######################################################################

__global__ void
devCalcPotential(int nVertex, ProblemDefinition problemDef,
                 const real2 *pVc, real *vertPot)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    CalcPotentialSingle(n, problemDef, pVc, vertPot);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Calculate external gravitational potential at vertices, based on vertex
coordinates and problem definition.*/
//#########################################################################

void Simulation::CalcPotential()
{
  int nVertex = mesh->GetNVertex();
  real *vertPot = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcPotential,
                                       (size_t) 0, 0);

    devCalcPotential<<<nBlocks, nThreads>>>
      (nVertex, problemDef, pVc, vertPot);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      CalcPotentialSingle(i, problemDef, pVc, vertPot);
  }
}

}  // namespace astrix
