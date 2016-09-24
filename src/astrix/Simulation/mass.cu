// -*-c++-*-
/*! \file mass.cu
\brief Functions to calculate total mass in Simulation*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
/*! \brief Compute mass associated with vertex \a n

\param n Vertex to consider
\param *pState Pointer to state vector
\param *pVarea Pointer to vertex area
\param *pVm Pointer to vertex mass (output)*/
//######################################################################

__host__ __device__
void FillMassArraySingle(unsigned int n, real4 *pState, 
			 const real *pVarea, real *pVm)
{
  pVm[n] = pVarea[n]*pState[n].x;
}

__host__ __device__
void FillMassArraySingle(unsigned int n, real *pState, 
			 const real *pVarea, real *pVm)
{
  pVm[n] = pVarea[n]*pState[n];
}

//######################################################################
/*! \brief Compute mass associated with vertices

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state vector
\param *pVarea Pointer to vertex area
\param *pVm Pointer to vertex mass (output)*/
//######################################################################

__global__ void
devFillMassArray(unsigned int nVertex, realNeq *pState,
		 const real *pVarea, real *pVm)
{
  // n=vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    FillMassArraySingle(n, pState, pVarea, pVm);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Compute total mass in simulation*/
//######################################################################

real Simulation::TotalMass()
{
  unsigned int nVertex = mesh->GetNVertex();
  realNeq *pState = vertexState->GetPointer();

  // Mass in every cell 
  Array<real> *vertexMass = new Array<real>(1, cudaFlag, nVertex);
  real *pVm = vertexMass->GetPointer();
  
  const real *pVarea = mesh->VertexAreaData();
  
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillMassArray, 
				       (size_t) 0, 0);

    // Execute kernel...
    devFillMassArray<<<nBlocks, nThreads>>>
      (nVertex, pState, pVarea, pVm);
      
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {    
    for (unsigned int n = 0; n < nVertex; n++) 
      FillMassArraySingle(n, pState, pVarea, pVm);
  }

  real mass = vertexMass->Sum();

  delete vertexMass;
  
  return mass;
}

}
