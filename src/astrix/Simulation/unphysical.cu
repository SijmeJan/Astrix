// -*-c++-*-
/*! \file unphysical.cu
\brief File containing functions detect unphysical states at vertices.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

namespace astrix {

//######################################################################
/*! \brief Check unphysical state at vertex \a v

\param v Vertex to consider
\param *pState Pointer to state at vertices
\param *pVertexUnphysicalFlag Pointer to array of flags indicating whether state is physical (0) or unphysical (1) (output)
\param G1 Ratio of specific heats - 1*/
//######################################################################

__host__ __device__
void FlagUnphysicalVertex(const int v, real4 *pState,
			  int *pVertexUnphysicalFlag, const real G1)
{
  const real zero = (real) 0.0;
  const real half = (real) 0.5;

  // Assume everything is fine
  int ret = 0;

  real dens = pState[v].x;
  real momx = pState[v].y;
  real momy = pState[v].z;
  real ener = pState[v].w;

  // Pressure
  real p = G1*(ener - half*(Sq(momx) + Sq(momy))/dens);
  
  // Flag if negative density or pressure
  if(dens < zero || p < zero || isnan(p)) ret = 1;
  
  // Output flag
  pVertexUnphysicalFlag[v] = ret;
}

__host__ __device__
void FlagUnphysicalVertex(const int v, real *pState,
			  int *pVertexUnphysicalFlag, const real G1)
{
  // Output flag
  pVertexUnphysicalFlag[v] = 0;
}

//######################################################################
/*! \brief Kernel checking vertices for unphysical state

\param nVertex Total number of vertices in Mesh
\param *pState Pointer to state at vertices
\param *pVertexUnphysicalFlag Pointer to array of flags indicating whether state is physical (0) or unphysical (1) (output)
\param G1 Ratio of specific heats - 1*/ 
//######################################################################

__global__ void 
devFlagUnphysical(const int nVertex, realNeq *pState,
		  int *pVertexUnphysicalFlag, const real G1)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while (n < nVertex) {
    FlagUnphysicalVertex(n, pState, pVertexUnphysicalFlag, G1);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Check all vertices for unphysical state. 

  \param *pVertexUnphysicalFlag Pointer to array of flags indicating whether state is physical (0) or unphysical (1) (output)*/
//######################################################################

void Simulation::FlagUnphysical(Array<int> *vertexUnphysicalFlag)
{
  // Total number of vertices in Mesh
  int nVertex = mesh->GetNVertex();

  // State vector at vertices
  realNeq *state = vertexState->GetPointer();

  // Pointer to output
  int *pVertexUnphysicalFlag = vertexUnphysicalFlag->GetPointer();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlagUnphysical, 
				       (size_t) 0, 0);
    
    // Execute kernel... 
    devFlagUnphysical<<<nBlocks,nThreads>>>
      (nVertex, state, pVertexUnphysicalFlag, specificHeatRatio - 1.0);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int v = 0; v < nVertex; v++)
      FlagUnphysicalVertex(v, state, pVertexUnphysicalFlag,
			   specificHeatRatio - 1.0);
  }
}

}
