// -*-c++-*-
/*! \file refine_wantrefine.cu
\brief File containing functions to determine which triangles need refining based on an estimate of the local truncation error.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Param/meshparameter.h"

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
/*! Flag triangles for refinement or coarsening based on an estimate of the local truncation error (LTE). First the LTE is computed; then we fill the Array triangleWantRefine with either 1 (triangle needs refining), -1 (triangle can be coarsened) or 0 (nothing needs to happen).

\param *vertexState Pointer to Array containing state vector (density etc). Needed to compute LTE
\param specificHeatRatio Ratio of specific heats*/ 
// #########################################################################

void Mesh::FillWantRefine(Array<realNeq> *vertexState, real specificHeatRatio)
{
  CalcErrorEstimate(vertexState, specificHeatRatio);
  real *pErrorEstimate = triangleErrorEstimate->GetPointer();
  int *pWantRefine = triangleWantRefine->GetPointer();

  real minError = meshParameter->minError;
  real maxError = meshParameter->maxError;
  
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

}
