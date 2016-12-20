// -*-c++-*-
/*! \file source.cu
\brief File containing function to calculate source term contribution to residual.*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {
  
//######################################################################
//######################################################################

__host__ __device__
void CalcSourceSingle(int i, const real2 *pVc, const real2 *pTn1,
		      const real2 *pTn2, const real2 *pTn3,
		      const real3 *pTl, const real *pVp,
		      const realNeq *pState, realNeq *pSource)
{
  real dPotdx =
    tnx1*tl1*pVp[v1] + tnx2*tl2*pVp[v2] + tnx3*tl3*pVp[v3];
  real dPotdy =
    tny1*tl1*pVp[v1] + tny2*tl2*pVp[v2] + tny3*tl3*pVp[v3];
  
  pSource[i].x = zero;
  pSource[i].y = half*rhoAve*dPotdx;
  pSource[i].z = half*rhoAve*dPotdy;
  pSource[i].w = half*momxAve*dPotdx + half*momyAve*dPotdy;
}
  
//######################################################################
//######################################################################

__global__ void 
devCalcSource(int nTriangle, const real2 *pVc, const real2 *pTn1,
	      const real2 *pTn2, const real2 *pTn3,
	      const real3 *pTl, const real *pVp,
	      const realNeq *pState, realNeq *pSource)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nTriangle){
    CalcSourceSingle(n, pVc, pTn1, pTn2, pTn3, pTl, pVp, pState, pSource);

    n += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
//#########################################################################

void Simulation::CalcSource(Array<realNeq> *state)
{
  int nTriangle = mesh->GetNTriangle();
  const real *pVp = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();
  const realNeq *pState = state->GetPointer();
  realNeq *pSource = triangleSource->GetPointer();
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcSource, 
				       (size_t) 0, 0);

    devCalcSource<<<nBlocks, nThreads>>>
      (nTriangle, pVc, pVp, pState, pSource);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcSourceSingle(i, pVc, pVp, pState, pSource);
  }  
}

}
