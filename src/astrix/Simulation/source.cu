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
void CalcSourceSingle(int n, const int3 *pTv, const real2 *pVc,
		      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
		      const real3 *pTl, const real *pVp,
		      const realNeq *pState, realNeq *pSource)
{
  /*
  // Vertices belonging to triangle: 3 coalesced reads
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tnx2 = pTn2[n].x;
  real tnx3 = pTn3[n].x;
  real tny1 = pTn1[n].y;
  real tny2 = pTn2[n].y;
  real tny3 = pTn3[n].y;
  
  real dPotdx =
    tnx1*tl1*pVp[v1] + tnx2*tl2*pVp[v2] + tnx3*tl3*pVp[v3];
  real dPotdy =
    tny1*tl1*pVp[v1] + tny2*tl2*pVp[v2] + tny3*tl3*pVp[v3];
  
  pSource[n].x = 0.0;
  pSource[n].y = 0.5*rhoAve*dPotdx;
  pSource[n].z = 0.5*rhoAve*dPotdy;
  pSource[n].w = 0.5*momxAve*dPotdx + 0.5*momyAve*dPotdy;
  */
  pSource[n].x = 0.0;
  pSource[n].y = 0.0;
  pSource[n].z = 0.0;
  pSource[n].w = 0.0;
  
}
  
//######################################################################
//######################################################################

__global__ void 
devCalcSource(int nTriangle, const int3 *pTv, const real2 *pVc,
	      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
	      const real3 *pTl, const real *pVp,
	      const realNeq *pState, realNeq *pSource)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nTriangle){
    CalcSourceSingle(n, pTv, pVc, pTn1, pTn2, pTn3,
		     pTl, pVp, pState, pSource);

    n += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
//#########################################################################

void Simulation::CalcSource(Array<realNeq> *state)
{
  int nTriangle = mesh->GetNTriangle();
  const int3 *pTv = mesh->TriangleVerticesData();
  const real *pVp = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();
  const realNeq *pState = state->GetPointer();
  realNeq *pSource = triangleResidueSource->GetPointer();

  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);
  
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcSource, 
				       (size_t) 0, 0);

    devCalcSource<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVc, pTn1, pTn2, pTn3, pTl, pVp, pState, pSource);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcSourceSingle(i, pTv, pVc, pTn1, pTn2, pTn3, pTl,
		       pVp, pState, pSource);
  }  
}

}

