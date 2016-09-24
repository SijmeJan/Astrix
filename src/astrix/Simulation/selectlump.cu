// -*-c++-*-
/*! \file selectlump.cu*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "../Common/helper_math.h"

namespace astrix {

//######################################################################
//######################################################################

__host__ __device__
void SelectLumpLDASingle(int n, real dt, int massMatrix, int selectLumpFlag,
			 const int3* __restrict__ pTv, realNeq *pDstate,
			 realNeq *pTresLDA0, realNeq *pTresLDA1,
			 realNeq *pTresLDA2, const real3 *pTl, int nVertex)
{
  real half = (real) 0.5;
  real three = (real) 3.0;

  real f = (real) 0.0;
  if (selectLumpFlag == 1) f += (real) 1.0/(real) 12.0;
  if (massMatrix == 2) f += (real) 1.0/ (real) 36.0;
  
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

  realNeq dW0 = pDstate[v1];
  realNeq dW1 = pDstate[v2];
  realNeq dW2 = pDstate[v3];
  
  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  // |T|/(12*dt)
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*f/dt;

  realNeq ResLDA0 = (-three*dW0 + dW1 + dW2)*Adt;
  realNeq ResLDA1 = (dW0 - three*dW1 + dW2)*Adt;
  realNeq ResLDA2 = (dW0 + dW1 - three*dW2)*Adt;

  pTresLDA0[n] -= ResLDA0;
  pTresLDA1[n] -= ResLDA1;
  pTresLDA2[n] -= ResLDA2; 
}

//######################################################################
//######################################################################

__global__ void
devSelectLumpLDA(int nTriangle, real dt, int massMatrix, int selectLumpFlag,
		 const int3* __restrict__ pTv, realNeq *pDstate,
		 realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
		 const real3 *pTl, int nVertex)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    SelectLumpLDASingle(n, dt, massMatrix, selectLumpFlag, pTv, pDstate,
			pTresLDA0, pTresLDA1, pTresLDA2,
			pTl, nVertex);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
//######################################################################

void Simulation::SelectLumpLDA(real dt, int massMatrix, int selectLumpFlag)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();
  
  realNeq *pDstate = vertexStateDiff->GetPointer();
  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);
  
  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *pTl  = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devSelectLumpLDA, 
				       (size_t) 0, 0);

    devSelectLumpLDA<<<nBlocks, nThreads>>>
      (nTriangle, dt, massMatrix, selectLumpFlag,
       pTv, pDstate, pTresLDA0, pTresLDA1, pTresLDA2,
       pTl, nVertex);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      SelectLumpLDASingle(n, dt, massMatrix, selectLumpFlag, pTv, pDstate,
			  pTresLDA0, pTresLDA1, pTresLDA2,
			  pTl, nVertex);
  }
}

}
