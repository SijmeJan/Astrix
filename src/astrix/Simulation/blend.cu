// -*-c++-*-
/*! \file blend.cu
\brief File containing function calculating blend factor*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################################
/*! \brief Calculate blend parameter for triangle \a n

\param n Triangle to consider
\param *pTl Pointer to triangle edge lengths
\param *pTresN0 Pointer to triangle residue N direction 0
\param *pTresN1 Pointer to triangle residue N direction 1
\param *pTresN2 Pointer to triangle residue N direction 2
\param *pTres Pointer to triangle total residue
\param *pBlend Pointer to blend parameter (output)*/
//######################################################################

__host__ __device__
void CalcBlendSingle(int n, const real3* __restrict__ pTl, 
		     real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
		     real4 *pTres, real4 *pBlend)
{
  real small = (real) 1.0e-6;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Total residual
  real tTot0 = pTres[n].x;
  real tTot1 = pTres[n].y;
  real tTot2 = pTres[n].z;
  real tTot3 = pTres[n].w;

  // N residuals
  real tN00 = pTresN0[n].x;
  real tN01 = pTresN0[n].y;
  real tN02 = pTresN0[n].z;
  real tN03 = pTresN0[n].w;

  real tN10 = pTresN1[n].x;
  real tN11 = pTresN1[n].y;
  real tN12 = pTresN1[n].z;
  real tN13 = pTresN1[n].w;

  real tN20 = pTresN2[n].x;
  real tN21 = pTresN2[n].y;
  real tN22 = pTresN2[n].z;
  real tN23 = pTresN2[n].w;

  // Calculate blend parameter
  pBlend[n].x = fabs(tTot0)/
    (fabs(tN00)*tl1 + fabs(tN10)*tl2 + fabs(tN20)*tl3 + small);
  pBlend[n].y = fabs(tTot1)/
    (fabs(tN01)*tl1 + fabs(tN11)*tl2 + fabs(tN21)*tl3 + small);
  pBlend[n].z = fabs(tTot2)/
    (fabs(tN02)*tl1 + fabs(tN12)*tl2 + fabs(tN22)*tl3 + small);
  pBlend[n].w = fabs(tTot3)/
    (fabs(tN03)*tl1 + fabs(tN13)*tl2 + fabs(tN23)*tl3 + small);
}
  
//######################################################################
/*! \brief Kernel calculating blend parameter for all triangles

\param nTriangle Total number of triangles in Mesh
\param *pTl Pointer to triangle edge lengths
\param *pTresN0 Pointer to triangle residue N direction 0
\param *pTresN1 Pointer to triangle residue N direction 1
\param *pTresN2 Pointer to triangle residue N direction 2
\param *pTres Pointer to triangle total residue
\param *pBlend Pointer to blend parameter (output)*/

//######################################################################

__global__ void 
devCalcBlendFactor(int nTriangle, const real3* __restrict__ pTl, 
		   real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
		   real4 *pTres, real4 *pBlend)
{
  // n=vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nTriangle){
    CalcBlendSingle(n, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend);

    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate blend parameter for all triangles. When using the B scheme, we need to blend the N and LDA residuals. This function calculates the blend parameter \a triangleBlendFactor*/
//######################################################################

void Simulation::CalcBlend()
{
  int nTriangle = mesh->GetNTriangle();

  // N residuals
  real4 *pTresN0 = triangleResidueN->GetPointer(0);
  real4 *pTresN1 = triangleResidueN->GetPointer(1);
  real4 *pTresN2 = triangleResidueN->GetPointer(2);

  // Total residual
  real4 *pTres = triangleResidueTotal->GetPointer();

  // Blend parameter (output)
  real4 *pBlend = triangleBlendFactor->GetPointer();

  // Triangle edge lengths
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcBlendFactor, 
				       (size_t) 0, 0);
    
    // Execute kernel... 
    devCalcBlendFactor<<<nBlocks,nThreads>>>
      (nTriangle, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      CalcBlendSingle(n, pTl, pTresN0, pTresN1, pTresN2, pTres, pBlend);
  }
}

}
