// -*-c++-*-
/*! \file kh.cu
\brief Diagnostics for Kelvin-Helmholtz test problem*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

namespace astrix {

//##############################################################################
//##############################################################################
  
__host__ __device__
void CalcDiagnosticsSingle(unsigned int i, const real2 *pVc, 
			   const real *pVarea,
			   real4 *pState, real *d, real *s, real *c, real *E)
{
  real x = pVc[i].x;
  real y = pVc[i].y;
  
  d[i] = pVarea[i]*exp(-4.0*M_PI*abs(1.0 - y - 0.25));
  if (y < 0.5) d[i] = pVarea[i]*exp(-4.0*M_PI*abs(y - 0.25));

  real dens = pState[i].x;
  real momy = pState[i].z;
  
  s[i] = momy*sin(4.0*M_PI*x)*d[i]/dens;
  c[i] = momy*cos(4.0*M_PI*x)*d[i]/dens;

  E[i] = 0.5*momy*momy/dens;
}
  
__host__ __device__
void CalcDiagnosticsSingle(unsigned int i, const real2 *pVc, 
			   const real *pVarea,
			   real *pState, real *d, real *s, real *c, real *E)
{
  // Dummy function; nothing to be done if solving only one equation
}
  
//######################################################################
//######################################################################

__global__ void 
devCalcDiagnostics(unsigned int nVertex, const real2 *pVc, const real *pVarea,
		   realNeq *pState, real *d, real *s, real *c, real *E)
{
  // n = vertex number
  unsigned int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nVertex){
    CalcDiagnosticsSingle(n, pVc, pVarea, pState, d, s, c, E);
 
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
//######################################################################

void Simulation::KHDiagnostics(real& M, real& Ekin)
{
  unsigned int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  const real *pVarea = mesh->VertexAreaData();
  const real2 *pVc = mesh->VertexCoordinatesData();
  
  Array<real> *D = new Array<real>(1, cudaFlag, nVertex);
  Array<real> *S = new Array<real>(1, cudaFlag, nVertex);
  Array<real> *C = new Array<real>(1, cudaFlag, nVertex);
  Array<real> *E = new Array<real>(1, cudaFlag, nVertex);

  real *pD = D->GetPointer();
  real *pS = S->GetPointer();
  real *pC = C->GetPointer();
  real *pE = E->GetPointer();
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcDiagnostics, 
				       (size_t) 0, 0);

    devCalcDiagnostics<<<nBlocks, nThreads>>>
      (nVertex, pVc, pVarea, pState, pD, pS, pC, pE);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for(unsigned int n = 0; n < nVertex; n++) 
      CalcDiagnosticsSingle(n, pVc, pVarea, pState, pD, pS, pC, pE);
  }
  
  real d = D->Sum();
  real c = C->Sum();
  real s = S->Sum();

  M = 2.0*sqrt(Sq(s/d) + Sq(c/d));
  Ekin = E->Maximum();

  delete D;
  delete S;
  delete C;
  delete E; 
}

}
