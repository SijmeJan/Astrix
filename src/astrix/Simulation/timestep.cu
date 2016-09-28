// -*-c++-*-
/*! \file timestep.cu
\brief File containing function to calculate the maximum possible time step to take.*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"

namespace astrix {
  
//######################################################################
/*! \brief Find maximum signal speed for triangle t

\param t Triangle to consider
\param a First vertex of triangle 
\param b Second vertex of triangle 
\param c Third vertex of triangle 
\param *pState Pointer to vertex state vector
\param *pTl Pointer to triangle edge lengths
\param G Ratio of specific heats
\param G1 G - 1*/
//######################################################################

__host__ __device__
real FindMaxSignalSpeed(int t, int a, int b, int c,
			real4 *pState, const real3* __restrict__ pTl,
			real G, real G1)
{
  real zero = (real) 0.0;
  real half = (real) 0.5;
  real one = (real) 1.0;
  
  real vmax = zero;

  // First vertex
  real dens = pState[a].x;
  real momx = pState[a].y;
  real momy = pState[a].z;
  real ener = pState[a].w;

  real id = one/dens;
  real u = momx;
  real v = momy;
  real absv = sqrt(u*u+v*v)*id;

  // Pressure
  real p = G1*(ener - half*id*(u*u + v*v));
  // Sound speed
  real cs = sqrt(G*p*id);

  // Maximum signal speed
  vmax = absv + cs;

  // Second vertex
  dens = pState[b].x;
  momx = pState[b].y;
  momy = pState[b].z;
  ener = pState[b].w;

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v)*id;

  p = G1*(ener - half*id*(u*u + v*v));
  cs = sqrt(G*p*id);
    
  vmax = max(vmax, absv + cs);

  // Third vertex
  dens = pState[c].x;
  momx = pState[c].y;
  momy = pState[c].z;
  ener = pState[c].w;

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v)*id;
    
  p = G1*(ener - half*id*(u*u + v*v));
  cs = sqrt(G*p*id);
    
  vmax = max(vmax, absv + cs);

  // Triangle edge lengths
  real tl1 = pTl[t].x;
  real tl2 = pTl[t].y;
  real tl3 = pTl[t].z;

  // Scale with maximum edge length
  vmax = vmax*max(tl1, max(tl2, tl3));

  return vmax;
}

__host__ __device__
real FindMaxSignalSpeed(int t, int a, int b, int c,
			real *pState, const real3* __restrict__ pTl,
			real G, real G1)
{
  // Triangle edge lengths
  real tl1 = pTl[t].x;
  real tl2 = pTl[t].y;
  real tl3 = pTl[t].z;

#if BURGERS == 1
  real vmax = max(fabs(pState[a]), max(fabs(pState[b]), fabs(pState[c])));
  return vmax*max(tl1, max(tl2, tl3));
#else
  // Scalar advection with velocity unity
  return 1.0*max(tl1, max(tl2, tl3));
#endif
}

//######################################################################
/*! \brief Find maximum signal speed for triangle t and add it atomically to all of its vertices

\param t Triangle to consider
\param *pTv Pointer to vertices of triangle 
\param *pState Pointer to vertex state vector
\param *pTl Pointer to triangle edge lengths
\param *pVts Pointer to maximum time step at vertex (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1*/
//######################################################################

__host__ __device__
void CalcVmaxSingle(int t, const int3* __restrict__ pTv, realNeq *pState,
		    const real3* __restrict__ pTl, real *pVts,
		    int nVertex, real G, real G1)
{

  int a = pTv[t].x;
  int b = pTv[t].y;
  int c = pTv[t].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  real vMax = FindMaxSignalSpeed(t, a, b, c, pState, pTl, G, G1);
  
  AtomicAdd(&pVts[a], vMax);
  AtomicAdd(&pVts[b], vMax);
  AtomicAdd(&pVts[c], vMax); 
}

//######################################################################
/*! \brief Kernel finding maximum signal speed for triangles and add it atomically to all of the vertices

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *dens Pointer to density
\param *momx Pointer to x momentum
\param *momy Pointer to y momentum
\param *ener Pointer to energy
\param *pTriangleEdgeLengths Pointer to triangle edge lengths
\param *pVts Pointer to maximum time step at vertex (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats*/
//######################################################################

__global__ void 
devCalcVmax(int nTriangle, const int3* __restrict__ pTv, realNeq *pState,
	    const real3* __restrict__ pTl, real *pVts,
	    int nVertex, real G, real G1)
{
  // n = triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nTriangle){
    CalcVmaxSingle(n, pTv, pState, pTl, pVts, nVertex, G, G1);

    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! \brief Calculate maximum allowed time step for vertex \a n

\param n Vertex to consider
\param *pVts Pointer to array containing sum of maximum signal speeds for all triangles sharing \a n
\param *pVarea Pointer to array of areas assosiated with vertices (Voronoi cells)*/
//######################################################################

__host__ __device__
void CalcVertexTimeStepSingle(int n, real *pVts, const real *pVarea)
{
  real two = (real) 2.0;
  pVts[n] = two*pVarea[n]/pVts[n];
}

//######################################################################
/*! \brief Kernel calculating maximum allowed time step for vertices

\param nVertex Total number of vertices in Mesh
\param *pVts Pointer to array containing sum of maximum signal speeds for all triangles sharing vertex
\param *pVarea Pointer to array of areas assosiated with vertices (Voronoi cells)*/
//######################################################################

__global__ void 
devCalcVertexTimeStep(int nVertex, real *pVts, const real *pVarea)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nVertex){
    CalcVertexTimeStepSingle(n, pVts, pVarea);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate maximum possible time step.*/
//######################################################################

real Simulation::CalcVertexTimeStep()
{
  unsigned int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();

  realNeq *pState = vertexState->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *pTl = mesh->TriangleEdgeLengthData();
  const real *pVarea = mesh->VertexAreaData();

  Array<real> *vertexTimestep = new Array<real>(1, cudaFlag, nVertex);
  vertexTimestep->SetToValue(0.0);
  real *pVts = vertexTimestep->GetPointer();

  // First calculate maximum signal speed
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcVmax, 
				       (size_t) 0, 0);

    // Execute kernel... 
    devCalcVmax<<<nBlocks,nThreads>>>
      (nTriangle, pTv, pState, pTl, pVts, nVertex,
       specificHeatRatio, specificHeatRatio - 1.0);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      CalcVmaxSingle(n, pTv, pState, pTl, pVts, nVertex,
		     specificHeatRatio, specificHeatRatio - 1.0);
  }

  // Convert maximum signal speed into vertex time step
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcVertexTimeStep, 
				       (size_t) 0, 0);

    // Execute kernel... 
    devCalcVertexTimeStep<<<nBlocks,nThreads>>>
      (nVertex, pVts, pVarea);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (unsigned int n = 0; n < nVertex; n++)
      CalcVertexTimeStepSingle(n, pVts, pVarea);
  }

  // Find the minimum
  real dt = CFLnumber*vertexTimestep->Minimum();
  
  // End exactly on maxSimulationTime
  if (simulationTime + dt > maxSimulationTime) 
    dt = maxSimulationTime - simulationTime;
  
  delete vertexTimestep;
  
  return dt;
}  


}
