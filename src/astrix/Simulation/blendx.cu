// -*-c++-*-
/*! \file blendx.cu
\brief File containing functions to calculate shock sensor, used in the BX scheme*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"

namespace astrix {
  
//######################################################################
/*! \brief Calculate shock sensor at triangle i

\param i Triangle to consider
\param nVertex Total number of vertices in Mesh
\param *pTv Pointer to triangle vertices
\param *pTl Pointer to triangle edge lengths
\param *pTn1 Pointer to inward pointing normal first edge of triangle
\param *pTn2 Pointer to inward pointing normal second edge of triangle
\param *pTn3 Pointer to inward pointing normal third edge of triangle
\param iDv 1/(maxVel - minVel)
\param *pState Pointer to state at vertices
\param *pShockSensor Pointer to shock sensor (output)
\param G Specific heat ratio
\param G1 \a G - 1*/
//######################################################################

__host__ __device__
void CalcShockSensorSingle(int i, int nVertex, 
			   const int3* __restrict__ pTv,
			   const real3* __restrict__ pTl,
			   const real2 *pTn1, const real2 *pTn2,
			   const real2 *pTn3, real iDv, real4 *pState,
			   real *pShockSensor, const real G, const real G1)
{
  const real zero = (real) 0.0;
  const real half  = (real) 0.5;
  const real one = (real) 1.0;
  const real tenth = (real) 0.1;

  // Triangle vertices
  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // State at first vertex
  real dens = pState[a].x;
  real momx = pState[a].y;
  real momy = pState[a].z;
  real ener = pState[a].w;

  real id1 = one/dens;
  real u1 = momx;
  real v1 = momy;
  real p1 = G1*(ener - half*id1*(u1*u1 + v1*v1));

  // Make sure divergence is finite when velocity is zero
  u1 += tenth*sqrt(G*p1*id1);

  // State at second vertex
  dens = pState[b].x;
  momx = pState[b].y;
  momy = pState[b].z;
  ener = pState[b].w;

  real id2 = one/dens;
  real u2 = momx;
  real v2 = momy;
  real p2 = G1*(ener - half*id2*(u2*u2 + v2*v2));

  // Make sure divergence is finite when velocity is zero
  u2 += tenth*sqrt(G*p2*id2);

  // State at third vertex
  dens = pState[c].x;
  momx = pState[c].y;
  momy = pState[c].z;
  ener = pState[c].w;

  real id3 = one/dens;
  real u3 = momx;
  real v3 = momy;
  real p3 = G1*(ener - half*id3*(u3*u3 + v3*v3));

  // Make sure divergence is finite when velocity is zero
  u3 += tenth*sqrt(G*p3*id3);

  // Triangle edge lengths
  real tl1 = pTl[i].x;
  real tl2 = pTl[i].y;
  real tl3 = pTl[i].z;

  real s = half*(tl1 + tl2 + tl3);
  // 1/triangle area
  real iA = rsqrtf(s*(s - tl1)*(s - tl2)*(s - tl3));
  // Shock sensor
  real sc = -half*iDv*iA;
  
  // Triangle inward pointing normals
  real nx1 = pTn1[i].x*tl1;
  real ny1 = pTn1[i].y*tl1;
  real nx2 = pTn2[i].x*tl2;
  real ny2 = pTn2[i].y*tl2;
  real nx3 = pTn3[i].x*tl3;
  real ny3 = pTn3[i].y*tl3;

  // Velocity divergence
  real divuc = u1*nx1 + v1*ny1 + u2*nx2 + v2*ny2 + u3*nx3 + v3*ny3;
  
  sc = max(sc*divuc, zero);

  // Output shock sensor
  pShockSensor[i] = min(one, sc*sc*rsqrtf(iA));
}

// Version for 1 equation just says shock 
__host__ __device__
void CalcShockSensorSingle(int i, int nVertex, 
			   const int3* __restrict__ pTv,
			   const real3* __restrict__ pTl,
			   const real2 *pTn1, const real2 *pTn2,
			   const real2 *pTn3, real iDv, real *pState,
			   real *pShockSensor, const real G, const real G1)
{
  // Output shock sensor
  pShockSensor[i] = (real) 1.0;
}

// #########################################################################
/*! \brief Kernel calculating shock sensor at triangles

\param nVertex Total number of vertices in Mesh
\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pTl Pointer to triangle edge lengths
\param *pTn1 Pointer to inward pointing normal first edge of triangle
\param *pTn2 Pointer to inward pointing normal second edge of triangle
\param *pTn3 Pointer to inward pointing normal third edge of triangle
\param iDv 1/(maxVel - minVel)
\param *pState Pointer to state at vertices
\param *pShockSensor Pointer to shock sensor (output)
\param G Specific heat ratio
\param G1 \a G - 1*/
// #########################################################################

__global__ void 
devCalcShockSensor(int nVertex, int nTriangle,
		   const int3* __restrict__ pTv,
		   const real3* __restrict__ pTl,
		   const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
		   real iDv, realNeq *pState, real *pShockSensor,
		   const real G, const real G1)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    CalcShockSensorSingle(i, nVertex, pTv, pTl, pTn1, pTn2, pTn3,
			  iDv, pState, pShockSensor, G, G1);
    
    i += gridDim.x*blockDim.x;
  }
}
  
//##############################################################################
/*! Calculate shock sensor for every triangle in the Mesh. This is needed for the BX scheme. Result in \a triangleShockSensor.*/ 
//##############################################################################
  
void Simulation::CalcShockSensor()
{
  // Determine minimum/maximum velocity on mesh
  real minVel = 0.0;
  real maxVel = 1.0;
  FindMinMaxVelocity(minVel, maxVel);
  
  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();
  
  // Triangle vertex indices
  const int3 *pTv = mesh->TriangleVerticesData();
  
  // Inward pointing edge normals
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);
  
  // Edge lengths
  const real3 *pTl = mesh->TriangleEdgeLengthData();
  
  // State at vertices
  realNeq *pState = vertexState->GetPointer();

  // Shock sensor
  triangleShockSensor->SetSize(nTriangle);
  real *pShockSensor = triangleShockSensor->GetPointer();
  
  // Calculate operators: vertex-based and triangle-based
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCalcShockSensor, 
				       (size_t) 0, 0);

    devCalcShockSensor<<<nBlocks, nThreads>>>
      (nVertex, nTriangle, pTv, pTl, pTn1, pTn2, pTn3,
       1.0/(maxVel - minVel), pState, pShockSensor,
       specificHeatRatio, specificHeatRatio - 1.0);
  
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++) 
      CalcShockSensorSingle(i, nVertex, pTv, pTl, pTn1, pTn2, pTn3,
			    1.0/(maxVel - minVel), pState, pShockSensor,
			    specificHeatRatio, specificHeatRatio - 1.0);
  }
}

}
