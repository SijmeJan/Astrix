// -*-c++-*-
/*! \file interpolate.cu
\brief Functions for interpolating state when refining Mesh*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/helper_math.h"

namespace astrix {
    
//######################################################################
/*! \brief Interpolate state when inserting point (x, y)

Use linear interpolation to determine state at new vertex. 

\param t Triangle into which to insert point (x, y)
\param e Edge onto which to insert point (x, y)
\param indexInVertexArray Index of vertex to insert
\param *dens Pointer to density
\param *momx Pointer to x-momentum
\param *momy Pointer to y-momentum
\param *ener Pointer to energy
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param x x-coordinate of point to insert
\param y y-coordinate of point to insert
\param *wantRefine Pointer to array of flags specifying if triangle needs to be refined. Set to zero for all inserted points.
\param G Ratio of specific heats
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector*/
//######################################################################

__host__ __device__
void InterpolateSingle(int t, int e, int indexInVertexArray, realNeq *state,
		       int3 *pTv, int2 *pEt, real2 *pVc,
		       real x, real y,
		       int *wantRefine, real G,
		       int nVertex, real Px, real Py)
{
  const real half  = (real) 0.5;

  // Unflag triangles to be refined
  if (t != -1)
    wantRefine[t] = 0;
  if (t == -1) t = pEt[e].x;
  if (t != -1)
    wantRefine[t] = 0;
  if (t == -1) t = pEt[e].y;
  if (t != -1)
    wantRefine[t] = 0;  

  int v1 = pTv[t].x;
  int v2 = pTv[t].y;
  int v3 = pTv[t].z;
   
  real ax, bx, cx, ay, by, cy, dx = x, dy = y;
  GetTriangleCoordinates(pVc, v1, v2, v3,
			 nVertex, Px, Py,
			 ax, bx, cx, ay, by, cy);

  real T = half*((ax - cx)*(by - cy) - (ay - cy)*(bx - cx));
  real A = half*((dx - cx)*(by - cy) - (dy - cy)*(bx - cx));
  real B = half*((ax - cx)*(dy - cy) - (ay - cy)*(dx - cx));
  real C = half*((ax - dx)*(by - dy) - (ay - dy)*(bx - dx));
  
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;
  
  state[indexInVertexArray] =
    (A*state[v1] + B*state[v2] + C*state[v3])/T;
  /*
  state[indexInVertexArray].x =
    (A*state[v1].x + B*state[v2].x + C*state[v3].x)/T;
  state[indexInVertexArray].y =
    (A*state[v1].y + B*state[v2].y + C*state[v3].y)/T;
  state[indexInVertexArray].z =
    (A*state[v1].z + B*state[v2].z + C*state[v3].z)/T;
  state[indexInVertexArray].w =
    (A*state[v1].w + B*state[v2].w + C*state[v3].w)/T;
  */
}
  
//######################################################################
/*! \brief Kernel interpolating state for all points to be inserted

Use linear interpolation to determine state at new vertices. 

\param nRefine Total number of points to add
\param *pTriangleAdd Pointer to array of triangles into which to insert points
\param *pEdgeAdd Pointer to array of edges onto which to insert points
\param *dens Pointer to density
\param *momx Pointer to x-momentum
\param *momy Pointer to y-momentum
\param *ener Pointer to energy
\param nVertex Total number of vertices in Mesh
\param *refineX x-coordinates of points to insert
\param *refineY y-coordinates of points to insert
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *wantRefine Pointer to array of flags specifying if triangle needs to be refined. Set to zero for all inserted points.
\param G Ratio of specific heats
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//######################################################################

__global__ void
devInterpolateState(int nRefine, int *pElementAdd, realNeq *state,
		    int nVertex, int nTriangle, real2 *pVcAdd, 
		    real2 *pVc, int3 *pTv, int2 *pEt,
		    int *wantRefine, real G, real Px, real Py)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    int t = pElementAdd[i];
    int e = -1;
    if (t >= nTriangle) {
      e = t - nTriangle;
      t = -1;
    }
    
    InterpolateSingle(t, e, i + nVertex, state,
		      pTv, pEt, pVc,
		      pVcAdd[i].x, pVcAdd[i].y, wantRefine, G,
		      nVertex, Px, Py);
    
    i += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Use linear interpolation to give newly inserted vertices a state

\param *vertexState Pointer to Array containing state vector
\param specificHeatRatio Ratio of specific heats*/
//######################################################################

void Refine::InterpolateState(Connectivity * const connectivity,
			      const MeshParameter *meshParameter,
			      Array<realNeq> * const vertexState,
			      Array<int> * const triangleWantRefine,
			      const real specificHeatRatio)
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nRefine = elementAdd->GetSize();
  
  // Interpolate state
  vertexState->SetSize(nVertex + nRefine);
  realNeq *state = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  
  int *pWantRefine = triangleWantRefine->GetPointer();

  int *pElementAdd = elementAdd->GetPointer();
  
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devInterpolateState, 
				       (size_t) 0, 0);

    devInterpolateState<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd, state,
       nVertex, nTriangle, pVcAdd, pVc, pTv, pEt,
       pWantRefine, specificHeatRatio, Px, Py);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRefine; n++) {
      int t = pElementAdd[n];
      int e = -1;
      if (t >= nTriangle) {
	e = t - nTriangle;
	t = -1;
      }
      
      InterpolateSingle(t, e, n + nVertex, state,
			pTv, pEt, pVc, pVcAdd[n].x, pVcAdd[n].y,
			pWantRefine, specificHeatRatio, nVertex, Px, Py);
    }
  }  
}
  
}
