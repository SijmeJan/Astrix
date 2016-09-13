// -*-c++-*-
/*! \file encroach.cu
\brief Functions for determining whether removing a vertex leads to encroached segments.
*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {
  
//#########################################################################
/*! \brief  Check if removing vertex \v leads to encroached segment


\param n Index in \a vertexRemove Array
\param nVertex Total number of vertices in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *predicates Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pVertexTriangle Pointer to list of triangles containing *pVertexRemove[]
\param *pVertexRemove List of vertices to be removed
\param *pVertexRemoveFlag Flag whether vertex can be removed. Initially every entry should be 1, and is set to zero if removing vertex would lead to encroached segment*/
//#########################################################################

__host__ __device__
void CheckEncroachCoarsenSingle(int n, int nVertex,
				int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
				real Px, real Py,
				Predicates *predicates,
				real *pParam, int *pVertexTriangle,
				int *pVertexRemove,
				int *pVertexRemoveFlag)
{
  real zero = (real) 0.0;

  int v = pVertexRemove[n];
  int tStart = pVertexTriangle[n];
  
  int a = pTv[tStart].x;
  int b = pTv[tStart].y;
  int c = pTv[tStart].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = pTe[tStart].x; 
  int e2 = pTe[tStart].y;
  int e3 = pTe[tStart].z;
 
  int eStart = e1;
  if (b == v) eStart = e2;
  if (c == v) eStart = e3;
  
  int t = tStart;
  int e = eStart;
  int tNext = -1;
  int eNext = -1;
  
  // Move in clockwise direction around v
  int finished = 0;
  while (finished == 0) {
    // Move across edge e

    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    
    tNext = t1;
    if (tNext == t) tNext = t2;

    if (tNext == -1 || tNext == tStart) {
      finished = 1;
    } else {
      t = tNext;

      int e1 = pTe[t].x; 
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      
      // Move across *next* edge
      if (e == e1) eNext = e2;
      if (e == e2) eNext = e3;
      if (e == e3) eNext = e1;
      e = eNext;
    }
  }
  
  // Did we hit a segment?
  if (tNext == -1) {
    // If so, we have found one end triangle
    int tEnd1 = t;

    int e1 = pTe[tStart].x; 
    int e2 = pTe[tStart].y;
    int e3 = pTe[tStart].z;

    eStart = e3;
    if (b == v) eStart = e1;
    if (c == v) eStart = e2;
    
    t = tStart;
    e = eStart;
    eNext = -1;
    
    // Move across edge e
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;

    tNext = t1;
    if (tNext == t) tNext = t2;
    
    finished = 0;
    if (tNext == -1) {
      finished = 1;
    } else {
      t = tNext;

      int e1 = pTe[t].x; 
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;
      // Move across *previous* edge
      if (e == e1) eNext = e3;
      if (e == e2) eNext = e1;
      if (e == e3) eNext = e2;
      e = eNext;
    }
    
    // Move in counterclockwise direction around v
    while (finished == 0) {
      // Move across edge e
      int t1 = pEt[e].x;
      int t2 = pEt[e].y;
      tNext = t1;
      if (tNext == t) tNext = t2;
      
      if (tNext == -1) {
	finished = 1;
      } else {
	t = tNext;

	int e1 = pTe[t].x; 
	int e2 = pTe[t].y;
	int e3 = pTe[t].z;
	
	// Move across *previous* edge
	if (e == e1) eNext = e3;
	if (e == e2) eNext = e1;
	if (e == e3) eNext = e2;
	e = eNext;
      }
    }
    int tEnd2 = t;

    // Now find the coordinates of the segment formed by removing v
    t = tEnd1;
    
    int A = pTv[t].x;
    int B = pTv[t].y;
    int C = pTv[t].z;

    real Ax, Bx, Cx, Ay, By, Cy;
    GetTriangleCoordinates(pVc, A, B, C,
			   nVertex, Px, Py,
			   Ax, Bx, Cx, Ay, By, Cy);
    
    // Possibly translate whole triangle
    // pERIODIC
    TranslateTriangleToVertex(v, Px, Py, nVertex, A, B, C,
			      Ax, Ay, Bx, By, Cx, Cy);
    
    while (A >= nVertex) A -= nVertex;
    while (B >= nVertex) B -= nVertex;
    while (C >= nVertex) C -= nVertex;
    while (A < 0) A += nVertex;
    while (B < 0) B += nVertex;
    while (C < 0) C += nVertex;
    
    e1 = pTe[t].x; 
    e2 = pTe[t].y;
    e3 = pTe[t].z;
    
    int t12 = pEt[e2].x;
    int t22 = pEt[e2].y;
    int t13 = pEt[e3].x;
    int t23 = pEt[e3].y;

    real vx = Bx;
    real vy = By;
    if ((t12 == -1 || t22 == -1) && B == v) {
      vx = Cx;
      vy = Cy;
    }
    if ((t13 == -1 || t23 == -1) && C == v) {
      vx = Ax;
      vy = Ay;
    }
    
    t = tEnd2;

    int D = pTv[t].x;
    int E = pTv[t].y;
    int F = pTv[t].z;
    
    real Dx, Ex, Fx, Dy, Ey, Fy;
    GetTriangleCoordinates(pVc, D, E, F,
			   nVertex, Px, Py, 
			   Dx, Ex, Fx, Dy, Ey, Fy);
    
    
    // Possibly translate whole triangle
    TranslateTriangleToVertex(v, Px, Py, nVertex, D, E, F,
			      Dx, Dy, Ex, Ey, Fx, Fy);
    
    while (D >= nVertex) D -= nVertex;
    while (E >= nVertex) E -= nVertex;
    while (F >= nVertex) F -= nVertex;
    while (D < 0) D += nVertex;
    while (E < 0) E += nVertex;
    while (F < 0) F += nVertex;
    
    e1 = pTe[t].x; 
    e2 = pTe[t].y;
    e3 = pTe[t].z;

    t12 = pEt[e2].x;
    t22 = pEt[e2].y;
    t13 = pEt[e3].x;
    t23 = pEt[e3].y;

    real ux = Dx;
    real uy = Dy;
    real wx = Ex;
    real wy = Ey;
    if ((t12 == -1 || t22 == -1) && F == v) {
      ux = Ex;
      uy = Ey;
      wx = Fx;
      wy = Fy;
    }
    if ((t13 == -1 || t23 == -1) && D == v) {
      ux = Fx;
      uy = Fy;
      wx = Dx;
      wy = Dy;
    }
    
    real det = predicates->orient2d(ux, uy, wx, wy, vx, vy, pParam);

    int encroachedFlag = 0;
    // Never remove vertex from segment if not part of straight line
    if (det != zero) encroachedFlag = 1;
    
    if (det == zero) {
      tStart = tEnd2;

      int b = pTv[tStart].y;
      int c = pTv[tStart].z;
      
      while (b >= nVertex) b -= nVertex;
      while (c >= nVertex) c -= nVertex;
      while (b < 0) b += nVertex;
      while (c < 0) c += nVertex;
      
      int e1 = pTe[tStart].x; 
      int e2 = pTe[tStart].y;
      int e3 = pTe[tStart].z;

      eStart = e1;
      if (b == v) eStart = e2;
      if (c == v) eStart = e3;
      
      t = tStart;
      e = eStart;
      tNext = -1;
      eNext = -1;
      
      // Move in clockwise direction around v
      finished = 0;
      // Check all triangles containing v if they encroach UV
      while (finished == 0) {
	int A = pTv[t].x;
	int B = pTv[t].y;
	int C = pTv[t].z;

	real Ax, Bx, Cx, Ay, By, Cy;
	GetTriangleCoordinates(pVc, A, B, C,
			       nVertex, Px, Py, 
			       Ax, Bx, Cx, Ay, By, Cy);
	
	// Need to translate triangle
	TranslateTriangleToVertex(v, Px, Py, nVertex, A, B, C,
				  Ax, Ay, Bx, By, Cx, Cy);
    
	while (A >= nVertex) A -= nVertex;
	while (B >= nVertex) B -= nVertex;
	while (C >= nVertex) C -= nVertex;
	while (A < 0) A += nVertex;
	while (B < 0) B += nVertex;
	while (C < 0) C += nVertex;
	
	real x = Ax;
	real y = Ay;
	if (v == A) {
	  x = Bx;
	  y = By;
	}
	if (v == B) {
	  x = Cx;
	  y = Cy;
	}
	
	real dot = (ux - x)*(vx - x) + (uy - y)*(vy - y);
	if (dot < 0.0f) {
	  encroachedFlag = 1;
	  finished = 1;
	}
	
	// Move across edge e
	int t1 = pEt[e].x;
	int t2 = pEt[e].y;
	
	tNext = t1;
	if (tNext == t) tNext = t2;
	
	if (tNext == -1 || tNext == tStart) {
	  finished = 1;
	} else {
	  t = tNext;
	  int e1 = pTe[t].x; 
	  int e2 = pTe[t].y;
	  int e3 = pTe[t].z;

	  // Move across *next* edge
	  if (e == e1) eNext = e2;
	  if (e == e2) eNext = e3;
	  if (e == e3) eNext = e1;
	  e = eNext;
	}
      }
    }
    
    if (encroachedFlag == 1) pVertexRemoveFlag[n] = 0;
  }
}
  
//#########################################################################
/*! \brief  Check if removing vertex \v leads to encroached segment

\param nRemove Number of vertices to be removed
\param *pVertexRemove List of vertices to be removed
\param nVertex Total number of vertices in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *predicates Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pVertexTriangle Pointer to list of triangles containing *pVertexRemove[]
\param *pVertexRemoveFlag Flag whether vertex can be removed. Initially every entry should be 1, and is set to zero if removing vertex would lead to encroached segment*/ 
//#########################################################################

__global__
void devCheckEncroachCoarsen(int nRemove, int *pVertexRemove, int nVertex,
			     int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
			     real Px, real Py, Predicates *predicates,
			     real *pParam, int *pVertexTriangle,
			     int *pVertexRemoveFlag)
{     
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    CheckEncroachCoarsenSingle(n, nVertex,
			       pTv, pTe, pEt,
			       pVc, Px, Py,
			       predicates, pParam, pVertexTriangle,
			       pVertexRemove, pVertexRemoveFlag);

    n += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
/*! Check if removing any vertex in \a vertexRemove would lead to an encroached segment. If vertex \a vertexRemove[i] leads to an encroached segment, \a vertexRemoveFlag[i] is set to zero, otherwise it is left unchanged.

\param vertexRemoveFlag Pointer to output array.*/
//#########################################################################

void Coarsen::CheckEncroach(Connectivity *connectivity,
			    Predicates *predicates,
			    Array<int> *vertexRemoveFlag,
			    const MeshParameter *mp)
{
  int transformFlag = 0;
  int nRemove = vertexRemove->GetSize();
  
  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemove->TransformToHost();
      vertexTriangle->TransformToHost(); 
      vertexRemoveFlag->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemove->TransformToDevice();
      vertexTriangle->TransformToDevice(); 
      vertexRemoveFlag->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();
  
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  int *pVertexRemove = vertexRemove->GetPointer();
  int *pVertexTriangle = vertexTriangle->GetPointer();
  int *pVertexRemoveFlag = vertexRemoveFlag->GetPointer();

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCheckEncroachCoarsen, 
				       (size_t) 0, 0);

    devCheckEncroachCoarsen<<<nBlocks, nThreads>>>
      (nRemove, pVertexRemove, nVertex,
       pTv, pTe, pEt, pVc, Px, Py,
       predicates, pParam, pVertexTriangle,
       pVertexRemoveFlag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    // Check if removing vertex would lead to encroached segment
    for (int n = 0; n < nRemove; n++) 
      CheckEncroachCoarsenSingle(n, nVertex,
				 pTv, pTe, pEt, pVc,
				 Px, Py,
				 predicates, pParam, pVertexTriangle,
				 pVertexRemove, pVertexRemoveFlag);
  }
  
  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemove->TransformToHost();
      vertexTriangle->TransformToHost(); 
      vertexRemoveFlag->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemove->TransformToDevice();
      vertexTriangle->TransformToDevice(); 
      vertexRemoveFlag->TransformToDevice();

      cudaFlag = 1;
    }
  }
}

}
