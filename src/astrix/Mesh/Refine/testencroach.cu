// -*-c++-*-
/*! \file testencroach.cu
\brief File containing function to test if any points to add encroaches on a segment.*/
#include <iostream>
#include <fstream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//##############################################################################
/*! \brief Check if point (\a x, \a y) encroaches any segment in triangle \a t

If point (x,y) is found to encroach on any segment in \a t, adjust \a x and \a y to lie at the centre of the segment and return 1; else return -1.

\param t Triangle to consider
\param x x-coordinate of point, may be changed
\param y y-coordinate of point, may be changed
\param *et1 pointer to triangle 1 neighbouring edge.
\param *et2 pointer to triangle 2 neighbouring edge.
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 pointer to edge 1 part of triangle.
\param *te2 pointer to edge 2 part of triangle.
\param *te3 pointer to edge 3 part of triangle.
\param *pVertX pointer x-coordinates of existing vertices.
\param *pVertY pointer y-coordinates of existing vertices.
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//##############################################################################

__host__ __device__
int CheckEncroachTriangle(real& x, real& y,	      
			  const int e1IsSegment,
			  const int e2IsSegment,
			  const int e3IsSegment,
			  const int a, const int b , const int c,
			  const int e1, const int e2, const int e3, 
			  const real2* __restrict__ pVc,
			  int nVertex, real Px, real Py)
{
  real zero = (real) 0.0;
  real half = (real) 0.5;
  
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
  			 nVertex, Px, Py,
  			 ax, bx, cx, ay, by, cy);

  real dot1 = (ax - x)*(bx - x) + (ay - y)*(by - y);
  real dot2 = (bx - x)*(cx - x) + (by - y)*(cy - y);
  real dot3 = (cx - x)*(ax - x) + (cy - y)*(ay - y);

  // Encroached edge
  int e = -1;
  if (e1IsSegment && dot1 < zero) e = e1;
  if (e2IsSegment && dot2 < zero) e = e2;
  if (e3IsSegment && dot3 < zero) e = e3;

  // Move vertex if necessary
  if (e == e1) {
    x = half*(ax + bx);
    y = half*(ay + by);
  }
  if (e == e2) {
    x = half*(bx + cx);
    y = half*(by + cy);
  }
  if (e == e3) {
    x = half*(cx + ax);
    y = half*(cy + ay);
  }
  
  return e;
}

//######################################################################
/*! \brief Test single point if it encroaches a segment. 

This function tests if point to add \a i leads to an encroached segment. We take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encroached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question. 

 \param i index of point to be inserted.
 \param *pEdgeAdd pointer to array of edges to place vertices on.
 \param *pTriangleAdd pointer to array of triangles to place vertices in.
 \param *refineX pointer to array of x-coordinates of points to be inserted.
 \param *refineY pointer to array of y-coordinates of points to be inserted.
 \param *et1 pointer to triangle 1 neighbouring edge.
 \param *et2 pointer to triangle 2 neighbouring edge.
 \param *tv1 Pointer to first vertex of triangle 
 \param *tv2 Pointer to second vertex of triangle 
 \param *tv3 Pointer to third vertex of triangle 
 \param *te1 pointer to edge 1 part of triangle.
 \param *te2 pointer to edge 2 part of triangle.
 \param *te3 pointer to edge 3 part of triangle.
 \param *pVertX pointer x-coordinates of existing vertices.
 \param *pVertY pointer y-coordinates of existing vertices.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y*/
//######################################################################

__host__ __device__
void TestEncroachSingle(int i, int *pElementAdd, real2 *pVcAdd,
			const int3* __restrict__ pTv,
			const int3* __restrict__ pTe,
			const int2* __restrict__ pEt,
			const real2* __restrict__ pVc,
			int nVertex, real Px, real Py, int nTriangle)
{
  int tStart = pElementAdd[i];
  int eStart = -1;
  if (tStart >= nTriangle) {
    eStart = tStart - nTriangle;
    tStart = -1;
  }
  
  // Check if starting on edge
  if (tStart == -1) {
    int t1 = pEt[eStart].x;
    int t2 = pEt[eStart].y;

    tStart = t1;
    if (tStart != -1) {
      int a = pTv[tStart].x;
      int b = pTv[tStart].y;
      int c = pTv[tStart].z;
      
      real ax, bx, cx, ay, by, cy;
      GetTriangleCoordinates(pVc, a, b, c,
			     nVertex, Px, Py,
			     ax, bx, cx, ay, by, cy);
      real x = pVcAdd[i].x;
      real y = pVcAdd[i].y;

      // PERIODIC
      if ((ax < x && bx < x && cx < x) ||
	  (ax > x && bx > x && cx > x) ||
	  (ay < y && by < y && cy < y) ||
	  (ay > y && by > y && cy > y)) 
	tStart = t2;
      if (tStart == -1) tStart = t1;
    } else {
      tStart = t2;
    }
  }

  int3 vCheck;
  vCheck.x = pTv[tStart].x;
  vCheck.y = pTv[tStart].y;
  vCheck.z = pTv[tStart].z;

  int vCheckOrig = -1, v = -1;
  
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int n = 0; n < 3; n++) {
    if (n == 0) vCheckOrig = vCheck.x;
    if (n == 1) vCheckOrig = vCheck.y;
    if (n == 2) vCheckOrig = vCheck.z;
    if (n == 0) v = vCheck.x;
    if (n == 1) v = vCheck.y;
    if (n == 2) v = vCheck.z;
    
    while (v >= nVertex) v -= nVertex;
    while (v < 0) v += nVertex;

    int e1 = pTe[tStart].x;
    int e2 = pTe[tStart].y;
    int e3 = pTe[tStart].z;
    
    int eStart = e1;
    if (n == 1) eStart = e2;
    if (n == 2) eStart = e3;
      
    int t = tStart;
    int e = eStart;
    int tNext = -1;
    int eNext = -1;

    // Move in clockwise direction around v
    int finished = 0;
    int encroachEdge = -1;
    while (finished == 0) {
      real x = pVcAdd[i].x;
      real y = pVcAdd[i].y;

      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;
     
      if (a != vCheckOrig && b != vCheckOrig && c != vCheckOrig) {
	int N = nVertex;
	if (a > 3*N || b > 3*N || c > 3*N)
	  y = pVcAdd[i].y + Py;
	if (a < -N || b < -N || c < -N)
	  y = pVcAdd[i].y - Py;
	if ((a >= N && a < 2*N) || a >= 4*N || (a < -N && a >= -2*N) ||
	    (b >= N && b < 2*N) || b >= 4*N || (b < -N && b >= -2*N) ||
	    (c >= N && c < 2*N) || c >= 4*N || (c < -N && c >= -2*N))
	  x = pVcAdd[i].x + Px;
	if ((a >= 2*N && a < 3*N) || (a < 0 && a >= -N) || a < -3*N ||
	    (b >= 2*N && b < 3*N) || (b < 0 && b >= -N) || b < -3*N ||
	    (c >= 2*N && c < 3*N) || (c < 0 && c >= -N) || c < -3*N)
	  x = pVcAdd[i].x - Px;
      }

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;
     
      int e1IsSegment = (t11 == -1 || t21 == -1);
      int e2IsSegment = (t12 == -1 || t22 == -1);
      int e3IsSegment = (t13 == -1 || t23 == -1);
      
      if (e1IsSegment || e2IsSegment || e3IsSegment) 
	encroachEdge =
	  CheckEncroachTriangle(x, y,
				e1IsSegment, e2IsSegment, e3IsSegment,
				a, b, c,
				e1, e2, e3, 
				pVc, nVertex, Px, Py);
      
      if (encroachEdge >= 0) {
	finished = 1;
	pElementAdd[i] = nTriangle + encroachEdge;
	pVcAdd[i].x = x;
	pVcAdd[i].y = y;
      } 

      int t1 = pEt[e].x;
      int t2 = pEt[e].y;
      
      // Move across edge e
      tNext = t1;
      if (tNext == t) tNext = t2;
	
      if (tNext == -1 || tNext == tStart) finished = 1;

      if (tNext != -1) t = tNext;

      e1 = pTe[t].x;
      e2 = pTe[t].y;
      e3 = pTe[t].z;
      
      // Move across *next* edge
      if (e == e1) eNext = e2;
      if (e == e2) eNext = e3;
      if (e == e3) eNext = e1;
      e = eNext;
    }
      
    // Did we hit a segment? Reverse!
    if (tNext == -1 && encroachEdge == -1) {
      int e1 = pTe[tStart].x;
      int e2 = pTe[tStart].y;
      int e3 = pTe[tStart].z;
      
      int eStart = e3;
      if (n == 1) eStart = e1;
      if (n == 2) eStart = e2;

      t = tStart;
      e = eStart;
      eNext = -1;

      int t1 = pEt[e].x;
      int t2 = pEt[e].y;
      
      // Move across edge e
      tNext = t1;
      if (tNext == t) tNext = t2;
      
      finished = 0;
      if (tNext == -1) finished = 1;
      if (tNext != -1) t = tNext;

      e1 = pTe[t].x;
      e2 = pTe[t].y;
      e3 = pTe[t].z;
      
      // Move across *previous* edge
      if (e == e1) eNext = e3;
      if (e == e2) eNext = e1;
      if (e == e3) eNext = e2;
      e = eNext;
      
      // Move in counterclockwise direction around v
      while (finished == 0) {
	real x = pVcAdd[i].x;
	real y = pVcAdd[i].y;

	int a = pTv[t].x;
	int b = pTv[t].y;
	int c = pTv[t].z;
	
	if (a != vCheckOrig && b != vCheckOrig && c != vCheckOrig) {
	  int N = nVertex;
	  if (a > 3*N || b > 3*N || c > 3*N)
	    y = pVcAdd[i].y - Py;
	  if (a < -N || b < -N || c < -N)
	    y = pVcAdd[i].y + Py;
	  if ((a >= N && a < 2*N) || a >= 4*N || (a < -N && a >= -2*N) ||
	      (b >= N && b < 2*N) || b >= 4*N || (b < -N && b >= -2*N) ||
	      (c >= N && c < 2*N) || c >= 4*N || (c < -N && c >= -2*N))
	    x = pVcAdd[i].x - Px;
	  if ((a >= 2*N && a < 3*N) || (a < 0 && a >= -N) || a < -3*N ||
	      (b >= 2*N && b < 3*N) || (b < 0 && b >= -N) || b < -3*N ||
	      (c >= 2*N && c < 3*N) || (c < 0 && c >= -N) || c < -3*N)
	    x = pVcAdd[i].x + Px;
	}

	int e1 = pTe[t].x;
	int e2 = pTe[t].y;
	int e3 = pTe[t].z;

	int t11 = pEt[e1].x;
	int t21 = pEt[e1].y;
	int t12 = pEt[e2].x;
	int t22 = pEt[e2].y;
	int t13 = pEt[e3].x;
	int t23 = pEt[e3].y;
	
	int e1IsSegment = (t11 == -1 || t21 == -1);
	int e2IsSegment = (t12 == -1 || t22 == -1);
	int e3IsSegment = (t13 == -1 || t23 == -1);
	
	if (e1IsSegment || e2IsSegment || e3IsSegment) 
	  encroachEdge =
	     CheckEncroachTriangle(x, y,
				   e1IsSegment, e2IsSegment, e3IsSegment,
				   a, b, c,
				   e1, e2, e3, 
				   pVc, nVertex, Px, Py);
	
	if (encroachEdge >= 0) {
	  finished = 1;
	  pElementAdd[i] = nTriangle + encroachEdge;
	  pVcAdd[i].x = x;
	  pVcAdd[i].y = y;
	} 

	int t1 = pEt[e].x;
	int t2 = pEt[e].y;
	
	// Move across edge e
	tNext = t1;
	if (tNext == t) tNext = t2;
	
	if (tNext == -1) finished = 1;
	if (tNext != -1) t = tNext;

	e1 = pTe[t].x;
	e2 = pTe[t].y;
	e3 = pTe[t].z;
	
	// Move across *previous* edge
	if (e == e1) eNext = e3;
	if (e == e2) eNext = e1;
	if (e == e3) eNext = e2;
	e = eNext;
      }
    }
  }
}
  
//######################################################################
/*! \brief Kernel testing points if they encroache a segment. 

This function tests if point to add \a i leads to an encroached segment. We take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encriached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question. 

 \param nRefine number of points to be inserted.
 \param *pEdgeAdd pointer to array of edges to place vertices on.
 \param *pTriangleAdd pointer to array of triangles to place vertices in.
 \param *refineX pointer to array of x-coordinates of points to be inserted.
 \param *refineY pointer to array of y-coordinates of points to be inserted.
 \param *et1 pointer to triangle 1 neighbouring edge.
 \param *et2 pointer to triangle 2 neighbouring edge.
 \param *tv1 Pointer to first vertex of triangle 
 \param *tv2 Pointer to second vertex of triangle 
 \param *tv3 Pointer to third vertex of triangle 
 \param *te1 pointer to edge 1 part of triangle.
 \param *te2 pointer to edge 2 part of triangle.
 \param *te3 pointer to edge 3 part of triangle.
 \param *pVertX pointer x-coordinates of existing vertices.
 \param *pVertY pointer y-coordinates of existing vertices.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y*/
//######################################################################

__global__ void 
devTestEncroach(int nRefine, int *pElementAdd, real2 *pVcAdd,
		const int3* __restrict__ pTv,
		const int3* __restrict__ pTe,
		const int2* __restrict__ pEt,
		const real2* __restrict__ pVc,
		int nVertex, real Px, real Py, int nTriangle) 
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (i < nRefine) {
    TestEncroachSingle(i, pElementAdd, pVcAdd, 
		       pTv, pTe, pEt, pVc,
		       nVertex, Px, Py, nTriangle);
  
    i += gridDim.x*blockDim.x;
  }
}
  
//##############################################################################
/*! This function tests if any vertices to be inserted lead to an encroached segment. For every point \a i to be inserted, we take a triangle close to \a i (either the triangle containing \a i or, if \a i is to be placed on edge \a e, one of the neighbouring triangles of \a e) and consider all of its vertices v. We check all triangles sharing v to see if any edges will turn into encriached segments when inserting \a i. If this is the case, move \a i so that it lies on the segment in question. */
//##############################################################################

void Refine::TestEncroach(Connectivity * const connectivity,
			  const MeshParameter *meshParameter,
			  const int nRefine)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif
  
  nvtxEvent *nvtxEncroach = new nvtxEvent("TestEncroach", 2);

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();
  
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, (const void *)
				       devTestEncroach, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devTestEncroach<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd, pVcAdd,
       pTv, pTe, pEt, pVc, nVertex, Px, Py, nTriangle);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    for (int i = 0; i < nRefine; i++)
      TestEncroachSingle(i, pElementAdd, pVcAdd,
			 pTv, pTe, pEt, pVc, nVertex, Px, Py, nTriangle);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      
  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Kernel: devTestEncroach, # of elements: "
	    << nRefine << ", elapsed time: " << elapsedTime << std::endl;

  std::ofstream outfile;
  outfile.open("TestEncroach.txt", std::ios_base::app);
  outfile << nRefine << " " << elapsedTime << std::endl;
  outfile.close();
#endif

  delete nvtxEncroach;
}
  
}
