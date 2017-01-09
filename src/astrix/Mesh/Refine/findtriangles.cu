// -*-c++-*-
/*! \file findtriangles.cu
\brief File containing function to find triangles containing point (x,y).*/

#include <iostream>
#include <stdexcept>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Find triangle to put (x,y) in

Walk through the mesh, starting at \a tStart, trying to find a triangle containing point (x, y). Returns the index of the triangle that the point (x, y) is located in. If the point is located exactly on an edge, the index of this edge is stored in \a edgeIndex and the function will return -1. 
 
\param tStart Triangle to start in
\param x X-coordinate of point to find triangle for
\param y Y-coordinate of point to find triangle for
\param nTriangle Total number of triangles in Mesh
\param edgeIndex If point lies exactly on edge, the index of the edge will be stored here
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param printFlag Flag whether to print triangle walk on screen (for debugging purposes)*/
//#########################################################################

__host__ __device__
int FindTriangle(int tStart, real& x, real& y, int nTriangle,
		 const real2* __restrict__ pVc,
		 const int3* __restrict__ pTv,
		 const int3* __restrict__ pTe,
		 const int2* __restrict__ pEt,
		 const Predicates *pred, real *pParam,
		 int nVertex, real Px, real Py,
		 int printFlag, int& nSteps)
{
  const real zero = (real) 0.0;
  
  int ret = -1;

  // Adjust starting triangle if necessary
  if (tStart < 0 || tStart >= nTriangle) tStart = 0;
  int t = tStart;
  int tPrev = -1;
  int finished = 0;
  int crossTimes = 0;
  // Vertices belonging to edge we are moving across
  int edgeV2 = -1;// = pTv[t].y;
  int edgeCrossed = -1;
  int edgeIndex = -1;
  nSteps = 0;
 
  while (finished == 0 && nSteps <= 2*nTriangle) {
    int tNext = t;

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;

    //if (nSteps == 0) edgeV2 = b;
    
    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
			   nVertex, Px, Py,
			   ax, bx, cx, ay, by, cy);

    int e1 = pTe[t].x;
    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    // PERIODIC
    // Translate (x, y) such that edge (V1, V2) is part of t
    if (edgeCrossed >= 0) {
      int v = -1;
      if (edgeCrossed == e1) v = a;
      if (edgeCrossed == e2) v = b;
      if (edgeCrossed == e3) v = c;
      
      TranslateVertexToVertex(v, edgeV2, Px, Py, nVertex, x, y);
    }
 
    
    real dx = x, dy = y;

    //real A1 = pred->orient2d(dx, dy, ax, ay, bx, by, pParam);
    //real A2 = pred->orient2d(dx, dy, bx, by, cx, cy, pParam);
    //real A3 = pred->orient2d(dx, dy, cx, cy, ax, ay, pParam);

    real detleft = (dx - bx) * (ay - by);
    real detright = (dy - by) * (ax - bx);
    real A1 = detleft - detright;
    detleft = (dx - cx) * (by - cy);
    detright = (dy - cy) * (bx - cx);
    real A2 = detleft - detright;
    detleft = (dx - ax) * (cy - ay);
    detright = (dy - ay) * (cx - ax);
    real A3 = detleft - detright;

    edgeCrossed = -1;
    
    if (A1 < zero) {
      edgeCrossed = e1;
      edgeV2 = b;
    }
    if (A2 < zero && A2 < A1) {
      edgeCrossed = e2;
      edgeV2 = c;
    }
    if (A3 < zero && A3 < A1 && A3 < A2) {
      edgeCrossed = e3;
      edgeV2 = a;
    }

    if (edgeCrossed != -1) {
      int t1 = pEt[edgeCrossed].x;
      int t2 = pEt[edgeCrossed].y;
      tNext = t1;
      if(tNext == t) tNext = t2;
    }
    
    // Keep track how many times we go back to triangle we came from
    if (tNext == tPrev) crossTimes++; else crossTimes = 0;
    
    // Not moving to new triangle: A1, A2, A3 >= 0
    if (tNext == t) {
      // Must have found edge if any of A's are zero
      if (A1 == zero) edgeIndex = e1;
      if (A2 == zero) edgeIndex = e2;
      if (A3 == zero) edgeIndex = e3;

      // Otherwise, we have found a triangle
      finished = 1;     
    }

    // Either moved across segment or back to previous triangle for 2nd time
    if (tNext == -1 || crossTimes > 1) {
      // Choose best edge (minimum A)
      if (A1 < A2 && A1 < A3) edgeIndex = e1;
      if (A2 < A1 && A2 < A3) edgeIndex = e2;
      if (A3 < A2 && A3 < A1) edgeIndex = e3;
      
      finished = 1;
    }

    tPrev = t;
    t = tNext;
    
    nSteps++;
  }

  // Triangle found
  if (finished == 1) 
    if (edgeIndex == -1) ret = t; else ret = edgeIndex + nTriangle;
  
  return ret;
}
    
//######################################################################
/*! \brief Kernel finding triangles to put \a nRefine points in

Walk through the mesh trying to find triangles containing points specified in \a refineX and \a refineY.   
 
\param nRefine Total number of points to find triangles for
\param nTriangle Total number of triangles in Mesh
\param *refineIndex Pointer to triangles spawning the points to be inserted. These triangles will serve as starting locations for the walk through the Mesh
\param *pTriangleAdd Pointer to output array containing triangles found, or -1 if no triangle could be found
\param *pEdgeAdd Pointer to output array containing edges found, or -1 if no edge could be found 
\param refineX X-coordinates of points to find triangles for
\param refineY Y-coordinates of points to find triangles for
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//######################################################################

__global__ void 
devFindTriangles(int nRefine, int nTriangle, 
		 int *refineIndex, int *pElementAdd, real2 *pVcAdd,
		 const real2* __restrict__ pVc,
		 const int3* __restrict__ pTv,
		 const int3* __restrict__ pTe,
		 const int2* __restrict__ pEt,
		 const Predicates *pred, real *pParam,
		 int nVertex, real Px, real Py)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    // Coordinates of point to be inserted
    real x = pVcAdd[i].x;
    real y = pVcAdd[i].y;
      
    int a = 0;
    
    // Find triangle or edge to place vertex
    int t = FindTriangle(refineIndex[i], x, y, nTriangle,
			 pVc, pTv, pTe, pEt,
			 pred, pParam,
			 nVertex, Px, Py, 0, a);

    pVcAdd[i].x = x;
    pVcAdd[i].y = y;
   
    pElementAdd[i] = t;
    
    i += gridDim.x*blockDim.x;
  }
}
  
//#########################################################################
/*! When we have created a list of points (x,y) to insert into the mesh in \a vertexCoordinatesAdd, we have to find triangles to put these points in. This is done by walking through the grid, starting from the triangle that initiated the point, until we have found either a suitable triangle or a suitable edge. Results are put in \a elementAdd

\param *connectivity Pointer to basic Mesh data
\param *meshParameter Pointer to mesh parameters
\param *predicates Exact geomentric predicates*/  
//#########################################################################

void Refine::FindTriangles(Connectivity * const connectivity,
			   const MeshParameter *meshParameter,
			   const Predicates *predicates)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  nvtxEvent *nvtxFind = new nvtxEvent("FindTriangles", 1);

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nRefine = elementAdd->GetSize();
  
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  int *pBadTriangles = badTriangles->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
   
  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  // Find trangles to put new vertices in
  if (cudaFlag == 1) {
    int nBlocks = 26;
    int nThreads = 512;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
    				       devFindTriangles, 
    				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devFindTriangles<<<nBlocks, nThreads>>>
      (nRefine, nTriangle, pBadTriangles,
       pElementAdd, pVcAdd,
       pVc, pTv, pTe, pEt,
       predicates, pParam,
       nVertex, Px, Py);
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
    for (int i = 0; i < nRefine; i++) {
      // Coordinates of point to be inserted
      real x = pVcAdd[i].x;
      real y = pVcAdd[i].y;

      int printFlag = 0;
      
      int nSteps = 0;

      // Find triangle or edge to place vertex
      int t = FindTriangle(pBadTriangles[i], x, y, nTriangle,
			   pVc, pTv, pTe, pEt,
			   predicates, pParam,
			   nVertex, Px, Py, printFlag,
			   nSteps);

      pVcAdd[i].x = x;
      pVcAdd[i].y = y;
      
      pElementAdd[i] = t;

      if (t == -1) {
	std::cout << std::endl 
		  << "Error in FindTriangles: no triangle or edge found!"
		  << std::endl
		  << "Vertex location: " << x << " " << y << std::endl
		  << "Starting triangle: " << pBadTriangles[i] << std::endl;
	throw std::runtime_error("");
	//int qq; std::cin >> qq;
      }
    }
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      

  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("FindTriangle.prof", nRefine, elapsedTime, cudaFlag);
#endif

  delete nvtxFind;
}

}
