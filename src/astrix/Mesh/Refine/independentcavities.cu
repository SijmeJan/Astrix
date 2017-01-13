// -*-c++-*-
/*! \file independentcavities.cu
\brief File containing function to find independent cavities.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Predicates/predicates.h"
#include "../Param/meshparameter.h"
#include "./../triangleLow.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Check whether all triangles in cavity \i are available

Start at the insertion triangle and move in clockwise direction along the edge of the cavity, checking if all triangles are available (i.e. not locked by other insertion point). If all are available, set pUniqueFlag[i] = 1, otherwise pUniqueFlag[i] = 0.  

\param i Insertion point to consider
\param *pVcAdd Coordinates of insertion points
\param *pElementAdd Insertion triangles or edges
\param nTriangle Total number of triangles in Mesh
\param *pTiC Pointer to Array triangleInCavity (output)
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVc Pointer to vertex coordinates
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to Predicates object
\param *pParam Pointer to parameter vector associated with Predicates
\param *pRandom Random integers associated with insertion points 
\param *pUniqueFlag Pointer to array flagging whether point can be inserted in parallel*/
//#########################################################################

__host__ __device__
void FindIndependentCavity(int i, real2 *pVcAdd, int *pElementAdd,
			   int nTriangle, int *pTiC,
			   int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
			   //const int3* __restrict__ pTv,
			   //const int3* __restrict__ pTe,
			   //const int2* __restrict__ pEt,
			   //const real2* __restrict__ pVc,
			   int nVertex, real Px,
			   real Py, const Predicates *pred, real *pParam,
			   unsigned int *pRandom, int *pUniqueFlag)
{
  real dx = pVcAdd[i].x;
  real dy = pVcAdd[i].y;

  real dxOld = dx;
  real dyOld = dy;
  // Flag if cavity lies across periodic boundary
  int translateFlag = 0;

  int randomInt = (int) pRandom[i];
  
  // Start at insertion triangle
  int tStart = pElementAdd[i];
  if (tStart >= nTriangle) {
    // Inserting on edge
    int e = tStart - nTriangle;
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;
    
    tStart = t1;
    if (t2 > t1) tStart = t2;
  }

  int t = tStart;

  // Choose starting edge to have two neigbours
  int e[] = {pTe[t].x, pTe[t].y, pTe[t].z};
  int eStart = -1;
  for (int n = 0; n < 3; n++) 
    if (pEt[e[n]].x != -1 && pEt[e[n]].y != -1) eStart = e[n];
    
  int eCrossed = eStart;
  int finished = 0;  
  int ret = 1;
  
  while (finished == 0) {
    // We know t is in cavity: if pTiC == -i - 2, we have already encountered
    // t and we can move on; if pTiC == pRandom[i] it is available, otherwise
    // it is needed by a point with higher priority.
    if (pTiC[t] != -i - 2) {
      if (pTiC[t] != randomInt) {
	ret = 0;
	// We might as well stop
	finished = 1;
      } else {
	pTiC[t] = -i - 2;
      }
    }

    if (finished == 0) {
      
    int tNext = -1;
    int eNext = -1;
    
    int e1 = -1, e2 = -1, e3 = -1;

    for (int de = 2; de >= 0; de--) {
      // Choose eNext clockwise from eCrossed
      if (eCrossed == e[0]) eNext = e[de % 3];
      if (eCrossed == e[1]) eNext = e[(1 + de) % 3];
      if (eCrossed == e[2]) eNext = e[(2 + de) % 3];

      // Triangle sharing eNext
      int2 tNeighbour = pEt[eNext];
      tNext = tNeighbour.x;
      if (tNext == t) tNext = tNeighbour.y;

      if (tNext != -1) {
	// Check if vertex n lies in circumcircle of triangle tNext
	int a = pTv[tNext].x;
	int b = pTv[tNext].y;
	int c = pTv[tNext].z;

	real ax, bx, cx, ay, by, cy;
	GetTriangleCoordinates(pVc, a, b, c, nVertex, Px, Py,
			       ax, bx, cx, ay, by, cy);

	e1 = pTe[tNext].x;
	e2 = pTe[tNext].y;
	e3 = pTe[tNext].z;

	// Translate (dx, dy) so that it lies on the same side as tNext
	int f = a;
	if (e2 == eNext) f = b;
	if (e3 == eNext) f = c;
	
	int A = pTv[t].x;
	int B = pTv[t].y;
	int C = pTv[t].z;

	
	int F = B;
	if (e[1] == eNext) F = C;
	if (e[2] == eNext) F = A;

	real dxNew = dx;
	real dyNew = dy;
	
	// Indicate that cavity lies across periodic boundary
	if (f != F) translateFlag = 1;
	TranslateVertexToVertex(f, F, Px, Py, nVertex, dxNew, dyNew);

	real det = pred->incircle(ax, ay, bx, by, cx, cy, dxNew, dyNew, pParam);

	// Check if triangle is part of cavity if we translate triangle
	// in stead of vertex
	real det2 = det;
	// Do this only when cavity lies across periodic boundary
	if (translateFlag == 1) {
	  real DeltaX = dxOld - dxNew;
	  real DeltaY = dyOld - dyNew;

	  det2 = pred->incircle(ax + DeltaX, ay + DeltaY,
				bx + DeltaX, by + DeltaY,
				cx + DeltaX, cy + DeltaY,
				dxOld, dyOld, pParam);
	}
	
	// If triangle not part of cavity, do not move into it
	if (det < (real) 0.0 && det2 < (real) 0.0) {
	  tNext = -1;
	} else {
	  // Move into tNext; use new coordinates
	  dx = dxNew;
	  dy = dyNew;
	}
      }

      // Done if trying to move across eStart but failing
      if (eNext == eStart && tNext == -1) {
	finished = 1;
	break;
      }

      // Found a triangle to move into
      if (tNext != -1) break;
    }
    
    eCrossed = eNext;
    t = tNext;
    e[0] = e1;
    e[1] = e2;
    e[2] = e3;

    // Done if we are moving back into tStart across eStart or if no
    // next triangle can be found
    if ((t == tStart && eCrossed == eStart) || t == -1) finished = 1;
    }
  }

  pUniqueFlag[i] = ret;
}

//#########################################################################
/*! \brief Kernel finding points that can be inserted in parallel

Upon return, pUniqueFlag[i] = 1 is point can be inserted independently of all others, otherwise pUniqueFlag[i] = 0

\param nRefine Total number of insertion points
\param *pVcAdd Coordinates of insertion points
\param *pElementAdd Insertion triangles or edges
\param nTriangle Total number of triangles in Mesh
\param *pTiC Pointer to Array triangleInCavity (output)
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pVc Pointer to vertex coordinates
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to Predicates object
\param *pParam Pointer to parameter vector associated with Predicates
\param *pRandom Random integers associated with insertion points 
\param *pUniqueFlag Pointer to array flagging whether point can be inserted in parallel (output)*/
//#########################################################################

__global__ void 
devFindIndependentCavities(int nRefine, real2 *pVcAdd, int *pElementAdd,
			   int nTriangle, int *pTiC,
			   int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
			   //const int3* __restrict__ pTv,
			   //const int3* __restrict__ pTe,
			   //const int2* __restrict__ pEt,
			   //const real2* __restrict__ pVc,
			   int nVertex, real Px, real Py,
			   const Predicates *pred, real *pParam,
			   unsigned int *pRandom, int *pUniqueFlag)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    FindIndependentCavity(i, pVcAdd, pElementAdd, nTriangle,
			  pTiC, pTv, pTe, pEt, pVc, nVertex, Px, Py,
			  pred, pParam, pRandom, pUniqueFlag);

    i += gridDim.x*blockDim.x;
  }
}
  
//#########################################################################
/*! \brief Find independent set of cavities

Upon return, pUniqueFlag[i] = 1 is point can be inserted independently of all others, otherwise pUniqueFlag[i] = 0.

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to Predicates object
\param *meshParameter Pointer to Mesh parameters
\param *triangleInCavity pTriangleInCavity[n] = pRandomPermutation[i]: triangle n is part of cavity of insertion point i and available.  
\param *uniqueFlag Upon return, pUniqueFlag[i] = 1 is point can be inserted independently of all others, otherwise pUniqueFlag[i] = 0.  
*/
//#########################################################################
  
void Refine::FindIndependentCavities(Connectivity * const connectivity,
				     const Predicates *predicates,
				     const MeshParameter *meshParameter,
				     Array<int> * const triangleInCavity,
				     Array<int> *uniqueFlag)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  // Number of triangles and number of insertion points
  unsigned int nTriangle = connectivity->triangleVertices->GetSize();
  unsigned int nRefine = elementAdd->GetSize();

  int *pTiC = triangleInCavity->GetPointer();
  int *pUniqueFlag = uniqueFlag->GetPointer();

  // Shuffle points to add to maximise parallelisation
  unsigned int *pRandomPermutation = randomUnique->GetPointer();
  
  unsigned int nVertex = connectivity->vertexCoordinates->GetSize();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();

  // Find set of independent cavities
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       (const void *) 
				       devFindIndependentCavities, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devFindIndependentCavities<<<nBlocks, nThreads>>>
      (nRefine, pVcAdd, pElementAdd, nTriangle, pTiC,
       pTv, pTe, pEt, pVc, nVertex, Px, Py, predicates,
       pParam, pRandomPermutation, pUniqueFlag);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    for (int n = 0; n < (int) nRefine; n++) 
      FindIndependentCavity(n, pVcAdd, pElementAdd, nTriangle,
			    pTiC, pTv, pTe, pEt, pVc, nVertex, Px, Py,
			    predicates, pParam, pRandomPermutation,
			    pUniqueFlag);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      
  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("IndependentCavities.prof", nRefine, elapsedTime, cudaFlag);
#endif

}

}
