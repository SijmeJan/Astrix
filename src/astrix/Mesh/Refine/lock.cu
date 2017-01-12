// -*-c++-*-
/*! \file lock.cu
\brief File containing function lock triangles in cavities of insertion points.*/

#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Predicates/predicates.h"
#include "../Param/meshparameter.h"
#include "./../triangleLow.h"
#include "../../Common/atomic.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Lock the cavity of insertion point

Start at the insertion triangle and move in clockwise direction along the edge of the cavity, flagging all triangles as part of the cavity of \i by setting pTiC[t] = randomInt. If a triangle is associated with multiple cavities, pick the largest value of randomInt.  
 
\param VcAdd Coordinates of insertion point
\param elementAdd Insertion triangle or edge
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
\param randomInt Random integer associated with insertion point */
//#########################################################################

__host__ __device__
void LockTriangle(const real2 VcAdd,
		  const int elementAdd,
		  const int nTriangle,
		  int * const pTiC,
		  const int3 * const pTv,
		  const int3 * const pTe,
		  const int2 * const pEt,
		  const real2 * const pVc,
		  const int nVertex,
		  const real Px, const real Py,
		  const Predicates *pred,
		  const real * const pParam,
		  const int randomInt)
{ 
  real dx = VcAdd.x;
  real dy = VcAdd.y;

  real dxOld = dx;
  real dyOld = dy;
  // Flag if cavity lies across periodic boundary
  int translateFlag = 0;

  // Start at insertion triangle
  int tStart = elementAdd;
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
  
  while (finished == 0) {
    // Set pTiC to maximum of pTiC and pRandom
    AtomicMax(&(pTiC[t]), randomInt);

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
  
//#########################################################################
/*! \brief Kernel locking the cavities of insertion points

Upon return, pTiC[t] = pRandom[i] means that triangle \a t is part of the cavity of point \a i and available (i.e. not locked by another insertion point).

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
\param *pRandom Random integers associated with insertion points */
//#########################################################################

  __global__ void 
devLockTriangles(int nRefine, real2 *pVcAdd, int *pElementAdd, int nTriangle,
		 int *pTiC, int3 *pTv, int3 *pTe, int2 *pEt, real2 *pVc,
		 int nVertex, real Px, real Py, const Predicates *pred,
		 real *pParam, unsigned int *pRandom)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    LockTriangle(pVcAdd[i], pElementAdd[i], nTriangle, pTiC,
		 pTv, pTe, pEt, pVc, nVertex, Px, Py, pred,
		 pParam, pRandom[i]);
    
    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! \brief Lock triangles in cavities of insertion points

Upon return, pTriangleInCavity[n] = pRandomPermutation[i]: triangle n is part of cavity of insertion point i and available.    

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to Predicates object
\param *meshParameter Pointer to Mesh parameters
\param *triangleInCavity Pointer to output Array of size \a nTriangle*/
//#########################################################################
  
void Refine::LockTriangles(Connectivity * const connectivity,
			   const Predicates *predicates,
			   const MeshParameter *meshParameter,
			   Array<int> *triangleInCavity)
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

  // pTiC[n] = pRandom[i]: triangle n is part of cavity of new vertex i
  int *pTiC = triangleInCavity->GetPointer();
  triangleInCavity->SetToValue(-1);

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       (const void *) devLockTriangles, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devLockTriangles<<<nBlocks, nThreads>>>
      (nRefine, pVcAdd, pElementAdd, nTriangle, pTiC,
       pTv, pTe, pEt, pVc, nVertex, Px, Py, predicates,
       pParam, pRandomPermutation);
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
      LockTriangle(pVcAdd[n], pElementAdd[n], nTriangle, pTiC,
		   pTv, pTe, pEt, pVc, nVertex, Px, Py, predicates,
		   pParam, pRandomPermutation[n]);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      
  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("LockTriangle.prof", nRefine, elapsedTime, cudaFlag);
#endif
  
}

}
