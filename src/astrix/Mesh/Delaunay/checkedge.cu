// -*-c++-*-
/*! \file checkedge.cu
\brief Functions for checking edges in Mesh for Delaunay property*/
#include <iostream>
//#include <fstream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./delaunay.h"
#include "./../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"

namespace astrix {
    
//#########################################################################
/*! \brief Check edge \a i for Delaunay-hood

Check edge \a i and write result in \a eNonDel (1 if not Delaunay, 0 otherwise)

\param i Index of edge to check
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pEnd Pointer to list of edges that are not Delaunay (output)
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//#########################################################################

__host__ __device__
void CheckEdge(int i,
	       const real2* __restrict__ pVc,
	       const int3* __restrict__ pTv,
	       const int3* __restrict__ pTe,
	       const int2* __restrict__ pEt,
	       int *pEnd, const Predicates *pred, real *pParam,
	       int nVertex, real Px, real Py)
{
  // Assume edge is Delaunay
  int ret = -1;

  int t1 = pEt[i].x;
  int t2 = pEt[i].y;
  
  if (t1 != -1 && t2 != -1) {
    int a = pTv[t1].x;
    int b = pTv[t1].y;
    int c = pTv[t1].z;    

    int e1 = pTe[t1].x;
    int e2 = pTe[t1].y;
    int e3 = pTe[t1].z;

    int f =   (i == e1)*b +  (i == e2)*c +  (i == e3)*a;

    int d = (i == e1)*c + (i == e2)*a + (i == e3)*b;
    real dx, dy;
    GetTriangleCoordinatesSingle(pVc, d, nVertex, Px, Py, dx, dy);
    
    a = pTv[t2].x;
    b = pTv[t2].y;
    c = pTv[t2].z;
    
    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
			   nVertex, Px, Py, 
			   ax, bx, cx, ay, by, cy);

    // Going to test if d lies in circle of t2
    e1 = pTe[t2].x;
    e2 = pTe[t2].y;
    e3 = pTe[t2].z;
    
    b = (i == e1)*a + (i == e2)*b + (i == e3)*c;

    // Edge is between (e, c) and (f, b)
    
    // PERIODIC
    TranslateVertexToVertex(b, f, Px, Py, nVertex, dx, dy);

    real detNew = pred->incircle(ax, ay, bx, by, cx, cy, dx, dy, pParam);

    // Edge is not Delaunay
    if (detNew > (real) 0.0) ret = i;
  }

  pEnd[i] = ret;
}
  
//######################################################################
/*! \brief Kernel checking edges for Delaunay-hood

Check edges and write result in \a pEnd (1 if not Delaunay, 0 otherwise)

\param nEdge Total number of edges in Mesh
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param *pEnd Pointer to list of edges that are not Delaunay (output)
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//######################################################################

__global__ void
devCheckEdge(int nEdge,
	     const real2* __restrict__ pVc,
	     const int3* __restrict__ pTv,
	     const int3* __restrict__ pTe,
	     const int2* __restrict__ pEt,
	     int *pEnd, const Predicates *pred, real *pParam,
	     int nVertex, real Px, real Py)
{
  // i = edge number
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    CheckEdge(i, pVc, pTv, pTe, pEt, pEnd, pred, pParam, nVertex, Px, Py);

    // Next edge
    i += blockDim.x*gridDim.x;
  }
}

__global__ void
devCheckEdgeLimit(int nEdgeCheck,
		  const int *pEnC,
		  const real2* __restrict__ pVc,
		  const int3* __restrict__ pTv,
		  const int3* __restrict__ pTe,
		  const int2* __restrict__ pEt,
		  int *pEnd, const Predicates *pred, real *pParam,
		  int nVertex, real Px, real Py)
{
  // i = edge number
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdgeCheck) {
    int e = pEnC[i];
    CheckEdge(e, pVc, pTv, pTe, pEt, pEnd, pred, pParam, nVertex, Px, Py);

    // Next edge
    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Check edges for Delaunay-hood. Result is written in \a edgeNonDelaunay (-1 if Delaunay)

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to exact geometric predicates
\param *meshParameter Pointer to Mesh parameters*/
//#########################################################################

void Delaunay::CheckEdges(Connectivity * const connectivity,
			  const Predicates *predicates,
			  const MeshParameter *meshParameter,
			  Array<int> * const edgeNeedsChecking,
			  const int nEdgeCheck)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();
  
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
 
  int *pEnd = edgeNonDelaunay->GetPointer();
  
  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (edgeNeedsChecking == 0) {
    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;
      
      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devCheckEdge, 
					 (size_t) 0, 0);
      
#ifdef TIME_ASTRIX
      cudaEventRecord(start, 0);
#endif
      
      devCheckEdge<<<nBlocks, nThreads>>>
	(nEdge, pVc, pTv, pTe, pEt, pEnd, predicates, pParam, nVertex, Px, Py);
      
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

      for (int i = 0; i < nEdge; i++) 
	CheckEdge(i, pVc, pTv, pTe, pEt, pEnd, predicates,
		  pParam, nVertex, Px, Py);

      // Make structured mesh less uniform
      if (meshParameter->structuredFlag == 2) {
	real Px = meshParameter->maxx - meshParameter->minx;
	real Py = meshParameter->maxy - meshParameter->miny;
      
	int nx = (int) (sqrt(0.565/meshParameter->baseResolution)*Px) + 4;
	int ny = (int)(nx*Py/Px);
	for (int i = 0; i < nx - 1; i += 2) {
	  for (int j = 0; j < ny - 1; j += 3) {
	    int v = j*(nx - meshParameter->periodicFlagX) + i;
	    int e = 3*(v - j) + j + 1 + 2*j*meshParameter->periodicFlagX -
	      (i > 0)*meshParameter->periodicFlagX;

	    pEnd[e] = e;
	  }
	}
      }
      

#ifdef TIME_ASTRIX
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
#endif
    }
  } else {
    int *pEnC = edgeNeedsChecking->GetPointer();
    edgeNonDelaunay->SetToValue(-1);
    
    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;
      
      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devCheckEdgeLimit, 
					 (size_t) 0, 0);
      
#ifdef TIME_ASTRIX
      cudaEventRecord(start, 0);
#endif
      
      devCheckEdgeLimit<<<nBlocks, nThreads>>>
	(nEdgeCheck, pEnC, pVc, pTv, pTe, pEt, pEnd,
	 predicates, pParam, nVertex, Px, Py);
      
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

      for (int i = 0; i < nEdgeCheck; i++) 
	CheckEdge(pEnC[i], pVc, pTv, pTe, pEt, pEnd, predicates,
		  pParam, nVertex, Px, Py);
      
#ifdef TIME_ASTRIX
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
#endif
    }
  }
   
#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("CheckEdge.txt", nEdgeCheck, elapsedTime, cudaFlag);
#endif

}
  
}
