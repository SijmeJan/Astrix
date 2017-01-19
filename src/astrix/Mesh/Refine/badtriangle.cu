// -*-c++-*-
/*! \file badtriangle.cu
\brief File containing function to test all triangles for refinement.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../../Common/inlineMath.h"
#include "../triangleLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"

namespace astrix {
  
//##############################################################################
/*! \brief Test if triangle \a i is low-quality

This kernel tests triangles if they need to be refined. Upon return, the array \a pTriangleList[i] contains \a i if triangle \a i is low-quality and can be refined, -1 otherwise. 

 \param i Triangle to be considered
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to vertex coordinates
 \param *pTriangleList Pointer to list of low-quality triangle (output).
 \param *pWantRefine pointer to flag whether triangle refinement is wanted if it is too large according to \a dMax (skinny triangles are always refined). If creating a new mesh, refinement is always wanted. If refining dynamically, \a *pWantRefine is determined by for example gradients in the density. 
 \param dMax Maximum size of circumscribed radius of triangle.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain x
 \param Py Periodic domain y
 \param qualityBound Maximum ratio of circumscribed circle radius squared to largest edge length squared.*/
//##############################################################################

__host__ __device__
void TestQualitySingleW(int i,
			int3 *pTv,
			const real2* __restrict__ pVc, 
			int *pTriangleList, int *pWantRefine,
			real dMax, int nVertex, real Px, real Py,
			real qualityBound)
{
  int ret = -1;
  int wantRefine = pWantRefine[i];
  
  int tooLarge = 0;
  int tooSkinny = 0;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
			 nVertex, Px, Py,
			 ax, bx, cx, ay, by, cy);
  
  // Edge lengths squared
  real la = Sq(bx - ax) + Sq(by - ay);
  real lb = Sq(cx - bx) + Sq(cy - by);
  real lc = Sq(cx - ax) + Sq(cy - ay);
  
  real det = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
  real invdet = (real)1.0/(det + (real)1.0e-30);
  
  // Circumscribed radius (squared)
  real r2 = (real)0.25*la*lb*lc*Sq(invdet);
  
  // Check if triangle too large or too skinny
  if (r2 > dMax && wantRefine == 1) tooLarge = 1;
  if (r2 > qualityBound*min(la, min(lb, lc))) tooSkinny = 1;
  
  if (tooLarge || tooSkinny) ret = i;
  
  // Flag triangle for refinement (coalesced write)
  pTriangleList[i] = ret;
}

//##############################################################################
/*! \brief Test if triangle \a i is low-quality

This kernel tests triangles if they need to be refined. Upon return, the array \a pTriangleList[i] contains \a i if triangle \a i is low-quality, -1 otherwise. 

 \param i Triangle to be considered
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to vertex coordinates
 \param *pTriangleList Pointer to list of low-quality triangle (output).
 \param dMax Maximum size of circumscribed radius of triangle.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain x
 \param Py Periodic domain y
 \param qualityBound Maximum ratio of circumscribed circle radius squared to largest edge length squared.*/
//##############################################################################

__host__ __device__
void TestQualitySingle(int i,
		       int3 *pTv,
		       const real2* __restrict__ pVc, 
		       int *pTriangleList, 
		       real dMax, int nVertex, real Px, real Py,
		       real qualityBound)
{
  int ret = -1;
  
  int tooLarge = 0;
  int tooSkinny = 0;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
			 nVertex, Px, Py,
			 ax, bx, cx, ay, by, cy);
  
  // Edge lengths squared
  real la = Sq(bx - ax) + Sq(by - ay);
  real lb = Sq(cx - bx) + Sq(cy - by);
  real lc = Sq(cx - ax) + Sq(cy - ay);
  
  real det = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
  real invdet = (real)1.0/(det + (real)1.0e-30);
  
  // Circumscribed radius (squared)
  real r2 = (real)0.25*la*lb*lc*Sq(invdet);
  
  // Check if triangle too large or too skinny
  if (r2 > dMax) tooLarge = 1;
  if (r2 > qualityBound*min(la, min(lb, lc))) tooSkinny = 1;
  
  if (tooLarge || tooSkinny) ret = i;
  
  // Flag triangle for refinement (coalesced write)
  pTriangleList[i] = ret;
}
  
//##############################################################################
/*! \brief Kernel testing triangles for refinement. 

This kernel tests triangles if they need to be refined. Upon return, the array \a pTriangleList[i] contains \a i if triangle \a i is low-quality and can be refined, -1 otherwise. 

 \param nTriangle total number of triangles in mesh.
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to vertex coordinates
 \param *pTriangleList Pointer to list of low-quality triangle (output).
 \param *pWantRefine pointer to flag whether triangle refinement is wanted if it is too large according to \a dMax (skinny triangles are always refined). If creating a new mesh, refinement is always wanted. If refining dynamically, \a *pWantRefine is determined by for example gradients in the density. 
 \param dMax Maximum size of circumscribed radius of triangle.
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain x
\param Py Periodic domain y
\param qualityBound Maximum ratio of circumscribed circle radius squared to largest edge length squared.*/
//##############################################################################

__global__ void
devTestQualityW(int nTriangle,
		int3 *pTv,
		const real2* __restrict__ pVc, 
		int *pTriangleList, int *pWantRefine, 
		real dMax, int nVertex, real Px, real Py, real qualityBound)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    TestQualitySingleW(i, pTv, pVc,
		       pTriangleList, pWantRefine,
		       dMax, nVertex, Px, Py, qualityBound);
    
    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//##############################################################################
/*! \brief Kernel testing triangles for refinement. 

This kernel tests triangles if they need to be refined. Upon return, the array \a pTriangleList[i] contains \a i if triangle \a i is low-quality, -1 otherwise. 

 \param nTriangle total number of triangles in mesh.
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to vertex coordinates
 \param *pTriangleList Pointer to list of low-quality triangle (output).
 \param dMax Maximum size of circumscribed radius of triangle.
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain x
 \param Py Periodic domain y
 \param qualityBound Maximum ratio of circumscribed circle radius squared to largest edge length squared.*/
//##############################################################################

__global__ void
devTestQuality(int nTriangle,
	       int3 *pTv,
	       const real2* __restrict__ pVc, 
	       int *pTriangleList,
	       real dMax, int nVertex, real Px, real Py, real qualityBound)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    TestQualitySingle(i, pTv, pVc, pTriangleList, 
		      dMax, nVertex, Px, Py, qualityBound);
    
    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//##############################################################################
/*! When refining a mesh, we first test all triangles if they need refining, which will lead to a list of vertices to add to the mesh. A triangle needs refining if it is either too large or too skinny (low-quality triangle), or if the corresponding entry in \a triangleWantRefine is set. The function returns \a nBad, the number of low-quality triangles. Upon return, the first \a nBad elements of the Array \a badTriangle contain the indices of low-quality triangles.

\param *connectivity Pointer to basic Mesh data
\param *meshParameter Pointer to Mesh parameters
\param *triangleWantRefine  Flags if triangles need to be refined because of resolution requirements (parameter will be ignored if 0)*/
//##############################################################################

int Refine::TestTrianglesQuality(Connectivity * const connectivity,
				 const MeshParameter *meshParameter,
				 const Array<int> *triangleWantRefine)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) ) ;
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  nvtxEvent *nvtxQuality = new nvtxEvent("TestQuality", 0);
  nvtxEvent *nvtxTemp = new nvtxEvent("Init", 1);

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  badTriangles->SetSize(nTriangle);
  int *pBadTriangles = badTriangles->GetPointer();
  delete nvtxTemp;

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  real dMax = meshParameter->baseResolution;
  real qualityBound = meshParameter->qualityBound;

  // Check if we can ignore triangleWantRefine
  if (triangleWantRefine != 0) {
    // Can not be ignored
    int *pWantRefine = triangleWantRefine->GetPointer();

    // Test all triangles for refinement
    if (cudaFlag == 1) {
      nvtxEvent *nvtxTemp = new nvtxEvent("CUDA", 2);
      
      int nBlocks = 128;
      int nThreads = 128;
      
      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devTestQualityW, 
					 (size_t) 0, 0);
      
      devTestQualityW<<<nBlocks, nThreads>>>
	(nTriangle, pTv, pVc,
	 pBadTriangles, pWantRefine, dMax,
	 nVertex, Px, Py, qualityBound);
      
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      
      delete nvtxTemp;
    } else {
      for (int i = 0; i < nTriangle; i++) 
	TestQualitySingleW(i, pTv, pVc,
			   pBadTriangles, pWantRefine, dMax,
			   nVertex, Px, Py, qualityBound);
    }
  } else {
    // Test for refinement, ignoring triangleWantRefine
    if (cudaFlag == 1) {
      nvtxEvent *nvtxTemp = new nvtxEvent("CUDA", 2);
      
      int nBlocks = 128;
      int nThreads = 128;
      
      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devTestQuality, 
					 (size_t) 0, 0);
      
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(start, 0) );
#endif
      devTestQuality<<<nBlocks, nThreads>>>
	(nTriangle, pTv, pVc, pBadTriangles,
	 dMax, nVertex, Px, Py, qualityBound);
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(stop, 0) );
      gpuErrchk( cudaEventSynchronize(stop) );
#endif      
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      
      delete nvtxTemp;
    } else {
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(start, 0) );
#endif
      for (int i = 0; i < nTriangle; i++) 
	TestQualitySingle(i, pTv, pVc, pBadTriangles, dMax,
			  nVertex, Px, Py, qualityBound);
#ifdef TIME_ASTRIX
      gpuErrchk( cudaEventRecord(stop, 0) );
      gpuErrchk( cudaEventSynchronize(stop) );
#endif
    }
  }
  
#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("TestQuality.prof", nTriangle, elapsedTime, cudaFlag);
#endif

  nvtxTemp = new nvtxEvent("Remove", 3);

  // The first nBad values in badTriangles are low-quality triangles
  int nBad = badTriangles->RemoveValue(-1);

  delete nvtxTemp;
  delete nvtxQuality;

#ifdef REPLACE_INT3
  delete newTv;
#endif
  
  return nBad;
}

}
