// -*-c++-*-
/*! \file circum.cu
\brief File containing function to find circumcentres of low-quality triangles.*/
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
/*! \brief Compute circumcentre of ith triangle in \a *pBt.

 \param i Entry in \a *pBt to consider
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to current vertex coordinates
 \param *pVcAdd Pointer to circumcentre coordinates (output)
 \param *pBt Pointer to list of bad triangles
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y*/
//##############################################################################

__host__ __device__
void CircumSingle(int i,
		  const int3* __restrict__ pTv,
		  const real2* __restrict__ pVc, 
		  real2 *pVcAdd, int *pBt, int nVertex, real Px, real Py)
{
  int t = pBt[i];
  
  int a = pTv[t].x;
  int b = pTv[t].y;
  int c = pTv[t].z;
  
  real ax, bx, cx, ay, by, cy;
  GetTriangleCoordinates(pVc, a, b, c,
			 nVertex, Px, Py,
			 ax, bx, cx, ay, by, cy);
  
  // Edge lengths squared
  real lb = Sq(cx - bx) + Sq(cy - by);
  real lc = Sq(cx - ax) + Sq(cy - ay);
  
  // Circumcentre coordinates
  real det1 = lc*(by - cy) - lb*(ay - cy);
  real det2 = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
  real invdet2 = (real)1.0/(det2 + (real)1.0e-30);
  real Cx = cx + (real)0.5*det1*invdet2;
  det1 = (ax - cx)*lb - (bx - cx)*lc;
  real Cy = cy + (real)0.5*det1*invdet2;
  
  // Add point in circumcentre (coalesced write)
  pVcAdd[i].x = Cx;
  pVcAdd[i].y = Cy;
}
  
//##############################################################################
/*! \brief Kernel computing circumcentres of bad triangles.

This kernel computes the circumcentre of the first \a nRefine entries in \a *pBt
, putting the result in \a *pVcAdd.

 \param nRefine Total number of low-quality triangles
 \param *pTv Pointer to triangle vertices
 \param *pVc Pointer to current vertex coordinates
 \param *pVcAdd Pointer to circumcentre coordinates (output)
 \param *pBt Pointer to list of bad triangles
 \param nVertex Total number of vertices in Mesh
 \param Px Periodic domain size x
 \param Py Periodic domain size y*/
//##############################################################################

__global__ void
devCircum(int nRefine,
	  const int3* __restrict__ pTv,
	  const real2* __restrict__ pVc, 
	  real2 *pVcAdd, int *pBt, int nVertex, real Px, real Py)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    CircumSingle(i, pTv, pVc, pVcAdd, pBt, nVertex, Px, Py);
    
    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}
    
//##############################################################################
/*! Once we have a list of low-quality triangles in \a badTriangles, we can compute the locations where we need to insert vertices. Upon return, \a vertexCoordinatesAdd contains the coordinates of the circumcentres of all \a nRefine low-quality triangle in \a badTriangles.

\param *connectivity Pointer to basic mesh data
\param *meshParameter Pointer to mesh parameters
\param nRefine Number of low-quality triangles*/
//##############################################################################

void Refine::FindCircum(Connectivity * const connectivity,
			const MeshParameter *meshParameter,
			const int nRefine)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  nvtxEvent *nvtxCircum = new nvtxEvent("Circum", 0);

  int nVertex = connectivity->vertexCoordinates->GetSize();
  
  vertexCoordinatesAdd->SetSize(nRefine);

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();

  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pBt = badTriangles->GetPointer();
  
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devCircum, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devCircum<<<nBlocks, nThreads>>>
      (nRefine, pTv, pVc, pVcAdd, pBt, nVertex, Px, Py);
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
    for (int i = 0; i < nRefine; i++) 
      CircumSingle(i, pTv, pVc, pVcAdd, pBt, nVertex, Px, Py);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif      
  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("Circum.txt", nRefine, elapsedTime, cudaFlag);
#endif

  delete nvtxCircum;
}

}
