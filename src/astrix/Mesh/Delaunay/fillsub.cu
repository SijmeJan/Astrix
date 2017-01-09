// -*-c++-*-
/*! \file fillsub.cu
\brief Functions for determining substitution triangles for repairing edges in Mesh after flipping*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./delaunay.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../../Common/profile.h"

namespace astrix {
  
//#########################################################################
/*! \brief Fill triangle substitution Array

  When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles

\param i Index in \a *pEnd to consider
\param *pEnd Pointer to list of edges that need to be flipped
\param *pTsub Pointer to output array containing substitution triangles
\param *pEt Pointer edge triangles*/
//#########################################################################

__host__ __device__
void FillTriangleSubstituteSingle(int i, int *pEnd, int *pTsub, int2 *pEt)
{
  int e = pEnd[i];
  int t1 = pEt[e].x;
  int t2 = pEt[e].y;

#ifndef __CUDA_ARCH__
  if (t1 < 0 || t2 < 0) {
    std::cout << "Error: trying to flip edge " << e
	      << ", which does not have two triangles "
	      << t1 << " " << t2 << std::endl;
    int qq; std::cin >> qq;
  }
#endif
    
  pTsub[t1] = t2;
  pTsub[t2] = t1;
}
  
//#########################################################################
/*! \brief Kernel filling triangle substitution Array

  When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles

\param nNonDel Number of edges to be flipped
\param *pEnd Pointer to list of edges that need to be flipped
\param *pTsub Pointer to output array containing substitution triangles
\param *pEt Pointer edge triangles*/
//#########################################################################

__global__ void
devFillTriangleSubstitute(int nNonDel, int *pEnd, int *pTsub, int2 *pEt)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (i < nNonDel) {
    FillTriangleSubstituteSingle(i, pEnd, pTsub, pEt);

    i += gridDim.x*blockDim.x;
  }
}
  
//#########################################################################
/*! When flipping an edge, the Array \a edgeTriangles can become corrupted. Fortunately, this is easy to correct for; we just need to remember the original neighbouring triangles. These triangles are stored in the Array \a triangleSubstitute

\param *connectivity Pointer to basic Mesh data
\param nNonDel Number of edges to be flipped*/
//#########################################################################

void Delaunay::FillTriangleSubstitute(Connectivity * const connectivity,
				      const int nNonDel)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif
  
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  
  // Fill triangle substitution array
  int *pEnd = edgeNonDelaunay->GetPointer();
  
  int *pTsub = triangleSubstitute->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillTriangleSubstitute,
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devFillTriangleSubstitute<<<nBlocks, nThreads>>>
      (nNonDel, pEnd, pTsub, pEt);
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
    for (int i = 0; i < nNonDel; i++) 
      FillTriangleSubstituteSingle(i, pEnd, pTsub, pEt);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
  }  

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("FillSub.prof", nNonDel, elapsedTime, cudaFlag);
#endif

}

}
