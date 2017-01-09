// -*-c++-*-
/*! \file delaunay_flipedge.cu
\brief Functions for flipping edges to make trianglation Delaunay*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./delaunay.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../../Common/profile.h"

namespace astrix {
  
//#########################################################################
/*! \brief Flip edge \a pEdgeNonDelaunay[i]

\param i Index in \a pEnd to flip
\param *pEnd Pointer to array containing edges to be flipped
\param *pTv Pointer triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh*/
//#########################################################################

__host__ __device__
void FlipSingleEdge(int i, int *pEnd, int3 *pTv, int3 *pTe,
		    int2 *pEt, int nVertex)
{
  // Edge to be flipped
  int edge = pEnd[i];

  // Neighbouring triangles
  int t1 = pEt[edge].x;
  int t2 = pEt[edge].y;
  
  int d = pTv[t1].z;
  int e = pTv[t1].x;
  int f = pTv[t1].y;
  int c1 = 1;
  int e1 = pTe[t1].x;
  int e2 = pTe[t1].y;
  int e3 = pTe[t1].z;
  
  if (edge == e2) {
    d = pTv[t1].x;
    e = pTv[t1].y;
    f = pTv[t1].z;
    c1 = 2;
  }
  if (edge == e3) {
    d = pTv[t1].y;
    e = pTv[t1].z;
    f = pTv[t1].x;
    c1 = 3;
  }
    
  int a = pTv[t2].z;
  int b = pTv[t2].x;
  int c = pTv[t2].y;
  int c2 = 1;
  int e4 = pTe[t2].x;
  int e5 = pTe[t2].y;
  int e6 = pTe[t2].z;
  if (edge == e5) {
    a = pTv[t2].x;
    b = pTv[t2].y;
    c = pTv[t2].z;
    c2 = 2;
  }
  if (edge == e6) {
    a = pTv[t2].y;
    b = pTv[t2].z;
    c = pTv[t2].x;
    c2 = 3;
  }
  
  int makeValidFlag = 1;

  // PERIODIC
  if (abs(b - f) == abs(c - e)) {
    a -= (b - f);
    d += (b - f);
  } else {
#ifndef __CUDA_ARCH__
    std::cout << "Not sure what to do with edge " << edge << std::endl;
    int qq; std::cin >> qq;
#endif
  }
  
  if (c1 == 1) pTv[t1].x = a;
  if (c1 == 2) pTv[t1].y = a;
  if (c1 == 3) pTv[t1].z = a;

  if (c2 == 1) pTv[t2].x = d;
  if (c2 == 2) pTv[t2].y = d;
  if (c2 == 3) pTv[t2].z = d;

  if (makeValidFlag == 1) {
    MakeValidIndices(pTv[t1].x, pTv[t1].y, pTv[t1].z, nVertex);
    MakeValidIndices(pTv[t2].x, pTv[t2].y, pTv[t2].z, nVertex);
  }
      
  // Triangle edges
  if (c1 == 1) {
    if (c2 == 1) pTe[t1].x = e6;
    if (c2 == 2) pTe[t1].x = e4;
    if (c2 == 3) pTe[t1].x = e5;
    pTe[t1].z = edge;
  }
  if (c1 == 2) {
    if (c2 == 1) pTe[t1].y = e6;
    if (c2 == 2) pTe[t1].y = e4;
    if (c2 == 3) pTe[t1].y = e5;
    pTe[t1].x = edge;
  }
  if (c1 == 3) {
    pTe[t1].y = edge;
    if (c2 == 1) pTe[t1].z = e6;
    if (c2 == 2) pTe[t1].z = e4;
    if (c2 == 3) pTe[t1].z = e5;
  }
  
  if (c2 == 1) {
    if (c1 == 1) pTe[t2].x = e3;
    if (c1 == 2) pTe[t2].x = e1;
    if (c1 == 3) pTe[t2].x = e2;
    pTe[t2].z = edge;
  }
  if (c2 == 2) {
    pTe[t2].x = edge;
    if (c1 == 1) pTe[t2].y = e3;
    if (c1 == 2) pTe[t2].y = e1;
    if (c1 == 3) pTe[t2].y = e2;
  }
  if (c2 == 3) {
    pTe[t2].y = edge;
    if (c1 == 1) pTe[t2].z = e3;
    if (c1 == 2) pTe[t2].z = e1;
    if (c1 == 3) pTe[t2].z = e2;
  }
}
  
//#########################################################################
/*! \brief Kernel flipping edges in \a pEnd[i]

\param nNonDel Total number of edges to flip
\param *pEnd Pointer to array containing edges to be flipped
\param *pTv Pointer triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh*/
//#########################################################################

__global__ void
devFlipEdge(int nNonDel, int *pEnd,
	    int3 *pTv, int3 *pTe, int2 *pEt, int nVertex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (i < nNonDel) {
    FlipSingleEdge(i, pEnd, pTv, pTe, pEt, nVertex);

    i += gridDim.x*blockDim.x;
  }
}
  
//#########################################################################
/*! Flip all edges contained in the first \a nNonDel entries of Array \a edgeNonDelaunay

\param *connectivity Pointer to basic Mesh data
\param nNonDel Number of edges to be flipped*/
//#########################################################################

void Delaunay::FlipEdge(Connectivity * const connectivity,
			const int nNonDel)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

  int nVertex = connectivity->vertexCoordinates->GetSize();
  
  int *pEnd = edgeNonDelaunay->GetPointer();
  
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();  
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128; 

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlipEdge, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devFlipEdge<<<nBlocks, nThreads>>>
      (nNonDel, pEnd, pTv, pTe, pEt, nVertex);
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
      FlipSingleEdge(i, pEnd, pTv, pTe, pEt, nVertex);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
  }

#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  WriteProfileFile("FlipEdge.prof", nNonDel, elapsedTime, cudaFlag);
#endif

}

}
