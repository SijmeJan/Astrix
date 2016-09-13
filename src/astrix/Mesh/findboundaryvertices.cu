// -*-c++-*-
/*! \file findboundaryvertices.cu
\brief Functions to locate vertices at boundaries*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//######################################################################
/*! \brief Separate boundaries left/right/top/bottom

We know that if \a vertexBoundaryFlag < 0, the vertex lies on a boundary. Now we use \a vertexCoordinates together with \a minx, \a maxx, \a miny and \a maxy to determine whether is is the left/right/top/bottom boundary.

\param i Index of vertex to consider
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param *pVertexBoundaryFlag Pointer to array of boundary flags: +1 if on left boundary, +2 if on right boundary, +4 if on bottom boundary, +8 if on top bpoundary; i.e. 10 indicates a vertex both on top and right boundary*/
//######################################################################

__host__ __device__
void FillBoundaryFlagSingle(int i, real2 *pVc,
			    real minx, real miny, real maxx, real maxy,
			    int *pVertexBoundaryFlag)
{
  if (pVertexBoundaryFlag[i] < 0) {
    if (fabs(pVc[i].x - minx) < 1.0e-5f) pVertexBoundaryFlag[i] += 1;
    if (fabs(pVc[i].x - maxx) < 1.0e-5f) pVertexBoundaryFlag[i] += 2;
    if (fabs(pVc[i].y - miny) < 1.0e-5f) pVertexBoundaryFlag[i] += 4;
    if (fabs(pVc[i].y - maxy) < 1.0e-5f) pVertexBoundaryFlag[i] += 8;
    if (pVertexBoundaryFlag[i] >= 0) pVertexBoundaryFlag[i]++;
  }
}
  
//######################################################################
/*! \brief Kernel separating boundaries left/right/top/bottom

We know that if \a vertexBoundaryFlag < 0, the vertex lies on a boundary. Now we use \a vertexCoordinates together with \a minx, \a maxx, \a miny and \a maxy to determine whether is is the left/right/top/bottom boundary.

\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param *pVertexBoundaryFlag Pointer to array of boundary flags: +1 if on left boundary, +2 if on right boundary, +4 if on bottom boundary, +8 if on top bpoundary; i.e. 10 indicates a vertex both on top and right boundary*/
//######################################################################

__global__ void 
devFillBoundaryFlag(int nVertex, real2 *pVc,
		    real minx, real miny, real maxx, real maxy,
		    int *pVertexBoundaryFlag)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nVertex){
    FillBoundaryFlagSingle(n, pVc, minx, miny, maxx, maxy,
			   pVertexBoundaryFlag);
 
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! \brief Find vertices lying at boundaries

Consider triangle \a n. If any of its edges has only one triangle neighbour, it is a boundary, and we flag the vertices on that edge by setting \a vertexBoundaryFlag to -1

\param n Triangle to consider
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *pVertexBoundaryFlag Pointer to array of boundary flags: set to -1 if vertex on boundary*/
//######################################################################

__host__ __device__
void FindBoundariesSingle(int n, int3 *pTv, int nVertex, 
			  int2 *pEt, int3 *pTe, 
			  int *pVertexBoundaryFlag)
{
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  int e1 = pTe[n].x;
  int e2 = pTe[n].y;
  int e3 = pTe[n].z;

  int t11 = pEt[e1].x;
  int t21 = pEt[e1].y;
  int t12 = pEt[e2].x;
  int t22 = pEt[e2].y;
  int t13 = pEt[e3].x;
  int t23 = pEt[e3].y;
  
  if (t11 == -1 || t21 == -1) {
    pVertexBoundaryFlag[a] = -1;
    pVertexBoundaryFlag[b] = -1;
  }
  if (t12 == -1 || t22 == -1) {
    pVertexBoundaryFlag[b] = -1;
    pVertexBoundaryFlag[c] = -1;
  }
  if (t13 == -1 || t23 == -1) {
    pVertexBoundaryFlag[c] = -1;
    pVertexBoundaryFlag[a] = -1;
  }
}
  
//######################################################################
/*! \brief Find vertices lying at boundaries

Consider triangle \a n. If any of its edges has only one triangle neighbour, it is a boundary, and we flag the vertices on that edge by setting \a vertexBoundaryFlag to -1

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *pVertexBoundaryFlag Pointer to array of boundary flags: set to -1 if vertex on boundary*/
//######################################################################

__global__ void 
devFindBoundaries(int nTriangle, int3 *pTv, int nVertex, 
		  int2 *pEt, int3 *pTe, 
		  int *pVertexBoundaryFlag)
{
  // n = triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nTriangle){
    FindBoundariesSingle(n, pTv, nVertex, pEt, pTe, pVertexBoundaryFlag);
 
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Find vertices at boundaries; useful for setting boundary conditions. On return, \a vertexBoundaryFlag is -1 if vertex not on boundary, otherwise 0 and then +1 if on left boundary, +2 if on right boundary, +4 if on bottom boundary, +8 if on top bpoundary; i.e. 10 indicates a vertex both on top and right boundary*/
//######################################################################
  
void Mesh::FindBoundaryVertices()
{
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  
  vertexBoundaryFlag->SetSize(nVertex);
  vertexBoundaryFlag->SetToValue(0);
  int *pVertexBoundaryFlag = vertexBoundaryFlag->GetPointer();

  real minx = meshParameter->minx;
  real maxx = meshParameter->maxx;
  real miny = meshParameter->miny;
  real maxy = meshParameter->maxy;
  
  // Find boundaries
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFindBoundaries, 
				       (size_t) 0, 0);

    devFindBoundaries<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex, pEt, pTe, pVertexBoundaryFlag);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      FindBoundariesSingle(n, pTv, nVertex, pEt, pTe, pVertexBoundaryFlag);
  }

  // Fill boundary flags
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillBoundaryFlag, 
				       (size_t) 0, 0);

    devFillBoundaryFlag<<<nBlocks, nThreads>>>
      (nVertex, pVc, minx, miny, maxx, maxy,
       pVertexBoundaryFlag);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++) 
      FillBoundaryFlagSingle(i, pVc, minx, miny, maxx, maxy,
			     pVertexBoundaryFlag);
  }
}
  
}
