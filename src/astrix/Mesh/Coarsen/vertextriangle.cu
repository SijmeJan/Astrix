// -*-c++-*-
/*! \file vertextriangle.cu
\brief Functions for finding triangles containing certain vertices.
*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../../Common/atomic.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! \brief Set triangle \a n as the triangle for its vertices

  For every vertex we want to know one triangle sharing it and put the result in \a *pVertexTriangle. Here, we consider triangle \a n and set it as the \a vertexTriangle for its vertices using atomic operations.

\param n Index of triangle to consider
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *pVertexTriangle Pointer to output array*/
//#########################################################################

__host__ __device__
void FillVertexTriangleSingle(int n, int3 *pTv,
			      int nVertex, int *pVertexTriangle)
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

  AtomicExch(&(pVertexTriangle[a]), n);
  AtomicExch(&(pVertexTriangle[b]), n);
  AtomicExch(&(pVertexTriangle[c]), n);
}
    
//#########################################################################
/*! \brief Kernel setting \a vertexTriangle for all vertices

  For every vertex we want to know one triangle sharing it and put the result in \a *pVertexTriangle. Here, we loop through all triangles and set it as the \a vertexTriangle for its vertices using atomic operations. Then \a pVertexTriangle will contain the triangle that has last written to it. 

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *pVertexTriangle Pointer to output array*/
//#########################################################################

__global__
void devFillVertexTriangle(int nTriangle, int3 *pTv,
			   int nVertex, int *pVertexTriangle)
{     
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    FillVertexTriangleSingle(n, pTv, nVertex, pVertexTriangle);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! For every vertex we want to know one triangle sharing it, and put the result in \a vertexTriangle. Here, we loop through all triangles and set it as the \a vertexTriangle for its vertices using atomic operations. Then \a vertexTriangle will contain the triangle that has last written to it.*/
//#########################################################################

void Coarsen::FillVertexTriangle(Connectivity *connectivity)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangle->TransformToHost();
      
      cudaFlag = 0;
    } else {
      vertexTriangle->TransformToDevice();
  
      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  
  int *pVertexTriangle = vertexTriangle->GetPointer();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillVertexTriangle, 
				       (size_t) 0, 0);

    devFillVertexTriangle<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex, pVertexTriangle);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      FillVertexTriangleSingle(n, pTv, nVertex, pVertexTriangle);
  }
  
  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangle->TransformToHost();
      
      cudaFlag = 0;
    } else {
      vertexTriangle->TransformToDevice();
  
      cudaFlag = 1;
    }
  }

}

}
