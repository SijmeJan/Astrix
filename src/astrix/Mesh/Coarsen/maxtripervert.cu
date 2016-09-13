// -*-c++-*-
/*! \file maxtripervert.cu
\brief Functions for finding maximum number of triangles sharing a vertex
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
/*! \brief Consider triangle \a n and add 1 to \a *pTriangleVertex for all its vertices

Consider triangle \a n and add 1 to \a *pTriangleVertex for all its vertices using atomic operations. 

\param n Index of triangle to consider
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *pTrianglePerVertex Pointer to output array; after considering all triangles it contains the number of triangles per vertex for all vertices */
//#########################################################################

__host__ __device__
void FillTrianglePerVertexSingle(int n, int3 *pTv,
				 int nVertex, int *pTrianglePerVertex)
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
  
  AtomicAdd(&(pTrianglePerVertex[a]), 1);
  AtomicAdd(&(pTrianglePerVertex[b]), 1);
  AtomicAdd(&(pTrianglePerVertex[c]), 1);
}
  
//#########################################################################
/*! \brief Kernel counting number of triangles sharing every vertex

For every triangle in Mesh, add 1 to \a *pTriangleVertex for all its vertices using atomic operations. 

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param nVertex Total number of vertices in Mesh
\param *pTrianglePerVertex Pointer to output array; after considering all triangles it contains the number of triangles per vertex for all vertices */
//#########################################################################

__global__
void devFillTrianglePerVertex(int nTriangle, int3 *pTv,
			      int nVertex, int *pTrianglePerVertex)
{     
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    FillTrianglePerVertexSingle(n, pTv, nVertex, pTrianglePerVertex);
 
    n += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
/*! Returns the maximum number of triangles sharing any single vertex. An Array \a trianglePerVertex of size nVertex is created, which is then filled with the number of triangles sharing every vertex. The maximum value of this Array is then returned.*/
//#########################################################################

int Coarsen::MaxTriPerVert(Connectivity *connectivity)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      cudaFlag = 0;
    } else {
      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  Array<int> *trianglePerVertex =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  trianglePerVertex->SetToValue(0);
  int *pTrianglePerVertex = trianglePerVertex->GetPointer();

  int3 *pTv = connectivity->triangleVertices->GetPointer();

  // Count number of triangles for every vertex
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillTrianglePerVertex, 
				       (size_t) 0, 0);

    devFillTrianglePerVertex<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex, pTrianglePerVertex);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      FillTrianglePerVertexSingle(n, pTv, nVertex, pTrianglePerVertex);
  }
  
  int maxTriPerVert = trianglePerVertex->Maximum();
  
  delete trianglePerVertex;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      cudaFlag = 0;
    } else {
      cudaFlag = 1;
    }
  }

  return maxTriPerVert;
}

}
