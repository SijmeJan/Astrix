// -*-c++-*-
/*! \file target.cu
\brief Functions for finding target triangles for vertex removal.
*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! \brief Find a target triangle containing a vertex to safely move \a vRemove onto in order to remove it

After we have found all possible target triangles (in \a *tAllowed) we now select one. If \a vRemove is part of a segment, we have to move along the segment, otherwise we choose the first allowed target triangle.

\param vRemove Vertex to be removed
\param *vTri Pointer to list of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param *pTriangleWantRefine Pointer to array of flags indicating if triangles need refining (=1) or coarsening (=-1). If we can not find a suitable target triangle, we set an entry to zero so that \a vRemove will not be removed 
\param *tAllowed Pointer to array containing allowed target triangles*/
//#########################################################################

__host__ __device__
int FindTargetTriangle(int vRemove, int *vTri, int maxTriPerVert,
		       int3 *pTv, int3 *pTe, int2 *pEt, int nVertex,
		       int *pTriangleWantRefine, int *tAllowed)
{
  int t1 = -1;

  int segmentFlag = 0;
  
  // If vRemove part of segment, move along segment
  for (int i = 0; i < maxTriPerVert; i++) {
    int t = vTri[i];

    if (t != -1) {
      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;
      while (a >= nVertex) a -= nVertex;
      while (b >= nVertex) b -= nVertex;
      while (c >= nVertex) c -= nVertex;
      while (a < 0) a += nVertex;
      while (b < 0) b += nVertex;
      while (c < 0) c += nVertex;

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;
      
      // If t has segment containing vRemove, select t
      if (((t11 == -1 || t21 == -1) &&
	   (a == vRemove || b == vRemove)) ||
	  ((t12 == -1 || t22 == -1) &&
	   (b == vRemove || c == vRemove)) ||
	  ((t13 == -1 || t23 == -1) &&
	   (c == vRemove || a == vRemove))) {
	segmentFlag = 1;
	if (tAllowed[i] != 0) t1 = t;
      }
      
    }
  }

  // If vRemove not part of segment, choose first allowed triangle
  if (segmentFlag == 0) {
    int i = 0;
    while (tAllowed[i] == 0) {
      i++;
      if (i == maxTriPerVert) break;
    }

    if (i < maxTriPerVert) t1 = vTri[i];
  }

  // No triangle available
  if (t1 == -1)
    pTriangleWantRefine[vTri[0]] = 0;
  
  return t1;
}
  
//#########################################################################
/*! \brief Kernel finding a target triangle containing a vertex to safely move \a vRemove onto in order to remove it

After we have found all possible target triangles (in \a *pAllowed) we now select one. If \a vRemove is part of a segment, we have to move along the segment, otherwise we choose the first allowed target triangle.

\param pVertexRemove Pointer to array containing vertices to be removed
\param nRemove Total number of vertices to be removed
\param *pVertexTriangleList Pointer to list of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle 
\param *te1 Pointer to first edge of triangle 
\param *te2 Pointer to second edge of triangle 
\param *te3 Pointer to third edge of triangle 
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param *pTriangleWantRefine Pointer to array of flags indicating if triangles need refining (=1) or coarsening (=-1). If we can not find a suitable target triangle, we set an entry to zero so that \a vRemove will not be removed 
\param *pAllowed Pointer to array containing allowed target triangles
\param *pTriangleTarget Pointer to output array; will contain target triangle or -1 if no target triangle could be found*/
//#########################################################################

__global__
void devFindTargetTriangle(int *pVertexRemove, int nRemove,
			   int *pVertexTriangleList, int maxTriPerVert,
			   int3 *pTv, int3 *pTe, int2 *pEt, int nVertex, 
			   int *pTriangleWantRefine, int *pAllowed,
			   int *pTriangleTarget)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    pTriangleTarget[n] =
      FindTargetTriangle(pVertexRemove[n], 
			 &(pVertexTriangleList[n*maxTriPerVert]),
			 maxTriPerVert, pTv, pTe, pEt,
			 nVertex, pTriangleWantRefine,
			 &(pAllowed[n*maxTriPerVert]));
    
    n += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
/*! After we have found all possible target triangles (in \a *vertexTriangleAllowed) we now select one. If the vertex to be removed is part of a segment, we have to move along the segment, otherwise we choose the first allowed target triangle.

\param *triangleTarget Pointer to output Array; will contain target triangle or -1 if no target triangle could be found
\param *vertexTriangleAllowed Pointer to Array containing allowed target triangles 
\param *vertexTriangleList Pointer to Array of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex*/ 
//#########################################################################

void Coarsen::FindTargetTriangles(Connectivity *connectivity,
				  Array<int> *triangleWantRefine,
				  Array<int> *triangleTarget,
				  Array<int> *vertexTriangleAllowed,
				  Array<int> *vertexTriangleList,
				  int maxTriPerVert)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();
      vertexTriangleAllowed->TransformToHost();
      triangleWantRefine->TransformToHost();
      
      cudaFlag = 0;
    } else {
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();
      vertexTriangleAllowed->TransformToDevice();
      triangleWantRefine->TransformToDevice();
  
      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();

  int *pTriangleTarget = triangleTarget->GetPointer();
  int *pAllowed = vertexTriangleAllowed->GetPointer();
  int *pVertexTriangleList = vertexTriangleList->GetPointer();
  int *pVertexRemove = vertexRemove->GetPointer();
  int nRemove = vertexRemove->GetSize();
  
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  
  int *pTriangleWantRefine = triangleWantRefine->GetPointer();

  // Select target triangle
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFindTargetTriangle, 
				       (size_t) 0, 0);

    devFindTargetTriangle<<<nBlocks, nThreads>>>
      (pVertexRemove, nRemove, pVertexTriangleList,
       maxTriPerVert, pTv, pTe, pEt,
       nVertex, pTriangleWantRefine, pAllowed, pTriangleTarget);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      pTriangleTarget[n] =
	FindTargetTriangle(pVertexRemove[n], 
			   &(pVertexTriangleList[n*maxTriPerVert]),
			   maxTriPerVert, pTv, pTe, pEt,
			   nVertex, pTriangleWantRefine,
			   &(pAllowed[n*maxTriPerVert]));
  }
  
  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      triangleTarget->TransformToHost();
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();
      vertexTriangleAllowed->TransformToHost();
      triangleWantRefine->TransformToHost();
      
      cudaFlag = 0;
    } else {
      triangleTarget->TransformToDevice();
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();
      vertexTriangleAllowed->TransformToDevice();
      triangleWantRefine->TransformToDevice();
  
      cudaFlag = 1;
    }
  }

}

}
