// -*-c++-*-
/*! \file flagvertex.cu
\brief Functions for selecting vertices for removal.
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
/*! \brief Select vertices for removal

If triangle \a i is not flagged as wanting coarsening, none of its vertices can be removed. In this case, \a vertexRemoveFlag is set to zero for these vertices using atomic operations.

\param i Triangle under consideration
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param nVertex Total number of vertices in Mesh
\param *pTriangleWantRefine Pointer to array specifying if triangles need to be refined (=1) or are allowed to be coarsened (=-1)
\param *pVertexRemoveFlag Pointer to array specifying whether vertices can be removed. Entries are set to zero if vertex can not be removed, otherwise it is left unchanged*/
//#########################################################################

__host__ __device__
void FillVertexRemoveFlagSingle(int i, int3 *pTv,
                                int nVertex, int *pTriangleWantRefine,
                                int *pVertexRemoveFlag)
{
  // If wantRefine != -1, vertex can not be removed
  if (pTriangleWantRefine[i] != -1) {
    int v1 = pTv[i].x;
    int v2 = pTv[i].y;
    int v3 = pTv[i].z;
    while (v1 >= nVertex) v1 -= nVertex;
    while (v2 >= nVertex) v2 -= nVertex;
    while (v3 >= nVertex) v3 -= nVertex;
    while (v1 < 0) v1 += nVertex;
    while (v2 < 0) v2 += nVertex;
    while (v3 < 0) v3 += nVertex;

    AtomicExch(&(pVertexRemoveFlag[v1]), 0);
    AtomicExch(&(pVertexRemoveFlag[v2]), 0);
    AtomicExch(&(pVertexRemoveFlag[v3]), 0);
  }
}

//#########################################################################
/*! \brief Kernel selecting vertices for removal

If a triangle is not flagged as wanting coarsening, none of its vertices can be removed. In this case, \a vertexRemoveFlag is set to zero for these vertices using atomic operations.

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param nVertex Total number of vertices in Mesh
\param *pTriangleWantRefine Pointer to array specifying if triangles need to be refined (=1) or are allowed to be coarsened (=-1)
\param *pVertexRemoveFlag Pointer to array specifying whether vertices can be removed. Entries are set to zero if vertex can not be removed, otherwise it is left unchanged*/
//#########################################################################

__global__
void devFillVertexRemoveFlag(int nTriangle, int3 *pTv,
                             int nVertex, int *pTriangleWantRefine,
                             int *pVertexRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FillVertexRemoveFlagSingle(i, pTv, nVertex,
                               pTriangleWantRefine, pVertexRemoveFlag);

    i += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! If a triangle is not flagged as wanting coarsening, none of its vertices can be removed. In this case, \a vertexRemoveFlag is set to zero for these vertices using atomic operations.

\param *vertexRemoveFlag Pointer to Array specifying whether vertices can be removed. Entries are set to zero if vertex can not be removed, otherwise it is left unchanged*/

//#########################################################################

void Coarsen::FlagVertexRemove(Connectivity *connectivity,
                               Array<int> *vertexRemoveFlag,
                               Array<int> *triangleWantRefine)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemoveFlag->TransformToHost();
      triangleWantRefine->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemoveFlag->TransformToDevice();
      triangleWantRefine->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  int3 *pTv = connectivity->triangleVertices->GetPointer();

  int *pVertexRemoveFlag = vertexRemoveFlag->GetPointer();
  int *pTriangleWantRefine = triangleWantRefine->GetPointer();

  // Flag vertices for removal
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillVertexRemoveFlag,
                                       (size_t) 0, 0);

    devFillVertexRemoveFlag<<<nBlocks, nThreads>>>
      (nTriangle, pTv, nVertex,
       pTriangleWantRefine, pVertexRemoveFlag);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      FillVertexRemoveFlagSingle(i, pTv, nVertex,
                                 pTriangleWantRefine, pVertexRemoveFlag);
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexRemoveFlag->TransformToHost();
      triangleWantRefine->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexRemoveFlag->TransformToDevice();
      triangleWantRefine->TransformToDevice();

      cudaFlag = 1;
    }
  }

}

}
