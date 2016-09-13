// -*-c++-*-
/*! \file triangle.cu
\brief Functions for Morton ordering triangles*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./morton.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {
  
//#########################################################################
/*! \brief Calculate Morton value for triangle \a i

The Morton value for a triangle is just the Morton value for its first vertex.

\param i Triangle index to consider
\param *pTv Pointer to triangle vertices
\param *pVmort Pointer to array of vertex Morton values
\param *pMortValues Pointer to output array
\param nVertex Total number of vertices in Mesh*/
//#########################################################################

__host__ __device__
void FillMortonTriangleSingle(int i, int3 *pTv, unsigned int *pVmort,
			      unsigned int *pMortValues, int nVertex)
{
  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;

  while (a >= nVertex) a -= nVertex;
  while (a < 0) a += nVertex;
  while (b >= nVertex) b -= nVertex;
  while (b < 0) b += nVertex;
  while (c >= nVertex) c -= nVertex;
  while (c < 0) c += nVertex;

  int vExpect = (int)((real)(a + b + c)/3.0f);
  
  int v = a;
  if (abs(b - vExpect) <= abs(a - vExpect) &&
      abs(b - vExpect) <= abs(c - vExpect)) v = b;
  if (abs(c - vExpect) <= abs(a - vExpect) &&
      abs(c - vExpect) <= abs(b - vExpect)) v = c;  
  
  pMortValues[i] = pVmort[v];
}
  
//######################################################################
/*! \brief Kernel calculating Morton values for triangles

The Morton value for a triangle is just the Morton value for its first vertex.

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVmort Pointer to array of vertex Morton values
\param *pMortValues Pointer to output array
\param nVertex Total number of vertices in Mesh*/
//######################################################################

__global__ void 
devFillMortonTriangle(int nTriangle, int3 *pTv, unsigned int *pVmort,
		      unsigned int *pMortValues, int nVertex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  while (i < nTriangle) {
    FillMortonTriangleSingle(i, pTv, pVmort, pMortValues, nVertex);

    i += gridDim.x*blockDim.x;
  }
}
  
//#########################################################################
/*! Sort triangles according to their Morton value to improve data locality. The Morton value for a triangle is just the Morton value for its first vertex. We sort \a triangleVertices, \a triangleEdges and \a triangleWantRefine and adjust \a edgeTriangle.

\param *connectivity Pointer to basic Mesh data
\param *triangleWantRefine Pointer to flags if triangles need to be refined*/
//#########################################################################

void Morton::OrderTriangle(Connectivity * const connectivity,
			   Array<int> * const triangleWantRefine)
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = vertexMorton->GetSize();
  
  unsigned int *pIndex = index->GetPointer();
  unsigned int *pInverseIndex = inverseIndex->GetPointer();
  
  unsigned int *pMortValues = mortValues->GetPointer();
  unsigned int *pVmort = vertexMorton->GetPointer();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  
  // Morton values for edges
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFillMortonTriangle, 
				       (size_t) 0, 0);

    devFillMortonTriangle<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVmort, pMortValues, nVertex);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < nTriangle; i++) 
      FillMortonTriangleSingle(i, pTv, pVmort, pMortValues, nVertex);
  }

  // Sort Morton values
  index->SetToSeries();
  mortValues->SortByKey(index, nTriangle);

  // Reorder triangles
  connectivity->triangleVertices->Reindex(pIndex);
  connectivity->triangleEdges->Reindex(pIndex);
  if (triangleWantRefine != 0)
    triangleWantRefine->Reindex(pIndex);

  // pInverseIndex[pIndex[i]] = i
  inverseIndex->ScatterSeries(index, nTriangle);

  // Adjust edge triangles
  connectivity->edgeTriangles->InverseReindex(pInverseIndex, nTriangle, true);
}

}
