// -*-c++-*-
/*! \file edge.cu
\brief Functions for Morton ordering edges

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./morton.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! \brief Compute Morton value for edge \a i

The Morton value for an edge is just the Morton value of the first vertex of one of the neighbouring triangles.

\param i Edge index to consider
\param *pEt Pointer edge triangles
\param *pTv Pointer triangle vertices
\param *pVmort Pointer to array of vertex Morton values
\param *pMortValues Pointer to output array
\param nVertex Total number of vertices in Mesh*/
//#########################################################################

__host__ __device__
void FillMortonEdgeSingle(int i, int2 *pEt, int3 *pTv, unsigned int *pVmort,
                          unsigned int *pMortValues, int nVertex)
{
  int t = max(pEt[i].x, pEt[i].y);
  int v = pTv[t].x;
  while (v >= nVertex) v -= nVertex;
  while (v < 0) v += nVertex;

  pMortValues[i] = pVmort[v];
}

//######################################################################
/*! \brief Kernel computing Morton values for all edges

The Morton value for an edge is just the Morton value of the first vertex of one of the neighbouring triangles.

\param nEdge Total number of edges in Mesh
\param *pEt Pointer edge triangles
\param *pTv Pointer triangle vertices
\param *pVmort Pointer to array of vertex Morton values
\param *pMortValues Pointer to output array
\param nVertex Total number of vertices in Mesh*/
//######################################################################

__global__ void
devFillMortonEdge(int nEdge, int2 *pEt, int3 *pTv, unsigned int *pVmort,
                  unsigned int *pMortValues, int nVertex)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    FillMortonEdgeSingle(i, pEt, pTv, pVmort, pMortValues, nVertex);

    i += gridDim.x*blockDim.x;
  }
}

//#########################################################################
/*! Sort edges according to their Morton value to improve data locality. The Morton value for an edge is just the Morton value of the first vertex of one of the neighbouring triangles. We sort \a edgeTriangles and adjust \a triangleEdges

\param *connectivity Pointer to basic Mesh data*/
//#########################################################################

void Morton::OrderEdge(Connectivity * const connectivity)
{
  int nVertex = vertexMorton->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  unsigned int *pIndex = index->GetPointer();
  unsigned int *pInverseIndex = inverseIndex->GetPointer();

  unsigned int *pMortValues = mortValues->GetPointer();
  unsigned int *pVmort = vertexMorton->GetPointer();

  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  // Morton values for edges
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devFillMortonEdge,
                                       (size_t) 0, 0);

    devFillMortonEdge<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTv, pVmort, pMortValues, nVertex);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < nEdge; i++)
      FillMortonEdgeSingle(i, pEt, pTv, pVmort, pMortValues, nVertex);
  }

  // Sort Morton values
  index->SetToSeries();
  mortValues->SortByKey(index);

  // Reorder edges
  connectivity->edgeTriangles->Reindex(pIndex);

  // pInverseIndex[pIndex[i]] = i
  inverseIndex->ScatterSeries(index, nEdge);

  // Adjust triangle edges
  connectivity->triangleEdges->InverseReindex(pInverseIndex, nEdge, true);
}

}  // namespace astrix
