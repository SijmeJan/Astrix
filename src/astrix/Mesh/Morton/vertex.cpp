// -*-c++-*-
/*! \file vertex.cpp
\brief Functions for Morton ordering vertices*/
#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./morton.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"

namespace astrix {
    
//#########################################################################
/*! Order vertices according to Morton value to increase data locality. The Morton values are assumed to have been precomputed in \a vertexMorton, so all there is to do is sort the vertex coordinates and \a vertexState, and to reindex \a triangleVertices.

\param *connectivity Pointer to basic Mesh data
\param *vertexState Pointer to Array containing state vector (density, momentum, energy)*/
//#########################################################################

void Morton::OrderVertex(Connectivity * const connectivity,
			 Array<realNeq> * const vertexState)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();
  
  // Sort Morton values
  index->SetToSeries();
  vertexMorton->SortByKey(index, nVertex);

  unsigned int *pIndex = index->GetPointer();
  unsigned int *pInverseIndex = inverseIndex->GetPointer();

  // Reorder vertices and state
  connectivity->vertexCoordinates->Reindex(pIndex);
  if (vertexState != 0)
    vertexState->Reindex(pIndex);

  // pInverseIndex[pIndex[i]] = i
  inverseIndex->ScatterSeries(index, nVertex);

  // Adjust tv's
  connectivity->triangleVertices->InverseReindex(pInverseIndex, nVertex, false);
}

}
