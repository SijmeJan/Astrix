// -*-c++-*-
/*! \file vertex.cpp
\brief Functions for Morton ordering vertices

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>

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

template<class realNeq, ConservationLaw CL>
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

//##############################################################################
// Instantiate
//##############################################################################

template void
Morton::OrderVertex<real, CL_ADVECT>(Connectivity * const connectivity,
                                     Array<real> * const vertexState);
template void
Morton::OrderVertex<real, CL_BURGERS>(Connectivity * const connectivity,
                                      Array<real> * const vertexState);
template void
Morton::OrderVertex<real3, CL_CART_ISO>(Connectivity * const connectivity,
                                        Array<real3> * const vertexState);
template void
Morton::OrderVertex<real4, CL_CART_EULER>(Connectivity * const connectivity,
                                          Array<real4> * const vertexState);

}  // namespace astrix
