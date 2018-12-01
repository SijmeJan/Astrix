// -*-c++-*-
/*! \file order.cpp
\brief Functions for Morton ordering vertices, triangles and edges

*/ /* \section LICENSE
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
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//######################################################################
/*! Sort vertices, edges and triangles according to their Morton value to improve data locality. We first compute the Morton values for the vertex coordinates and use those to sort vertices, triangles and edges.

\param *connectivity Pointer to basic Mesh data
\param *triangleWantRefine Pointer to flags whether triangles need to be refined
\param *vertexState Pointer to state vector*/
//######################################################################

template<class realNeq>
void Morton::Order(Connectivity * const connectivity,
                   Array<int> * const triangleWantRefine,
                   Array<realNeq> * const vertexState)
{
  nvtxEvent *nvtxMorton = new nvtxEvent("Morton", 5);
  nvtxEvent *temp = new nvtxEvent("Minmax", 1);

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  // Determine domain edges
  minx = connectivity->vertexCoordinates->MinimumComb<real>(0);
  maxx = connectivity->vertexCoordinates->MaximumComb<real>(0);
  miny = connectivity->vertexCoordinates->MinimumComb<real>(1);
  maxy = connectivity->vertexCoordinates->MaximumComb<real>(1);

  delete temp;
  temp = new nvtxEvent("Values", 2);

  // Will contain Morton values for every vertex
  vertexMorton->SetSize(nVertex);

  // These need to be able to contain values for either:
  // vertices, triangles or edges. Therefore, set size
  // to nEdge > nTriangle > nVertex.
  mortValues->SetSize(nEdge);
  index->SetSize(nEdge);
  inverseIndex->SetSize(nEdge);

  // Calculate Morton values for every vertex
  CalcValues(connectivity);

  delete temp;
  temp = new nvtxEvent("Vertices", 3);

  // Reorder vertices
  OrderVertex<realNeq>(connectivity, vertexState);

  delete temp;
  temp = new nvtxEvent("Triangles", 4);

  // Reorder triangles
  OrderTriangle(connectivity, triangleWantRefine);

  delete temp;
  temp = new nvtxEvent("Edges", 6);

  // Reorder edges
  OrderEdge(connectivity);

  delete temp;
  delete nvtxMorton;
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Morton::Order<real>(Connectivity * const connectivity,
                    Array<int> * const triangleWantRefine,
                    Array<real> * const vertexState);
template void
Morton::Order<real3>(Connectivity * const connectivity,
                     Array<int> * const triangleWantRefine,
                     Array<real3> * const vertexState);
template void
Morton::Order<real4>(Connectivity * const connectivity,
                     Array<int> * const triangleWantRefine,
                     Array<real4> * const vertexState);


}  // namespace astrix
