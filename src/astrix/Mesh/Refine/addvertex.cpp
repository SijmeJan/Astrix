// -*-c++-*-
/*! \file addvertices.cpp
\brief Add specific list of vertices to Mesh

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
#include "./refine.h"
#include "../Delaunay/delaunay.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! Add a list of vertices to the Mesh. Used in cases where vertex locations are known in advance, i.e. when inserting boundaries. Can only be used at t = 0, so no interpolation of the state is necessary.

\param *connectivity Pointer to basic Mesh data: vertices, triangles, edges
\param *meshParameter Pointer to Mesh parameters, read from input file
\param *predicates Exact geometric predicates
\param *delaunay Pointer to Delaunay object, used to maintain Delaunay triangulation
\param *vertexCoordinatesToAdd Pointer to coordinates of vertices to be added
\param *vertexOrder Output: the order in which vertices were inserted.*/
//#########################################################################

int Refine::AddVertices(Connectivity * const connectivity,
                        const MeshParameter *meshParameter,
                        const Predicates *predicates,
                        Delaunay * const delaunay,
                        Array<real2> * const vertexCoordinatesToAdd,
                        Array<int> * const vertexOrder)
{
  nvtxEvent *nvtxRefine = new nvtxEvent("Refine", 1);

  // Going to add all vertices
  int nToAdd = vertexCoordinatesToAdd->GetSize();
  if (verboseLevel > 1)
    std::cout << "Going to add " << nToAdd << " vertices" << std::endl;

  int nAdded = 0;

  Array<int> *vertexOrderInsert = new Array<int>(1, cudaFlag);
  Array<int> *vertexOrderTotal =
    new Array<int>(1, cudaFlag, (unsigned int) nToAdd);
  vertexOrderTotal->SetToSeries();
  vertexOrderTotal->AddValue(4, 0, nToAdd);

  // While there are still points to add
  while (nToAdd > 0) {
    vertexOrderInsert->SetSize(nToAdd);
    vertexOrderInsert->SetEqual(vertexOrderTotal);

    elementAdd->SetSize(nToAdd);
    vertexCoordinatesAdd->SetSize(nToAdd);
    vertexCoordinatesAdd->SetEqual(vertexCoordinatesToAdd);

    badTriangles->SetSize(nToAdd);
    badTriangles->SetToValue(-1);

    // Find triangles for all new vertices
    try {
      FindTriangles(connectivity, meshParameter, predicates);
    }
    catch (...) {
      std::cout << "Error finding triangles" << std::endl;
      throw;
    }

    // Find unique triangle set
    FindParallelInsertionSet(connectivity,
                             vertexOrderTotal,
                             vertexOrderInsert,
                             vertexCoordinatesToAdd,
                             predicates, meshParameter);

    // Add inserted vertices to vertexOrder
    if (vertexOrder != 0)
      vertexOrder->Concat(vertexOrderInsert);

    nToAdd = elementAdd->GetSize();
    nAdded += nToAdd;

    AddToPeriodic(connectivity, nToAdd);

    // Insert vertices into Mesh
    InsertVertices<real>(connectivity,
                         meshParameter,
                         predicates, (Array<real> *) 0, 0);

    nToAdd = vertexCoordinatesToAdd->GetSize();

    // Maintain Delaunay triangulation
    delaunay->MakeDelaunay<real>(connectivity, 0, predicates,
                                 meshParameter, 0, 0, 0, 0);

    if (verboseLevel > 1)
      std::cout << ", added so far: " << nAdded << std::endl;
  }

  delete vertexOrderInsert;
  delete vertexOrderTotal;

  delete nvtxRefine;

  return nAdded;
}

}  // namespace astrix
