// -*-c++-*-
/*! \file removevertices.cpp
\brief Top-level function for coarsening Mesh

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../../Common/nvtxEvent.h"
#include "../Delaunay/delaunay.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../Predicates/predicates.h"

namespace astrix {

//#########################################################################
//#########################################################################

template<class realNeq>
int Coarsen::RemoveVertices(Connectivity *connectivity,
                            Predicates *predicates,
                            Array<realNeq> *vertexState,
                            Array<int> *triangleWantRefine,
                            const MeshParameter *meshParameter,
                            Delaunay *delaunay,
                            int maxCycle)
{
  nvtxEvent *nvtxCoarsen = new nvtxEvent("Coarsen", 0);

  if (verboseLevel > 1)
    std::cout << "Coarsening mesh..." << std::endl;

  //real areaBefore = 0.0;
  // Check if Mesh is valid
  if (debugLevel > 0) {
    connectivity->Save(900);
    try {
      connectivity->CheckEdgeTriangles();
      connectivity->CheckTriangleAreas(predicates, meshParameter);
      connectivity->CheckEncroach(meshParameter);
    }
    catch (...) {
      std::cout << "Invalid mesh at start of coarsening step, exiting"
                << std::endl;
      throw;
    }
  }

  int nVertexOld = connectivity->vertexCoordinates->GetSize();

  edgeCoordinates->SetSize(connectivity->edgeTriangles->GetSize());

  int nCycle = 0;
  int finishedCoarsen = 0;

  // Reject triangles larger than half the domain size
  RejectLargeTriangles(connectivity, meshParameter, triangleWantRefine);

  while (!finishedCoarsen) {
    if (verboseLevel > 1) std::cout << "Coarsen cycle " << nCycle;

    FillEdgeCollapseList(connectivity, triangleWantRefine);
    TestEdgeCollapse(connectivity, predicates, meshParameter);

    int nRemove = edgeCollapseList->GetSize();

    // If no valid points left, we are done
    if (nRemove == 0) {
      finishedCoarsen = 1;
      if (verboseLevel > 1) std::cout << std::endl;
      break;
    }

    if (verboseLevel > 1)
      std::cout << ", vertices to be removed: " << nRemove << ", ";


    // Find list of vertices that can be removed in parallel
    FindParallelDeletionSet(connectivity);
    nRemove = edgeCollapseList->GetSize();

    if (verboseLevel > 1)
      std::cout << "in parallel: " << nRemove << std::endl;

    // Should not happen
    if (nRemove == 0) {
      std::cout << "No vertices to remove in parallel, exiting" << std::endl;
      throw std::runtime_error("");
    }

    // Adjust state for conservation
    AdjustState<realNeq>(connectivity,
                         vertexState,
                         meshParameter);

    // Remove vertices from mesh
    Remove<realNeq>(connectivity, triangleWantRefine, vertexState);

    if (debugLevel > 0) {
      connectivity->Save(901);
      try {
        connectivity->CheckEdgeTriangles();
        connectivity->CheckTriangleAreas(predicates, meshParameter);
      }
      catch (...) {
        std::cout << "Invalid mesh after removing vertices, exiting"
                  << std::endl;
        throw;
      }
    }

    int nEdgeCheck = edgeNeedsChecking->RemoveValue(-1);

    // Maintain Delaunay triangulation
    delaunay->MakeDelaunay<realNeq>(connectivity, vertexState,
                                    predicates, meshParameter, 0,
                                    edgeNeedsChecking, nEdgeCheck, 0);

    nCycle++;
    if (nCycle >= maxCycle) finishedCoarsen = 1;
  }

  // Check validity of Mesh after coarsening
  if (debugLevel > 0) {
    connectivity->Save(902);
    try {
      connectivity->CheckEdgeTriangles();
      connectivity->CheckTriangleAreas(predicates, meshParameter);
      connectivity->CheckEncroach(meshParameter);
    }
    catch (...) {
      std::cout << "Invalid mesh at end of coarsening step, exiting"
                << std::endl;
      throw;
    }
  }

  delete nvtxCoarsen;

  //std::cout << "Hallo" << std::endl;
  //int qq; std::cin >>qq;

  return nVertexOld - connectivity->vertexCoordinates->GetSize();
}

//#############################################################################
// Instantiate
//#############################################################################

template int
Coarsen::RemoveVertices<real>(Connectivity *connectivity,
                              Predicates *predicates,
                              Array<real> *vertexState,
                              Array<int> *triangleWantRefine,
                              const MeshParameter *meshParameter,
                              Delaunay *delaunay,
                              int maxCycle);
template int
Coarsen::RemoveVertices<real3>(Connectivity *connectivity,
                               Predicates *predicates,
                               Array<real3> *vertexState,
                               Array<int> *triangleWantRefine,
                               const MeshParameter *meshParameter,
                               Delaunay *delaunay,
                               int maxCycle);
template int
Coarsen::RemoveVertices<real4>(Connectivity *connectivity,
                               Predicates *predicates,
                               Array<real4> *vertexState,
                               Array<int> *triangleWantRefine,
                               const MeshParameter *meshParameter,
                               Delaunay *delaunay,
                               int maxCycle);

}
