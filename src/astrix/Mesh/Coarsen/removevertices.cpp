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
/*! Coarsen mesh. First calculate an estimate of the discretization error and flag triangles that can be coarsened. Then remove as many vertices as possible from mesh.

\param *vertexState Pointer to state vector at vertices
\param specificHeatRatio Ratio of specific heats
\param nTimeStep Number of time steps taken so far. Used in combination with \a nStepSkipCoarsen to possibly avoid coarsening every timestep*/
//#########################################################################

int Coarsen::RemoveVertices(Connectivity *connectivity,
                            Predicates *predicates,
                            Array<realNeq> *vertexState,
                            real specificHeatRatio,
                            Array<int> *triangleWantRefine,
                            const MeshParameter *meshParameter,
                            Delaunay *delaunay,
                            int maxCycle,
                            Array<unsigned int> *randomVector)
{
  nvtxEvent *nvtxCoarsen = new nvtxEvent("Coarsen", 0);

  if (verboseLevel > 1)
    std::cout << "Coarsening mesh..." << std::endl;

  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nVertexOld = nVertex;

  int nCycle = 0;
  int finishedCoarsen = 0;
  int removedVerticesFlag = 0;

  while (!finishedCoarsen) {
    if (verboseLevel > 1) std::cout << "Coarsen cycle " << nCycle;

    RejectLargeTriangles(connectivity, meshParameter, triangleWantRefine);

    // For every vertex, a single triangle associated with it
    nVertex = connectivity->vertexCoordinates->GetSize();
    vertexTriangle->SetSize(nVertex);

    FillVertexTriangle(connectivity);
    int maxTriPerVert = MaxTriPerVert(connectivity);

    Array<int> *vertexRemoveFlag =
      new Array<int>(1, cudaFlag, nVertex);
    // Assume all vertices can be removed
    vertexRemoveFlag->SetToValue(1);

    FlagVertexRemove(connectivity, vertexRemoveFlag, triangleWantRefine);

    Array<int> *vertexRemoveFlagScan =
      new Array<int>(1, cudaFlag, nVertex);
    int nRemove = vertexRemoveFlag->ExclusiveScan(vertexRemoveFlagScan,
                                                  nVertex);

    if (verboseLevel > 1)
      std::cout << " nRemove: " << nRemove;

    if (nRemove == 0) {
      finishedCoarsen = 1;
      if (verboseLevel > 1) std::cout << std::endl;
    } else {
      vertexRemove->SetSize(nVertex);
      vertexRemove->SetToSeries();
      vertexRemove->Compact(nRemove, vertexRemoveFlag,
                            vertexRemoveFlagScan);
      vertexTriangle->Compact(nRemove, vertexRemoveFlag,
                              vertexRemoveFlagScan);

      vertexRemoveFlag->SetSize(nRemove);
      vertexRemoveFlag->SetToValue(1);

      // Check if removing vertex leads to encroached triangle
      CheckEncroach(connectivity, predicates, vertexRemoveFlag, meshParameter);

      vertexRemoveFlagScan->SetSize(nRemove);
      nRemove = vertexRemoveFlag->ExclusiveScan(vertexRemoveFlagScan,
                                                nRemove);

      if (nRemove == 0) {
        finishedCoarsen = 1;
        if (verboseLevel > 1) std::cout << std::endl;
      } else {
        vertexRemove->Compact(nRemove, vertexRemoveFlag,
                              vertexRemoveFlagScan);
        vertexTriangle->Compact(nRemove, vertexRemoveFlag,
                                vertexRemoveFlagScan);

        if (verboseLevel > 1)
          std::cout << ", vertices to be removed: " << nRemove << ", ";

        // Find list of vertices that can be removed in parallel
        FindParallelDeletionSet(connectivity, maxTriPerVert, randomVector);
        nRemove = vertexRemove->GetSize();

        if (verboseLevel > 1)
          std::cout << "in parallel: " << nRemove << std::endl;

        if (debugLevel > 0) {
          if (nRemove == 0) {
            std::cout << "Error!" << std::endl;
            int qq; std::cin >>qq;
          }
        }

        Array<int> *vertexTriangleList =
          new Array<int>(1, cudaFlag, nRemove*maxTriPerVert);
        vertexTriangleList->SetToValue(-1);

        CreateVertexTriangleList(connectivity,
                                 vertexTriangleList,
                                 maxTriPerVert);

        Array<int> *vertexTriangleAllowed =
          new Array<int>(1, cudaFlag, nRemove*maxTriPerVert);

        FindAllowedTargetTriangles(connectivity, predicates,
                                   vertexTriangleAllowed,
                                   vertexTriangleList, maxTriPerVert,
                                   meshParameter);

        Array<int> *triangleTarget =
          new Array<int>(1, cudaFlag, nRemove);

        FindTargetTriangles(connectivity, triangleWantRefine,
                            triangleTarget, vertexTriangleAllowed,
                            vertexTriangleList, maxTriPerVert);

        delete vertexTriangleAllowed;

        Array<int> *vertexNeighbour =
          new Array<int>(1, cudaFlag, nRemove*maxTriPerVert);

        FindVertexNeighbours(connectivity, vertexNeighbour, triangleTarget,
                             vertexTriangleList, maxTriPerVert);

        AdjustState(connectivity, maxTriPerVert,
                    vertexTriangleList,
                    triangleTarget,
                    vertexState,
                    specificHeatRatio,
                    meshParameter,
                    vertexNeighbour);

        delete vertexNeighbour;

        // Remove vertices from mesh
        Remove(connectivity, triangleWantRefine,
               vertexTriangleList, maxTriPerVert,
               triangleTarget, vertexState);

        delete vertexTriangleList;
        delete triangleTarget;

        delaunay->MakeDelaunay(connectivity, vertexState,
                               predicates, meshParameter, 0, 0, 0, 0);

        removedVerticesFlag = 1;
      }
    }

    delete vertexRemoveFlag;
    delete vertexRemoveFlagScan;

    nCycle++;

    if (nCycle >= maxCycle) finishedCoarsen = 1;
  }

  delete nvtxCoarsen;

  return nVertexOld - nVertex;
}

}
