// -*-c++-*-
/*! \file improve.cpp
\brief Function to improve quality of Mesh by adding vertices

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../Morton/morton.h"
#include "../Delaunay/delaunay.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! Function to improve quality of Mesh by adding new vertices, until the requirements as specified in MeshParameter are met. Returns the number of vertices that were added.

\param *connectivity Pointer to basic Mesh data: vertices, triangles, edges
\param *meshParameter Pointer to Mesh parameters, read from input file
\param *predicates Exact geometric predicates
\param *morton Pointer to Morton object, used for sorting to improve data locality
\param *delaunay Pointer to Delaunay object, used to maintain Delaunay triangulation
\param *vertexState Pointer to state at vertices. If refining at t > 0, we need to interpolate the state at new vertices. Otherwise, this pointer needs to be 0 and will be ignored.
\param specificHeatRatio Ratio of specific heats. Needed when interpolating the state.
\param *triangleWantRefine Pointer to flags if triangle needs to be refined based on current state. Only used when t > 0; otherwise it needs to be 0.*/
//#########################################################################

template<class realNeq, ConservationLaw CL>
int Refine::ImproveQuality(Connectivity * const connectivity,
                           const MeshParameter *meshParameter,
                           const Predicates *predicates,
                           Morton * const morton,
                           Delaunay * const delaunay,
                           Array<realNeq> * const vertexState,
                           const real specificHeatRatio,
                           Array<int> * const triangleWantRefine)
{
  nvtxEvent *nvtxRefine = new nvtxEvent("Refine", 1);

  int nVertexOld = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();
  edgeNeedsChecking->SetSize(nEdge);

  int finished = 0;
  int ncycle = 0;
  int addedVerticesFlag = 0;
  int nAddedSinceMorton = 0;
  real maxFracAddedMorton = 0.07;

  // Maintain Delaunay triangulation
  delaunay->MakeDelaunay<realNeq, CL>(connectivity, vertexState,
                                      predicates, meshParameter, 0, 0, 0, 0);

  while (!finished) {
    if (verboseLevel > 1)
      std::cout << "Refine cycle " << ncycle;

    if (debugLevel > 0)
      Save(ncycle, connectivity);

    if (verboseLevel > 2)
      std::cout << std::endl << "Testing triangles..." << std::endl;

    // Look for low-quality triangles; result in badTriangles
    int nRefine = TestTrianglesQuality(connectivity,
                                       meshParameter,
                                       triangleWantRefine);

    if (verboseLevel > 2)
      std::cout << "Finding circumcentres..." << std::endl;

    // New points will be added in circumcentres of bad triangles
    FindCircum(connectivity, meshParameter, nRefine);

    if (nRefine == 0) {
      // No bad triangles: done
      finished = 1;
    } else {
      // Adding points on triangle or edge
      elementAdd->SetSize(nRefine);

      if (verboseLevel > 2)
        std::cout << "Finding triangles..." << std::endl;

      // Find triangles for all new vertices
      try {
        FindTriangles(connectivity, meshParameter, predicates);
      }
      catch (...) {
        std::cout << "Error finding triangles" << std::endl;
        throw;
      }

      if (verboseLevel > 2)
        std::cout << "Testing encroachment..." << std::endl;

      // Check if any new vertex encroaches segment
      TestEncroach(connectivity, meshParameter, nRefine);

      if (verboseLevel > 1)
        std::cout << ", nBadTriangle = " << nRefine << ", ";

      if (verboseLevel > 2)
        std::cout << std::endl << "Parallel insertion..." << std::endl;

      // Find unique triangle set
      FindParallelInsertionSet(connectivity, 0, 0, 0,
                               predicates, meshParameter);

      nRefine = elementAdd->GetSize();
      if (nRefine == 0) {
        finished = 1;
      } else {
        addedVerticesFlag = 1;
        nAddedSinceMorton += nRefine;

        // If necessary, interpolate state
        if (vertexState != 0)
          InterpolateState<realNeq, CL>(connectivity,
                                        meshParameter,
                                        vertexState,
                                        triangleWantRefine,
                                        specificHeatRatio);

        if (verboseLevel > 2)
          std::cout << "Add periodic..." << std::endl;

        // Adjust periodic vertices
        AddToPeriodic(connectivity, nRefine);

        if (verboseLevel > 2)
          std::cout << "Inserting vertices..." << std::endl;

        // Insert new vertices into Mesh
        int nTriangleOld = connectivity->triangleVertices->GetSize();
        InsertVertices<realNeq, CL>(connectivity, meshParameter, predicates,
                                    vertexState, triangleWantRefine);

        // Output memory usage to stdout
        if (verboseLevel > 1) {
          if (cudaFlag == 0) {
            std::cout << ((real)(Array<real>::memAllocatedHost) +
                          (real)(Array<real2>::memAllocatedHost) +
                          (real)(Array<real3>::memAllocatedHost) +
                          (real)(Array<real4>::memAllocatedHost) +
                          (real)(Array<int>::memAllocatedHost) +
                          (real)(Array<int2>::memAllocatedHost) +
                          (real)(Array<int3>::memAllocatedHost) +
                          (real)(Array<int4>::memAllocatedHost) +
                          (real)(Array<unsigned int>::memAllocatedHost))/
              (real) (1073741824) << " Gb"
                      << std::endl;
          } else {
            std::cout << ((real)(Array<real>::memAllocatedDevice) +
                          (real)(Array<real2>::memAllocatedDevice) +
                          (real)(Array<real3>::memAllocatedDevice) +
                          (real)(Array<real4>::memAllocatedDevice) +
                          (real)(Array<int>::memAllocatedDevice) +
                          (real)(Array<int2>::memAllocatedDevice) +
                          (real)(Array<int3>::memAllocatedDevice) +
                          (real)(Array<int4>::memAllocatedDevice) +
                          (real)(Array<unsigned int>::memAllocatedDevice))/
              (real) (1073741824) << " Gb"
                      << std::endl;
          }
        }

        if (verboseLevel > 2)
          std::cout << std::endl << "Splitting segments..." << std::endl;

        // Check if any of the points inserted on a segment encroach
        // a second segment; if so, split this second segment.
        SplitSegment<realNeq, CL>(connectivity, meshParameter, predicates,
                                  vertexState, triangleWantRefine,
                                  specificHeatRatio, nTriangleOld);


        int nEdgeCheck = edgeNeedsChecking->RemoveValue(-1);

        if (verboseLevel > 2)
          std::cout << "Delaunay..." << std::endl;

        // Maintain Delaunay triangulation
        delaunay->MakeDelaunay<realNeq, CL>(connectivity, vertexState,
                                            predicates, meshParameter, 0,
                                            edgeNeedsChecking, nEdgeCheck, 0);

        if (verboseLevel > 2)
          std::cout << "Morton..." << std::endl;

        // Morton ordering to preserve data locality
        int nVertex = connectivity->vertexCoordinates->GetSize();
        real fracAdded = (real) nAddedSinceMorton/(real) nVertex;
        if (debugLevel < 10 && fracAdded > maxFracAddedMorton) {
          morton->Order<realNeq, CL>(connectivity,
                                     triangleWantRefine,
                                     vertexState);
          nAddedSinceMorton = 0;
        }
      }
    }

    ncycle++;
  }

  // Final Morton ordering
  if (debugLevel >= 10 || nAddedSinceMorton > 0)
    morton->Order<realNeq, CL>(connectivity, triangleWantRefine, vertexState);

  if (verboseLevel > 1)
    std::cout << std::endl
              << "Number of cycles needed: " << ncycle << std::endl;

  delete nvtxRefine;

  // Return number of vertices added
  return connectivity->vertexCoordinates->GetSize() - nVertexOld;
}

//##############################################################################
// Instantiate
//##############################################################################

template int
Refine::ImproveQuality<real,
                       CL_ADVECT>(Connectivity * const connectivity,
                                  const MeshParameter *meshParameter,
                                  const Predicates *predicates,
                                  Morton * const morton,
                                  Delaunay * const delaunay,
                                  Array<real> * const vertexState,
                                  const real specificHeatRatio,
                                  Array<int> * const triangleWantRefine);
template int
Refine::ImproveQuality<real,
                       CL_BURGERS>(Connectivity * const connectivity,
                                   const MeshParameter *meshParameter,
                                   const Predicates *predicates,
                                   Morton * const morton,
                                   Delaunay * const delaunay,
                                   Array<real> * const vertexState,
                                   const real specificHeatRatio,
                                   Array<int> * const triangleWantRefine);
template int
Refine::ImproveQuality<real3,
                       CL_CART_ISO>(Connectivity * const connectivity,
                                    const MeshParameter *meshParameter,
                                    const Predicates *predicates,
                                    Morton * const morton,
                                    Delaunay * const delaunay,
                                    Array<real3> * const vertexState,
                                    const real specificHeatRatio,
                                    Array<int> * const triangleWantRefine);
template int
Refine::ImproveQuality<real4,
                       CL_CART_EULER>(Connectivity * const connectivity,
                                      const MeshParameter *meshParameter,
                                      const Predicates *predicates,
                                      Morton * const morton,
                                      Delaunay * const delaunay,
                                      Array<real4> * const vertexState,
                                      const real specificHeatRatio,
                                      Array<int> * const triangleWantRefine);

}  // namespace astrix
