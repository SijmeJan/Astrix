// -*-c++-*-
/*! \file makedelaunay.cpp
\brief Main routine transforming Mesh into Delaunay Mesh

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
#include "./delaunay.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! Transform triangulated Mesh into Delaunay Mesh. This is achieved by flipping edges that do not have the Delaunay property. First, we make a list of edges that are not Delaunay, then we select those that can be flipped in parallel, we adjust the state vector in order to conserve mass, momentum and energy, and finally we flip the edges. A repair step ensures all edges have the correct neighbouring triangles. This is repeated until all edges are Delaunay.

\param *connectivity Pointer to basic Mesh data
\param *vertexState Pointer to state vector
\param *predicates Pointer to Predicates object, used to check Delaunay property without roundoff error
\param *meshParameter Pointer to Mesh parameters
\param maxCycle Maximum number of cycles. If <= 0, cycle until all edges are Delaunay*/
//#########################################################################

template<class realNeq, ConservationLaw CL>
void Delaunay::MakeDelaunay(Connectivity * const connectivity,
                            Array<realNeq> * const vertexState,
                            const Predicates *predicates,
                            const MeshParameter *meshParameter,
                            const int maxCycle,
                            Array<int> * const edgeNeedsChecking,
                            const int nEdgeCheck,
                            const int flopFlag)
{
  nvtxEvent *nvtxDelaunay = new nvtxEvent("Delaunay", 6);

  int nEdge = connectivity->edgeTriangles->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  triangleSubstitute->SetSize(nTriangle);

  if (edgeNeedsChecking == 0)
    edgeNonDelaunay->SetSize(nEdge);
  else
    edgeNonDelaunay->SetSize(nEdgeCheck);

  triangleAffected->SetSize(2*nEdge);
  triangleAffectedEdge->SetSize(2*nEdge);

  int finished = 0;
  int nCycle = 0;
  while (!finished) {
    nvtxEvent *nvtxTemp = new nvtxEvent("CheckEdge", 1);

    if (flopFlag != 1) {
      // Check all edges for Delaunay property
      CheckEdges(connectivity, predicates, meshParameter,
                 edgeNeedsChecking, nEdgeCheck);
    } else {
      // Check all edges if can be flopped
      CheckEdgesFlop(connectivity, predicates, meshParameter,
                     edgeNeedsChecking, nEdgeCheck);
      finished = 1;
    }

    delete nvtxTemp;
    nvtxTemp = new nvtxEvent("Compact", 2);

    // Select edges that are not Delaunay (note: size of Array not changed!)
    int nNonDel = edgeNonDelaunay->RemoveValue(-1);

    delete nvtxTemp;

    if (nNonDel == 0) {
      // No more edges to flip: done
      finished = 1;
    } else {
      nvtxTemp = new nvtxEvent("Parallel", 3);

      // Find edges that can be flipped in parallel
      nNonDel = FindParallelFlipSet(connectivity, nNonDel);

      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Sub", 4);

      // Fill substitution triangles for repair step
      FillTriangleSubstitute(connectivity, nNonDel);

      // Adjust state for conservation
      if (vertexState != 0)
        AdjustState<realNeq, CL>(connectivity, vertexState, predicates,
                                 meshParameter, nNonDel);

      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Flip", 5);

      // Flip edges
      FlipEdge(connectivity, nNonDel);

      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Repair", 6);

      // Repair
      EdgeRepair(connectivity, edgeNeedsChecking, nEdgeCheck);

      delete nvtxTemp;
    }

    nCycle++;
    if (maxCycle > 0)
      if (nCycle >= maxCycle) finished = 1;
  }

  delete nvtxDelaunay;
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Delaunay::MakeDelaunay<real,
                       CL_ADVECT>(Connectivity * const connectivity,
                                  Array<real> * const vertexState,
                                  const Predicates *predicates,
                                  const MeshParameter *meshParameter,
                                  const int maxCycle,
                                  Array<int> * const edgeNeedsChecking,
                                  const int nEdgeCheck,
                                  const int flopFlag);
template void
Delaunay::MakeDelaunay<real,
                       CL_BURGERS>(Connectivity * const connectivity,
                                   Array<real> * const vertexState,
                                   const Predicates *predicates,
                                   const MeshParameter *meshParameter,
                                   const int maxCycle,
                                   Array<int> * const edgeNeedsChecking,
                                   const int nEdgeCheck,
                                   const int flopFlag);
template void
Delaunay::MakeDelaunay<real3,
                       CL_CART_ISO>(Connectivity * const connectivity,
                                    Array<real3> * const vertexState,
                                    const Predicates *predicates,
                                    const MeshParameter *meshParameter,
                                    const int maxCycle,
                                    Array<int> * const edgeNeedsChecking,
                                    const int nEdgeCheck,
                                    const int flopFlag);
template void
Delaunay::MakeDelaunay<real4,
                       CL_CART_EULER>(Connectivity * const connectivity,
                                      Array<real4> * const vertexState,
                                      const Predicates *predicates,
                                      const MeshParameter *meshParameter,
                                      const int maxCycle,
                                      Array<int> * const edgeNeedsChecking,
                                      const int nEdgeCheck,
                                      const int flopFlag);

}  // namespace astrix
