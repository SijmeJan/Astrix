// -*-c++-*-
/*! \file recoversegments.cpp
\brief Perform edge flips to recover segments.

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
#include "./delaunay.h"
#include "../Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! When the outer mesh boundary is complicated, extra care is needed to make sure all necessary segments are part of the triangulation. Assuming the boundary vertices were inserted in counterclockwise order, every consecutive vertex needs to be connected. Edge flips are performed until this is the case.

\param *connectivity Pointer to basic Mesh data
\param *predicates Pointer to Predicates object, used to check Delaunay property without roundoff error
\param *meshParameter Pointer to Mesh parameters
\param *vertexOrder Pointer to Array containing vertex insertion order*/
//#########################################################################

void Delaunay::RecoverSegments(Connectivity * const connectivity,
                               const Predicates *predicates,
                               const MeshParameter *meshParameter,
                               Array<int> * const vertexOrder)
{
  int nEdge = connectivity->edgeTriangles->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  triangleSubstitute->SetSize(nTriangle);

  edgeNonDelaunay->SetSize(nEdge);

  triangleAffected->SetSize(2*nEdge);
  triangleAffectedEdge->SetSize(2*nEdge);

  int finished = 0;
  int nCycle = 0;
  while (!finished) {
    // Check all edges for missing segments
    CheckSegments(connectivity, predicates, meshParameter, vertexOrder);

    // Select edges that are not Delaunay (note: size of Array not changed!)
    int nNonDel = edgeNonDelaunay->RemoveValue(-1);

    if (nNonDel == 0) {
      // No more edges to flip: done
      finished = 1;
    } else {
      // Find edges that can be flipped in parallel
      nNonDel = FindParallelFlipSet(connectivity, nNonDel);

      // Fill substitution triangles for repair step
      FillTriangleSubstitute(connectivity, nNonDel);

      // Flip edges
      FlipEdge(connectivity, nNonDel);

      // Repair
      EdgeRepair(connectivity, 0, 0);

    }

    // Don't go on forever...
    nCycle++;
    if (nCycle >= 100) finished = 1;
  }
}

}  // namespace astrix
