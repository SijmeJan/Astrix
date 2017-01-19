/*! \file findparallelinsertion.cpp
\brief File containing function to find parallel insertion set.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Predicates/predicates.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! \brief Find independent set of insertion points

Upon return, \a elementAdd and \a vertexCoordinatesAdd are compacted to a set that can be inserted in one parallel step. This set is found by finding the cavities of the insertion points, keeping an insertion point \i only if none of the triangles in its cavity are needed by points > \a i. To optimize the set, we randomize the order by assigning a unique random number to each insertion point, and keep insertion point \a i with associated random number \a r only if none of the triangles in its cavity are needed by points with associated random number > \a r.

\param *connectivity Pointer to basic Mesh data
\param *vertexOrder The order in which vertices were inserted. All entries relating to the independent set will be removed
\param *vertexOrderInsert Will be compacted with \a elementAdd and \a vertexCoordinatesAdd
\param *vOBCoordinates Coordinates of vertices to be inserted. All entries relating to the independent set will be removed
\param *predicates Pointer to Predicates object
\param *meshParameter Pointer to Mesh parameters*/
//#########################################################################

void Refine::FindParallelInsertionSet(Connectivity * const connectivity,
                                      Array<int> * const vertexOrder,
                                      Array<int> * const vertexOrderInsert,
                                      Array<real2> * const vOBCoordinates,
                                      const Predicates *predicates,
                                      const MeshParameter *meshParameter)
{
  nvtxEvent *nvtxUnique = new nvtxEvent("unique", 3);

  // Number of triangles and number of insertion points
  unsigned int nTriangle = connectivity->triangleVertices->GetSize();
  unsigned int nRefine = elementAdd->GetSize();

  Array <int> *triangleInCavity = new Array<int>(1, cudaFlag, nTriangle);
  Array <int> *uniqueFlag = new Array<int>(1, cudaFlag, nRefine);
  Array <int> *uniqueFlagScan = new Array<int>(1, cudaFlag, nRefine);

  // Set pTriangleInCavity[n] = pRandomPermutation[i] if triangle \a t is part
  // of the cavity of point i and available (i.e. not locked by another
  // insertion point).
  LockTriangles(connectivity, predicates, meshParameter, triangleInCavity);

  // Select cavities that are independent
  FindIndependentCavities(connectivity, predicates, meshParameter,
                          triangleInCavity, uniqueFlag);

  // Compact arrays to new nRefine
  nRefine = uniqueFlag->ExclusiveScan(uniqueFlagScan, nRefine);
  elementAdd->Compact(nRefine, uniqueFlag, uniqueFlagScan);
  vertexCoordinatesAdd->Compact(nRefine, uniqueFlag, uniqueFlagScan);

  if (vertexOrder != 0) {
    // Compact insertion order
    vertexOrderInsert->Compact(nRefine, uniqueFlag, uniqueFlagScan);
    // Remove inserted vertices from list of boundary vertices to be inserted
    uniqueFlag->Invert();
    int nIgnore = uniqueFlag->ExclusiveScan(uniqueFlagScan,
                                            uniqueFlag->GetSize());
    vOBCoordinates->Compact(nIgnore, uniqueFlag, uniqueFlagScan);
    vertexOrder->Compact(nIgnore, uniqueFlag, uniqueFlagScan);
  }

  // Flag edges to be checked for Delaunay-hood later
  FlagEdgesForChecking(connectivity, predicates, meshParameter);

  delete triangleInCavity;

  delete uniqueFlag;
  delete uniqueFlagScan;

  delete nvtxUnique;
}

}
