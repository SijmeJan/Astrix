// -*-c++-*-
/*! \file findparalleldeletion.cpp
\brief File containing function to find parallel deletion set.

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
#include "./coarsen.h"
#include "../Connectivity/connectivity.h"


namespace astrix {

//#############################################################################
/*! Find set of vertices that can be removed in parallel. First, we create a list of triangles that will be affected by vertex removal. Then, we find the unique values and compact Arrays \a vertexRemove and \a vertexTriangle

\param maxTriPerVert Maximum number of triangles sharing a vertex*/
//#############################################################################

void Coarsen::FindParallelDeletionSet(Connectivity *connectivity)
{
  // Number of triangles and number of insertion points
  unsigned int nTriangle = connectivity->triangleVertices->GetSize();
  unsigned int nRemove = vertexRemove->GetSize();

  Array <int> *triangleLock = new Array<int>(1, cudaFlag, nTriangle);
  Array <int> *uniqueFlag = new Array<int>(1, cudaFlag, nRemove);
  Array <int> *uniqueFlagScan = new Array<int>(1, cudaFlag, nRemove);

  LockTriangles(connectivity, triangleLock);

  // Select cavities that are independent
  FindIndependent(connectivity, triangleLock, uniqueFlag);

  // Compact arrays to new nRefine
  nRemove = uniqueFlag->ExclusiveScan(uniqueFlagScan, nRemove);
  vertexRemove->Compact(nRemove, uniqueFlag, uniqueFlagScan);
  vertexTriangle->Compact(nRemove, uniqueFlag, uniqueFlagScan);
  triangleTarget->Compact(nRemove, uniqueFlag, uniqueFlagScan);

  // Flag edges to be checked for Delaunay-hood later
  //FlagEdgesForChecking(connectivity, predicates, meshParameter);

  delete triangleLock;

  delete uniqueFlag;
  delete uniqueFlagScan;
}

}
