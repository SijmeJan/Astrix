// -*-c++-*-
/*! \file check.cpp
\brief Functions for checking validity of Connectivity object

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
#include "./connectivity.h"

namespace astrix {

//#########################################################################
/*! Check if the neighbouring triangles of every edge \a e do have \a e as an edge. This is done on the host
*/
//#########################################################################

void Connectivity::CheckEdgeTriangles()
{
  // Copy data to host if necessary
  if (cudaFlag == 1) CopyToHost();

  int2 *pEt = edgeTriangles->GetHostPointer();
  int3 *pTe = triangleEdges->GetHostPointer();

  int nEdge = edgeTriangles->GetSize();

  for (int i = 0; i < nEdge; i++) {
    int t1 = pEt[i].x;
    int t2 = pEt[i].y;

    if (t1 != -1) {
      int e1 = pTe[t1].x;
      int e2 = pTe[t1].y;
      int e3 = pTe[t1].z;

      if (i != e1 && i != e2 && i != e3) {
        std::cout << "Edge " << i << " has neighbouring triangle "
                  << t1 << " but triangle " << t1 << " does not have an edge "
                  << i << std::endl;
        int qq; std::cin >> qq;
      }
    }

    if (t2 != -1) {
      int e1 = pTe[t2].x;
      int e2 = pTe[t2].y;
      int e3 = pTe[t2].z;

      if (i != e1 && i != e2 && i != e3) {
        std::cout << "Edge " << i << " has neighbouring triangle "
                  << t2 << " but triangle " << t2 << " does not have an edge "
                  << i << std::endl;
        int qq; std::cin >> qq;
      }
    }
  }
}


}  // namespace astrix
