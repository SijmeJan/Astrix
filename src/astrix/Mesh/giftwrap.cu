// -*-c++-*-
/*! \file giftwrap.cu
\brief Gift-wrapping to find convex hull

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <stdexcept>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./Predicates/predicates.h"
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Delaunay/delaunay.h"
#include "./Refine/refine.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! \brief Find next coordinate of the convex hull

Brute force calculation of the next point in the convex hull. Figure out the next point so that all points are to the 'left' of the edge startIndex - index.

\param startIndex Index of previous convex hull point
\param maxIndex Size of coordinate array
\param pC Pointer to coordinate array
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector*/
//#########################################################################

int FindNextHull(int startIndex, int maxIndex, real2 *pC,
                 const Predicates *pred, real *pParam)
{
  int index = startIndex + 1;
  if (index == maxIndex) index = 0;

  real2 cStart = pC[startIndex];

  while (true) {
    real2 cCur = pC[index];

    // Test if all points to the left of edge startIndex - index
    int res = 1;
    for (int i = 0; i < maxIndex; i++)
      if (pred->orient2d(cStart.x, cStart.y,
                         cCur.x, cCur.y,
                         pC[i].x, pC[i].y, pParam) < 0) res = 0;

    if (res == 1) break;

    index++;
    if (index == maxIndex) index = 0;
  }

  return index;
}

//#########################################################################
/*! Find the convex hull of a set of points. Returns an Array containing the convex hull coordinates. These entries are removed from the input array, so that the input array now only contains internal vertices! Note that this is a host-only function.

  \param *pointCoordinates Input points. Note: convex hull coordinates will be removed!*/
//#########################################################################

Array<real2>* Mesh::ConvexHull(Array<real2> *pointCoordinates)
{

  real *pParam = predicates->GetParamPointer(0);

  int nPoint = pointCoordinates->GetSize();

  Array<unsigned int> *hullFlag = new Array<unsigned int>(1, 0, nPoint);
  hullFlag->SetToValue(nPoint);

  unsigned int *hFlag = hullFlag->GetPointer();
  real2 *pC = pointCoordinates->GetPointer();

  // Find rightmost point (must be part of convex hull)
  unsigned int startIndex = pointCoordinates->MaximumCombIndex(0);

  int next = FindNextHull(startIndex, nPoint, pC, predicates, pParam);

  int prev = startIndex;
  int mid = next;

  while (mid != startIndex) {
    mid = next;
    next = FindNextHull(next, nPoint, pC, predicates, pParam);

    if (predicates->orient2d(pC[prev].x, pC[prev].y,
                             pC[mid].x, pC[mid].y,
                             pC[next].x, pC[next].y, pParam) != 0) {
      // Add mid to convex hull
      hFlag[mid] = mid;
    }

    prev = mid;
  }

  // Sort pointCoordinates according to hullFlag
  hullFlag->SortByKey(pointCoordinates);

  // Remove all non-convex-hull entries
  int nHull = hullFlag->RemoveValue(nPoint);
  hullFlag->SetSize(nHull);

  // Now the first nHull entries in pointCoordinates make up the convex hull

  Array<real2> *cHull = new Array<real2>(1, 0, nHull);
  real2 *pcH = cHull->GetPointer();

  if (verboseLevel > 0)
    std::cout << "Convex hull coordinates: " << std::endl;

  hFlag = hullFlag->GetPointer();
  for (int i = 0; i < nHull; i++) {
    pcH[i] = pC[i];
    if (verboseLevel > 0)
      std::cout << pcH[i].x << " " << pcH[i].y << std::endl;
  }

  // Remove convex hull coordinates from list
  pointCoordinates->Remove(0, nHull);

  return cHull;
}

}
