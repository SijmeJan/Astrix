// -*-c++-*-
/*! \file Mesh/Refine/refine.cpp
\brief Functions for creating and destroying a Refine object

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
#include "./refine.h"

namespace astrix {

//#########################################################################
/*! Constructor for Refine class.

\param _cudaFlag Flag indicating whether to use device (1) or host (0)
\param _debugLevel Level of extra checks to do
\param _verboseLevel Level of output to stdout*/
//#########################################################################

Refine::Refine(int _cudaFlag, int _debugLevel, int _verboseLevel)
{
  cudaFlag = _cudaFlag;
  debugLevel = _debugLevel;
  verboseLevel = _verboseLevel;

  // Allocate Arrays of default size
  vertexCoordinatesAdd = new Array<real2>(1, cudaFlag);
  badTriangles = new Array<int>(1, cudaFlag);
  elementAdd = new Array<int>(1, cudaFlag);
  triangleAffected = new Array<int>(1, cudaFlag);
  triangleAffectedIndex = new Array<int>(1, cudaFlag);
  edgeNeedsChecking = new Array<int>(1, cudaFlag);

  randomUnique = new Array<unsigned int>(1, cudaFlag, 10000000);
  randomUnique->SetToSeries();
  randomUnique->Shuffle();
}

//#########################################################################
// Destructor
//#########################################################################

Refine::~Refine()
{
  delete vertexCoordinatesAdd;
  delete badTriangles;
  delete elementAdd;
  delete triangleAffected;
  delete triangleAffectedIndex;
  delete edgeNeedsChecking;
  delete randomUnique;
}

}  // namespace astrix
