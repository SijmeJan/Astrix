// -*-c++-*-
/*! \file coarsen.cpp
\brief Functions for creating and destroying a Coarsen object

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
#include "./coarsen.h"

namespace astrix {

//#########################################################################
/*! Constructor for Coarsen class.

\param _cudaFlag Flag indicating whether to use device (1) or host (0)
\param _debugLevel Level of extra checks to do
\param _verboseLevel Level of screen output*/
//#########################################################################

Coarsen::Coarsen(int _cudaFlag, int _debugLevel, int _verboseLevel)
{
  cudaFlag = _cudaFlag;
  debugLevel = _debugLevel;
  verboseLevel = _verboseLevel;

  // Allocate Arrays of default size
  vertexRemove = new Array<int>(1, cudaFlag);
  vertexTriangle = new Array<int>(1, cudaFlag);
  vertexArea = new Array<real>(1, cudaFlag);

  /*
  // Create vector of random values
  randomVector = new Array<unsigned int>(1, cudaFlag);
  if (cudaFlag == 1)
    randomVector->TransformToHost();
  randomVector->SetSize(10000000);
  randomVector->SetToRandom();
  if (cudaFlag == 1)
    randomVector->TransformToDevice();
  */
}

//#########################################################################
// Destructor
//#########################################################################

Coarsen::~Coarsen()
{
  delete vertexRemove;
  delete vertexTriangle;
  delete vertexArea;

  //delete randomVector;
}

}
