// -*-c++-*-
/*! \file morton.cpp
\brief Functions for creating and destroying a Morton object

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
#include "./morton.h"

namespace astrix {

//#########################################################################
/*! Constructor for Morton object. Allocates Arrays of standard size.

  \param _cudaFlag Flag whether to use device (1) or host (0) to compute*/
//#########################################################################

Morton::Morton(int _cudaFlag)
{
  cudaFlag = _cudaFlag;

  mortValues = new Array<unsigned int>(1, cudaFlag);
  index = new Array<unsigned int>(1, cudaFlag);
  inverseIndex = new Array<unsigned int>(1, cudaFlag);
  vertexMorton = new Array<unsigned int>(1, cudaFlag);
}

//#########################################################################
// Destructor
//#########################################################################

Morton::~Morton()
{
  delete mortValues;
  delete index;
  delete inverseIndex;
  delete vertexMorton;
}

}
