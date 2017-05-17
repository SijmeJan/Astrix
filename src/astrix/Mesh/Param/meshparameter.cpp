// -*-c++-*-
/*! \file meshparameter.cpp
\brief Functions for creating MeshParameter object

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
#include "./meshparameter.h"

namespace astrix {

//#########################################################################
// Constructor
//#########################################################################

MeshParameter::MeshParameter()
{
  // Parameters to be read from input file, set to invalid values
  problemDef = PROBLEM_UNDEFINED;
  equivalentPointsX = -1;
  minx = 1.0e10;
  maxx = -1.0e10;
  miny = 1.0e10;
  maxy = -1.0e10;
  periodicFlagX = -1;
  periodicFlagY = -1;
  adaptiveMeshFlag = -1;
  maxRefineFactor = -1;
  qualityBound = 0.0;
  nStepSkipRefine = -1;
  nStepSkipCoarsen = -1;
  structuredFlag = 0;

  baseResolution = -1.0;
  maxResolution = -1.0;
}

//#########################################################################
// Destructor, nothing to clean up
//#########################################################################

MeshParameter::~MeshParameter()
{
}

}  // namespace astrix
