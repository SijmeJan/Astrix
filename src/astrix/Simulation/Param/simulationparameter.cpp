// -*-c++-*-
/*! \file simulationparameter.cpp
\brief Functions for creating SimulationParameter object

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
#include "./simulationparameter.h"

namespace astrix {

//#########################################################################
// Constructor
//#########################################################################

SimulationParameter::SimulationParameter()
{
  // Only run in 2D for now
  nSpaceDim = 2;

  // Parameters to be read from input file, set to invalid values
  problemDef = PROBLEM_UNDEFINED;
  maxSimulationTime = -1.0;
  saveIntervalTime = -1.0;
  saveIntervalTimeFine = -1.0;
  writeVTK = -1;
  integrationOrder = -1;
  massMatrix = -1;
  selectiveLumpFlag = -1;
  intScheme = SCHEME_UNDEFINED;
  specificHeatRatio = -1.0;
  CFLnumber = -1.0;
  preferMinMaxBlend = 2;
  minError = 0.0;
  maxError = -1.0;
}

//#########################################################################
// Destructor, nothing to clean up
//#########################################################################

SimulationParameter::~SimulationParameter()
{
}

}  // namespace astrix
