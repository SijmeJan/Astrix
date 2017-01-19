// -*-c++-*-
/*! \file meshparameter.cpp
\brief Functions for creating MeshParameter object*/

#include <iostream>
#include <cuda_runtime_api.h>

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
  maxError = 1.0;
  minError = 0.5;
  structuredFlag = 0;

  baseResolution = -1.0;
  maxResolution = -1.0;
}

//#########################################################################
// Destructor
//#########################################################################

MeshParameter::~MeshParameter()
{
}

}
