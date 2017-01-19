// -*-c++-*-
/*! \file refine.cpp
\brief Functions for creating and destroying a Refine object*/

#include <iostream>
#include <cuda_runtime_api.h>

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

}
