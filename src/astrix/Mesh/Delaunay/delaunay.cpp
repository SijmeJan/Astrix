// -*-c++-*-
/*! \file delaunay.cpp
\brief Functions for creating and destroying a Delaunay object*/

#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./delaunay.h"

namespace astrix {
  
//#########################################################################
/*! Constructor for Delaunay class.

\param _cudaFlag Flag indicating whether to use device (1) or host (0)
\param _debugLevel Level of extra checks to do*/
//#########################################################################

Delaunay::Delaunay(int _cudaFlag, int _debugLevel)
{
  cudaFlag = _cudaFlag;
  debugLevel = _debugLevel;

  // Allocate Arrays of large size
  edgeNonDelaunay = new Array<int>(1, cudaFlag, 0, 128*8192);
  triangleSubstitute = new Array<int>(1, cudaFlag, 0, 128*8192);
  vertexArea = new Array<real>(1, cudaFlag);

  triangleAffected = new Array<int>(1, cudaFlag, 0, 128*8192);
  triangleAffectedEdge = new Array<int>(1, cudaFlag, 0, 128*8192);
}

//#########################################################################
// Destructor
//#########################################################################

Delaunay::~Delaunay()
{
  delete edgeNonDelaunay;
  delete triangleSubstitute;
  delete vertexArea;

  delete triangleAffected;
  delete triangleAffectedEdge;
}
  
}
