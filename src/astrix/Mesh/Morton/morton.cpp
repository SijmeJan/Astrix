// -*-c++-*-
/*! \file morton.cpp
\brief Functions for creating and destroying a Morton object*/

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
