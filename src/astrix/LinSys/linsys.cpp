// -*-c++-*-
/*! \file linsys.cpp
\brief Constructor, destructor and initialization of the LinSys class*/

#include <iostream>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "../Device/device.h"

namespace astrix {

//#########################################################################
/*! Define Arrays, create Mesh and setup simulation.
  \param _verboseLevel How much information to output to stdout in Astrix.
  \param _debugLevel Level of extra checks for correct mesh.
  \param *fileName Input file name
  \param *device Device to be used for computation.
  \param restartNumber Number of saved file to restore from*/
//#########################################################################

LinSys::LinSys(int _verboseLevel,
               int _debugLevel,
               Device *_device,
               Mesh *_mesh)
{
  // How much to output to screen
  verboseLevel = _verboseLevel;
  debugLevel = _debugLevel;
  device = _device;
  mesh = _mesh;

  cudaFlag = device->GetCudaFlag();
}

// #########################################################################
// Destructor for simulation object
// #########################################################################

LinSys::~LinSys()
{
}

}
