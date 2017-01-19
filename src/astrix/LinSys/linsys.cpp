// -*-c++-*-
/*! \file linsys.cpp
\brief Constructor, destructor and initialization of the LinSys class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

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
