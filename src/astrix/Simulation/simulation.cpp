// -*-c++-*-
/*! \file simulation.cpp
\brief Constructor, destructor and initialization of the Simulation class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "../Device/device.h"
#include "./simulation.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Define Arrays, create Mesh and setup simulation.
  \param _verboseLevel How much information to output to stdout in Astrix.
  \param _debugLevel Level of extra checks for correct mesh.
  \param *fileName Input file name
  \param *device Device to be used for computation.
  \param restartNumber Number of saved file to restore from*/
//#########################################################################

Simulation::Simulation(int _verboseLevel,
                       int _debugLevel,
                       char *fileName,
                       Device *_device,
                       int restartNumber)
{
  std::cout << "Setting up Astrix simulation using parameter file \'"
            << fileName << "\'" << std::endl;

  simulationParameter = new SimulationParameter;
  try {
    simulationParameter->ReadFromFile(fileName);
  }
  catch (...) {
    std::cout << "Reading " << fileName << " failed!" << std::endl;
    throw;
  }

  // How much to output to screen
  verboseLevel = _verboseLevel;
  debugLevel = _debugLevel;
  device = _device;

  cudaFlag = device->GetCudaFlag();

  try {
    // Create mesh object
    mesh = new Mesh(verboseLevel, debugLevel, cudaFlag,
                    fileName, device, restartNumber);
  }
  catch (...) {
    std::cout << "Mesh creation failed" << std::endl;
    throw;
  }

  int nSpaceDim = simulationParameter->nSpaceDim;

  nTimeStep = 0;
  nSave = 0;
  nSaveFine = 0;

  // Define arrays
  vertexState           = new Array<realNeq>(1, cudaFlag);
  vertexStateOld        = new Array<realNeq>(1, cudaFlag);
  vertexPotential       = new Array<real>(1, cudaFlag);
  vertexStateDiff       = new Array<realNeq>(1, cudaFlag);
  vertexParameterVector = new Array<realNeq>(1, cudaFlag);

  triangleResidueN  = new Array<realNeq>(nSpaceDim + 1, cudaFlag);
  triangleResidueLDA = new Array<realNeq>(nSpaceDim + 1, cudaFlag);
  triangleResidueTotal = new Array<realNeq>(1, cudaFlag);
  triangleShockSensor = new Array<real>(1, cudaFlag);
  triangleResidueSource  = new Array<realNeq>(1, cudaFlag);

  try {
    // Initialize simulation
    Init(restartNumber);
  }
  catch (...) {
    std::cout << "Simulation initialization failed" << std::endl;

    // Clean up; destructor will not be called
    delete vertexState;
    delete vertexStateOld;
    delete vertexPotential;
    delete vertexParameterVector;
    delete vertexStateDiff;

    delete triangleResidueN;
    delete triangleResidueLDA;
    delete triangleResidueTotal;
    delete triangleShockSensor;
    delete triangleResidueSource;

    delete mesh;

    throw;
  }
}

// #########################################################################
// Destructor for simulation object
// #########################################################################

Simulation::~Simulation()
{
  delete vertexState;
  delete vertexStateOld;
  delete vertexPotential;
  delete vertexParameterVector;
  delete vertexStateDiff;

  delete triangleResidueN;
  delete triangleResidueLDA;
  delete triangleResidueTotal;
  delete triangleShockSensor;
  delete triangleResidueSource;

  delete mesh;
}

// #########################################################################
/*! Set up the simulation. Allocate memory, set initial conditions, possivly restore from previous save.

  \param restartNumber Number of save file to restore from*/
// #########################################################################

void Simulation::Init(int restartNumber)
{
  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();

  // Allocate memory
  vertexState->SetSize(nVertex);
  vertexStateOld->SetSize(nVertex);
  vertexPotential->SetSize(nVertex);
  vertexStateDiff->SetSize(nVertex);
  vertexParameterVector->SetSize(nVertex);

  triangleResidueN->SetSize(nTriangle);
  triangleResidueLDA->SetSize(nTriangle);
  triangleResidueTotal->SetSize(nTriangle);
  if (simulationParameter->intScheme == SCHEME_BX)
    triangleShockSensor->SetSize(nTriangle);
  triangleResidueSource->SetSize(nTriangle);

  CalcPotential();

  if (restartNumber == 0) {
    // Start at t = 0.0
    simulationTime = 0.0;

    // Set initial conditions
    SetInitial(0.0);

    if (mesh->IsAdaptive() == 1) {
      ReplaceEnergyWithPressure();
      Coarsen(-1);
      Refine();
      Coarsen(-1);
      Refine();
      ReplacePressureWithEnergy();
    }
  } else {
    try {
      Restore(restartNumber);
    }
    catch (...) {
      std::cout << "Restoring failed!" << std::endl;
      throw;
    }
  }

  // Calculate source residual to make sure it contains sensible values
  CalcSource(vertexState);

  if (verboseLevel > 0) {
    std::cout << "Done creating simulation." << std::endl;
    std::cout << "Memory allocated on host: "
              << ((real)(Array<real>::memAllocatedHost) +
                  (real)(Array<real2>::memAllocatedHost) +
                  (real)(Array<real3>::memAllocatedHost) +
                  (real)(Array<real4>::memAllocatedHost) +
                  (real)(Array<int>::memAllocatedHost) +
                  (real)(Array<int2>::memAllocatedHost) +
                  (real)(Array<int3>::memAllocatedHost) +
                  (real)(Array<int4>::memAllocatedHost) +
                  (real)(Array<unsigned int>::memAllocatedHost))/
      (real) (1073741824) << " Gb, on device: "
              << ((real)(Array<real>::memAllocatedDevice) +
                  (real)(Array<real2>::memAllocatedDevice) +
                  (real)(Array<real3>::memAllocatedDevice) +
                  (real)(Array<real4>::memAllocatedDevice) +
                  (real)(Array<int>::memAllocatedDevice) +
                  (real)(Array<int2>::memAllocatedDevice) +
                  (real)(Array<int3>::memAllocatedDevice) +
                  (real)(Array<int4>::memAllocatedDevice) +
                  (real)(Array<unsigned int>::memAllocatedDevice))/
      (real) (1073741824) << " Gb" << std::endl;
  }
}

}  // namespace astrix
