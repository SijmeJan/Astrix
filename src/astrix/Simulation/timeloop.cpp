// -*-c++-*-
/*! \file timeloop.cpp
\brief File containing functions to run a simulation.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <fstream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/nvtxEvent.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Run simulation, possibly restarting from saved state, for a maximum
amount of wall clock hours.
  \param maxWallClockHours Maximum amount of wall clock hours to run for.*/
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::Run(real maxWallClockHours)
{
  int warning = 0;
  time_t startTime = time(NULL);
  double elapsedTimeHours = difftime(time(NULL), startTime)/3600.0;

  try {
    // Save state at t=0
    if (nSave == 0) {
      Save();
      FineGrainSave();
      nSave++;
      nSaveFine++;
    }
  }
  catch (...) {
    std::cout << "Error saving " << nSave << std::endl;
    throw;
  }

  if (verboseLevel > 0)
    std::cout << "Starting time loop... " << nSave << std::endl;

  while (warning == 0 &&
         simulationTime < simulationParameter->maxSimulationTime &&
         elapsedTimeHours < maxWallClockHours) {
    try {
      // Do one time step
      DoTimeStep();
    }
    catch (...) {
      std::cout << "Error after DoTimeStep()" << std::endl;
      nSave = 999;
      Save();

      throw;
    }

    try {
      // Save every saveFineInterval
      if (simulationTime >
          (real) nSaveFine*simulationParameter->saveIntervalTimeFine) {
        FineGrainSave();
        nSaveFine++;
      }

      // Save every saveInterval
      if (simulationTime >
          (real) nSave*simulationParameter->saveIntervalTime) {
        Save();
        nSave++;
      }
    }
    catch (...) {
      std::cout << "Save failed!" << std::endl;
      throw;
    }

    elapsedTimeHours = difftime(time(NULL), startTime)/3600.0;
  }

  try {
    // Save if end of simulation reached
    if (warning == 0 &&
        elapsedTimeHours < maxWallClockHours) {
      Save();
      nSave++;
      FineGrainSave();
      nSaveFine++;
    }
  }
  catch (...) {
    std::cout << "Save failed!" << std::endl;
    throw;
  }

}

//#########################################################################
/*! Do a single time step. Update mesh, calculate time step, and update
  state. */
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::DoTimeStep()
{
  ProblemDefinition problemDef = simulationParameter->problemDef;

  // Number of time steps taken
  nTimeStep++;

  if (verboseLevel > 0)
    std::cout << std::setprecision(12)
              << "Starting time step " << nTimeStep << ", ";

  // Refine / coarsen mesh
  if (mesh->IsAdaptive() == 1) {
    ReplaceEnergyWithPressure();

    try {
      Coarsen(-1);
    }
    catch (...) {
      std::cout << "Error coarsening in DoTimeStep()" << std::endl;
      throw;
    }

    try {
      Refine();
    }
    catch (...) {
      std::cout << "Error refining in DoTimeStep()" << std::endl;
      throw;
    }

    ReplacePressureWithEnergy();
  }

  nvtxEvent *nvtxHydro = new nvtxEvent("Hydro", 2);

  // Calculate time step
  real dt = CalcVertexTimeStep();

  /*
  if (problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_SOD)
    ExtrapolateBoundaries();
  */

  // Boundary conditions for 2D Riemann
  if (problemDef == PROBLEM_RIEMANN)
    SetRiemannBoundaries();

  // Set exact inflow boundaries
  if (problemDef == PROBLEM_SOURCE)
    SetInitial(simulationTime, 1);

  // Boundary conditions for 2D Noh
  if (problemDef == PROBLEM_NOH)
    SetNohBoundaries();

  // Set Wold = W
  vertexStateOld->SetEqual(vertexState);

  // Calculate source term
  if (problemDef == PROBLEM_SOURCE ||
      problemDef == PROBLEM_DISC)
    CalcSource(vertexState);

  // Calculate parameter vector Z at nodes
  CalculateParameterVector(0);

  // Calculate (space) residuals at triangles
  CalcResidual();

  // Update state at vertices
  try {
    UpdateState(dt, 0);
  }
  catch (...) {
    std::cout << "Updating state RK1 failed!" << std::endl;
    throw;
  }

  // Reflecting boundaries
  if (problemDef == PROBLEM_CYL ||
      problemDef == PROBLEM_SOD ||
      problemDef == PROBLEM_BLAST)
    ReflectingBoundaries(dt);

  if (problemDef == PROBLEM_RIEMANN)
    SetSymmetricBoundaries();

  // Nonreflecting boundaries
  if (problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_DISC)
    SetNonReflectingBoundaries();

  if (simulationParameter->integrationOrder == 2) {
    /*
    if (problemDef == PROBLEM_VORTEX ||
        problemDef == PROBLEM_SOD)
      ExtrapolateBoundaries();
    */

    // Boundary conditions for 2D Riemann
    if (problemDef == PROBLEM_RIEMANN)
      SetRiemannBoundaries();

    // Exact inflow boundaries
    if (problemDef == PROBLEM_SOURCE)
      SetInitial(simulationTime + dt, 1);

    // Boundary conditions for 2D Noh
    if (problemDef == PROBLEM_NOH)
      SetNohBoundaries();

    // Calculate source term
    if (problemDef == PROBLEM_SOURCE ||
        problemDef == PROBLEM_DISC)
      CalcSource(vertexState);

    // Calculate parameter vector Z at nodes
    CalculateParameterVector(0);

    // dW = W - Wold
    vertexStateDiff->SetToDiff(vertexState, vertexStateOld);

    // Calculate space-time residual N + total
    CalcTotalResNtot(dt);

    // Calculate parameter vector Z at nodes from old state
    CalculateParameterVector(1);

    int massMatrix = simulationParameter->massMatrix;
    int selectiveLumpFlag = simulationParameter->selectiveLumpFlag;

    if (massMatrix == 3 || massMatrix == 4)
      MassMatrixF34Tot(dt, massMatrix);

    // Calculate space-time residual LDA
    CalcTotalResLDA();

    if (massMatrix == 3 || massMatrix == 4)
      MassMatrixF34(dt, massMatrix);

    if (selectiveLumpFlag == 1 || massMatrix == 2)
      SelectLump(dt, massMatrix, selectiveLumpFlag);

    // Set Wold = W
    vertexStateOld->SetEqual(vertexState);

    // Update state at vertices
    try {
      UpdateState(dt, 1);
    }
    catch (...) {
      std::cout << "Updating state RK2 failed!" << std::endl;
      throw;
    }

    // Reflecting boundaries
    if (problemDef == PROBLEM_CYL ||
        problemDef == PROBLEM_SOD ||
        problemDef == PROBLEM_BLAST)
      ReflectingBoundaries(dt);

    if (problemDef == PROBLEM_RIEMANN)
      SetSymmetricBoundaries();

    // Nonreflecting boundaries
    if (problemDef == PROBLEM_VORTEX ||
        problemDef == PROBLEM_DISC)
      SetNonReflectingBoundaries();
  }

  //auto finish = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double> elapsed = finish - start;

  if (verboseLevel > 0) {
    std::cout << std::setprecision(6)
              << "t = " << simulationTime << " dt = " << dt << " ";
      //<< elapsed.count() << " ";
    if (cudaFlag == 0) {
      std::cout << ((real)(Array<real>::memAllocatedHost) +
                    (real)(Array<real2>::memAllocatedHost) +
                    (real)(Array<real3>::memAllocatedHost) +
                    (real)(Array<real4>::memAllocatedHost) +
                    (real)(Array<int>::memAllocatedHost) +
                    (real)(Array<int2>::memAllocatedHost) +
                    (real)(Array<int3>::memAllocatedHost) +
                    (real)(Array<int4>::memAllocatedHost) +
                    (real)(Array<unsigned int>::memAllocatedHost))/
        (real) (1073741824) << " Gb"
                << std::endl;
    } else {
      std::cout << ((real)(Array<real>::memAllocatedDevice) +
                    (real)(Array<real2>::memAllocatedDevice) +
                    (real)(Array<real3>::memAllocatedDevice) +
                    (real)(Array<real4>::memAllocatedDevice) +
                    (real)(Array<int>::memAllocatedDevice) +
                    (real)(Array<int2>::memAllocatedDevice) +
                    (real)(Array<int3>::memAllocatedDevice) +
                    (real)(Array<int4>::memAllocatedDevice) +
                    (real)(Array<unsigned int>::memAllocatedDevice))/
        (real) (1073741824)<< " Gb"
                << std::endl;
    }
  }

  // Increase time
  simulationTime += dt;

  delete nvtxHydro;
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::Run(real maxWallClockHours);
template void Simulation<real, CL_BURGERS>::Run(real maxWallClockHours);
template void Simulation<real3, CL_CART_ISO>::Run(real maxWallClockHours);
template void Simulation<real3, CL_CYL_ISO>::Run(real maxWallClockHours);
template void Simulation<real4, CL_CART_EULER>::Run(real maxWallClockHours);

}  // namespace astrix
