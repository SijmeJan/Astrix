// -*-c++-*-
/*! \file timeloop.cpp
\brief File containing functions to run a simulation.*/
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/nvtxEvent.h"

namespace astrix {
  
//#########################################################################
/*! Run simulation, possibly restarting from saved state, for a maximum 
amount of wall clock hours.
  \param restartNumber Previous saved state to start from. If <= 0, start 
new simulation.
  \param maxWallClockHours Maximum amount of wall clock hours to run for.*/
//#########################################################################

void Simulation::Run(int restartNumber, real maxWallClockHours)
{
  int warning = 0;
  time_t startTime = time(NULL);
  double elapsedTimeHours = difftime(time(NULL), startTime)/3600.0;
 
  // Number of saves so far
  int nSave = 0, nSaveFine = 0;
  // Restore state if necessary
  if (restartNumber > 0) {
    try {
      nSave = Restore(restartNumber);
    }
    catch (...) {
      std::cout << "Error restoring state" << std::endl;
      throw;
    }
  }
  
  // Save state at t=0
  if (nSave == 0) {
    Save(nSave);
    FineGrainSave(nSave);
    nSave++;
  }

  return;
  
  if (verboseLevel > 0)
    std::cout << "Starting time loop..." << std::endl;

  while (warning == 0 &&
	 simulationTime < maxSimulationTime &&
	 elapsedTimeHours < maxWallClockHours) {
    try {
      // Do one time step
      DoTimeStep();
    }
    catch (...) {
      std::cout << "Error after DoTimeStep()" << std::endl;
      Save(999);

      throw;
    }

    // Save every saveFineInterval
    if (simulationTime > (real) nSaveFine*saveIntervalTimeFine) {
      FineGrainSave(nSave);
      nSaveFine++;
    }

    // Save every saveInterval
    if (simulationTime > (real) nSave*saveIntervalTime) {
      Save(nSave);
      nSave++;
    }

    elapsedTimeHours = difftime(time(NULL), startTime)/3600.0;
  }

  // Save if end of simulation reached
  if (warning == 0 &&
      elapsedTimeHours < maxWallClockHours) {
    Save(nSave);
    nSave++;
  }

  /*
  if (problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_YEE ||
      problemDef == PROBLEM_LINEAR ||
      problemDef == PROBLEM_ADVECT) {
    int nVertex = mesh->GetNVertex();

    vertexStateOld->SetEqual(vertexState);
    SetInitial();

    // Now vertexState contains initial state, vertexOld final state
    if (cudaFlag == 1) {
      vertexState->CopyFromDevice();
      vertexStateOld->CopyFromDevice();
      mesh->vertexArea->CopyFromDevice();
      mesh->vertexCoordinates->CopyFromDevice();
    }

    real4 *state = vertexStateOld->GetHostPointer();
    real4 *stateOld = vertexStateOld->GetHostPointer();
    
    real *vArea = mesh->vertexArea->GetHostPointer();
    
    real L1dens = 0.0;
    real totalArea = 0.0;
    for (int i = 0; i < nVertex; i++) {
      L1dens += fabs(state[i].x - stateOld[i].x)*vArea[i];
      totalArea += vArea[i];
    }
    
    L1dens = L1dens/totalArea;
    
    std::cout << "L1 error in density: " << L1dens << " " << std::endl;
  }
  */
}

//#########################################################################
/*! Do a single time step. Update mesh, calculate time step, and update
  state. */
//#########################################################################

void Simulation::DoTimeStep()
{
  // Number of time steps taken
  nTimeStep++;

  if (verboseLevel > 0)
    std::cout << std::setprecision(12) 
	      << "Starting time step " << nTimeStep << ", ";

  //std::cout << "Mass: " << TotalMass() << " ";
  
  // Refine / coarsen mesh  
  /*
  if (mesh->adaptiveMeshFlag == 1) {
    
    ReplaceEnergyWithPressure();
    Coarsen(-1);
    std::cout << "Mass: " << TotalMass() << " ";
    try {
      Refine();
    }
    catch (...) {
      std::cout << "Error refining in DoTimeStep()" << std::endl;
      throw;
    }

    ReplacePressureWithEnergy();

    //std::cout << "Mass: " << TotalMass() << " ";
  }
  */
  
  nvtxEvent *nvtxHydro = new nvtxEvent("Hydro", 2);

  // Calculate time step
  real dt = CalcVertexTimeStep();

  
  /*
  if (problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_YEE ||
      problemDef == PROBLEM_SOD)
    ExtrapolateBoundaries();
  */
  
  // Boundary conditions for 2D Riemann
  if (problemDef == PROBLEM_RIEMANN)
    SetRiemannBoundaries();

  /*
  if (problemDef == PROBLEM_RIEMANN && mesh->structuredFlag == 1) {
    std::cout << "Asymmetry: ";
    CheckSymmetry();
  }
  */
  
  // Set Wold = W
  vertexStateOld->SetEqual(vertexState);
   
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
  if (problemDef == PROBLEM_RT ||
      problemDef == PROBLEM_SOD ||
      problemDef == PROBLEM_BLAST)
    ReflectingBoundaries(dt);

  // Nonreflecting boundaries
  //if(problemDef == PROBLEM_VORTEX)
  //SetNonReflectingBoundaries();
 
  if (integrationOrder == 2) {
    /*
    if (problemDef == PROBLEM_VORTEX ||
	problemDef == PROBLEM_YEE ||
	problemDef == PROBLEM_SOD)
      ExtrapolateBoundaries();
    */
    
    // Boundary conditions for 2D Riemann
    if (problemDef == PROBLEM_RIEMANN)
      SetRiemannBoundaries();
    
    // Calculate parameter vector Z at nodes
    CalculateParameterVector(0);

    // dW = W - Wold
    vertexStateDiff->SetToDiff(vertexState, vertexStateOld);
        
    // Calculate space-time residual N + total
    CalcTotalResNtot(dt);
        
    // Calculate parameter vector Z at nodes from old state
    CalculateParameterVector(1);
    
    // Calculate space-time residual LDA
    CalcTotalResLDA();
        
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
    if (problemDef == PROBLEM_RT ||
    	problemDef == PROBLEM_SOD ||
    	problemDef == PROBLEM_BLAST)
      ReflectingBoundaries(dt);
    
    // Nonreflecting boundaries
    //if(problemDef == PROBLEM_VORTEX)
    //SetNonReflectingBoundaries();
  }

  if (verboseLevel > 0) {
    std::cout << std::setprecision(6) 
	      << "t = " << simulationTime << " dt = " << dt << " ";
    if (cudaFlag == 0) {
      std::cout << ((real)(Array<real>::memAllocatedHost) +
		    (real)(Array<int>::memAllocatedHost) +
		    (real)(Array<unsigned int>::memAllocatedHost))/
	(real) (1073741824) << " Gb"
		<< std::endl;
    } else {
      std::cout << ((real)(Array<real>::memAllocatedDevice) +
		    (real)(Array<int>::memAllocatedDevice) +
		    (real)(Array<unsigned int>::memAllocatedDevice))/
	(real) (1073741824) << " Gb"
		<< std::endl;
    }
  }
  
  // Increase time
  simulationTime += dt;

  delete nvtxHydro;
}

}
