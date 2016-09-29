// -*-c++-*-
/*! \file timeloop.cpp
\brief File containing functions to run a simulation.*/
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <fstream>

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

  if (problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_YEE ||
      problemDef == PROBLEM_LINEAR ||
      problemDef == PROBLEM_ADVECT) {
    int nVertex = mesh->GetNVertex();

    vertexStateOld->SetEqual(vertexState);
    SetInitial();

    // Now vertexState contains initial state, vertexOld final state
    if (cudaFlag == 1) {
      mesh->Transform();
      vertexState->CopyToHost();
      vertexStateOld->CopyToHost();
    }

    realNeq *state = vertexState->GetHostPointer();
    realNeq *stateOld = vertexStateOld->GetHostPointer();

#if N_EQUATION == 1 
    if (problemDef == PROBLEM_ADVECT) {
      const real2 *pVc = mesh->VertexCoordinatesData();
      for (int i = 0; i < nVertex; i++) {
	real x = pVc[i].x;
	real y = pVc[i].y;
	real r = sqrt((x - 1.5)*(x - 1.5) + (y - 0.5)*(y - 0.5));
	
	state[i] = 1.0;
	if (r <= (real) 0.25)
	  state[i] = state[i] + cos(2.0*M_PI*r)*cos(2.0*M_PI*r);
      }
    }
#endif
#if N_EQUATION == 4
    if (problemDef == PROBLEM_YEE) {
      const real2 *pVc = mesh->VertexCoordinatesData();
      for (int i = 0; i < nVertex; i++) {
	real xc = (real) 15.0;
	real yc = (real) 5.0;

	real beta = (real) 5.0;
	real G = specificHeatRatio;
	
	real x = pVc[i].x;
	real y = pVc[i].y;
	real r = sqrt((x - xc)*(x - xc) + (y - yc)*(y - yc)) + (real) 1.0e-10;

	real T = 1.0 - (G - 1.0)*beta*beta*exp(1.0 - r*r)/
	  ((real)8.0*G*M_PI*M_PI);

	state[i].x = std::pow(T, (real)((real) 1.0/(G - (real) 1.0)));
      }
    }
#endif
   
    const real *vertArea = mesh->VertexAreaData();
  
    real L1dens = 0.0;
    real L2dens = 0.0;
    real Lmaxdens = 0.0;
    real totalArea = 0.0;
    for (int i = 0; i < nVertex; i++) {
#if N_EQUATION == 1
      real relDiff = (state[i] - stateOld[i])/stateOld[i];
#endif
#if N_EQUATION == 4
      real relDiff = (state[i].x - stateOld[i].x)/stateOld[i].x;
#endif
      L1dens += fabs(relDiff)*vertArea[i];
      L2dens += relDiff*relDiff*vertArea[i];
      if (fabs(relDiff) > Lmaxdens)
	Lmaxdens = fabs(relDiff);
      totalArea += vertArea[i];
    }
    
    L1dens = L1dens/totalArea;
    L2dens = sqrt(L2dens/totalArea);
    
    std::cout << "L1 error in density: " << L1dens << " " << std::endl;
    std::cout << "L2 error in density: " << L2dens << " " << std::endl;
    std::cout << "Lmax error in density: " << Lmaxdens << " " << std::endl;
    std::cout << "Mesh size paramenter: " << sqrt(totalArea/(real)nVertex)
	      << std::endl;

    std::ofstream uitvoer;
    if (extraFlag == 0 || extraFlag == 32)
      uitvoer.open("resolution.txt");
    else
      uitvoer.open("resolution.txt", std::ios::app);
    uitvoer << extraFlag << " "
	    << std::scientific << std::setprecision(2)
	    << sqrt(totalArea/(real)nVertex) << " "
	    << L1dens << " "
	    << L2dens << " "
	    << Lmaxdens << std::endl;
    uitvoer.close();
    
    if (cudaFlag == 1) 
      mesh->Transform();

    for (int i = 0; i < nVertex; i++) {
#if N_EQUATION == 1
      real relDiff = (state[i] - stateOld[i])/stateOld[i];
      state[i] = fabs(relDiff);
#endif
#if N_EQUATION == 4
      real relDiff = (state[i].x - stateOld[i].x)/stateOld[i].x;
      state[i].x = fabs(relDiff);
#endif

    }
    Save(nSave);
    nSave++;

  }
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

  
  //if (problemDef == PROBLEM_VORTEX ||
  //  problemDef == PROBLEM_YEE ||
  //  problemDef == PROBLEM_SOD)
  //ExtrapolateBoundaries();
  
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

  /*
  int nVertex = mesh->GetNVertex();
  real *state = triangleResidueTotal->GetHostPointer();
  for (int i = 0; i < mesh->GetNTriangle(); i++) 
    if (state[i] < 0.0) std::cout << "Hallo" << std::endl;
  */
  
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
  if(problemDef == PROBLEM_YEE)
    SetNonReflectingBoundaries();
 
  
  if (integrationOrder == 2) {
    
    //if (problemDef == PROBLEM_VORTEX ||
    //	problemDef == PROBLEM_YEE ||
    //	problemDef == PROBLEM_SOD)
    //ExtrapolateBoundaries();
    
    
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
    if (problemDef == PROBLEM_RT ||
    	problemDef == PROBLEM_SOD ||
    	problemDef == PROBLEM_BLAST)
      ReflectingBoundaries(dt);
    
    // Nonreflecting boundaries
    if(problemDef == PROBLEM_YEE)
      SetNonReflectingBoundaries();
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
