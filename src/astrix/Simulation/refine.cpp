// -*-c++-*-
/*! \file refine.cpp
\brief File containing functions to refine / coarsen simulation.*/

#include <iostream>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"

namespace astrix {

//##############################################################################
/*! Keep refining the mesh according to an estimate of the truncation error until no more refinement is needed*/
//##############################################################################

void Simulation::Refine()
{
  int ret = 0;
  try {
    ret = mesh->ImproveQuality(vertexState,
			       specificHeatRatio,
			       nTimeStep);
  }
  catch (...) {
    std::cout << "Error refining mesh" << std::endl;
    throw;
  }
 
  if (ret > 0) {
    int nCycle = 0;

    // Reset initial condition (override interpolation done in mesh->Refine)
    if (simulationTime == 0.0) {
      int nVertex   = mesh->GetNVertex();
      vertexPotential->SetSize(nVertex);
      CalcPotential();
      SetInitial();
      ReplaceEnergyWithPressure();
    }
    
    while(ret > 0) {
      try {
	ret = mesh->ImproveQuality(vertexState,
				   specificHeatRatio,
				   nTimeStep);
      }
      catch (...) {
	std::cout << "Error refining mesh" << std::endl;
	throw;
      }

      // Reset initial condition (override interpolation done in mesh->Refine
      if (simulationTime == 0.0) {
	int nVertex = mesh->GetNVertex();
	vertexPotential->SetSize(nVertex);
	CalcPotential();
	SetInitial();
	ReplaceEnergyWithPressure();
      }
      nCycle++;
    }
    
    int nVertex   = mesh->GetNVertex();
    int nTriangle = mesh->GetNTriangle();
    
    // Adjust memory allocation
    vertexStateOld->SetSize(nVertex);
    vertexPotential->SetSize(nVertex);
    vertexParameterVector->SetSize(nVertex);
    vertexStateDiff->SetSize(nVertex);
    
    triangleResidueN->SetSize(nTriangle);
    triangleResidueLDA->SetSize(nTriangle);
    triangleResidueTotal->SetSize(nTriangle);
    if (intScheme == SCHEME_B)
      triangleBlendFactor->SetSize(nTriangle);
    if (intScheme == SCHEME_BX)
      triangleShockSensor->SetSize(nTriangle);

    CalcPotential();
  }
}

//##############################################################################
/*! Coarsen the mesh according to an estimate of the truncation error
\param maxCycle Maximum number of coarsening cycles. If < 0, coarsen until no more triangles to do*/
//##############################################################################

void Simulation::Coarsen(int maxCycle)
{
  int nCycle = 0;
  int finishedFlag = 0;
  while (finishedFlag == 0) {
    if(mesh->RemoveVertices(vertexState, specificHeatRatio, nTimeStep) == 0)
      //if (mesh->Coarsen(vertexState, specificHeatRatio, nTimeStep) == 0)
      finishedFlag = 1;
    nCycle++;
    if (nCycle >= maxCycle && maxCycle >= 0)
      finishedFlag = 1;

    if (simulationTime == 0.0) {
      int nVertex = mesh->GetNVertex();
      vertexPotential->SetSize(nVertex);
      CalcPotential();
      SetInitial();
      ReplaceEnergyWithPressure();
    }
  }
  
  // Adjust memory allocation
  int nVertex   = mesh->GetNVertex();
  vertexStateOld->SetSize(nVertex);
  vertexParameterVector->SetSize(nVertex);
  vertexStateDiff->SetSize(nVertex);
  vertexPotential->SetSize(nVertex);
  CalcPotential();
  
  int nTriangle = mesh->GetNTriangle();
  triangleResidueN->SetSize(nTriangle);
  triangleResidueLDA->SetSize(nTriangle);
  triangleResidueTotal->SetSize(nTriangle);
  if (intScheme == SCHEME_B)
    triangleBlendFactor->SetSize(nTriangle);
  if (intScheme == SCHEME_BX)
    triangleShockSensor->SetSize(nTriangle);
}
  
}
