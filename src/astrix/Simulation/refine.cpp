// -*-c++-*-
/*! \file Simulation/refine.cpp
\brief File containing functions to refine / coarsen simulation.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "./Param/simulationparameter.h"
#include "../Common/state.h"

namespace astrix {

//##############################################################################
/*! Keep refining the mesh according to an estimate of the truncation error
until no more refinement is needed*/
//##############################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::Refine()
{
  int ret = 0;

  FillWantRefine();

  try {
    ret = mesh->ImproveQuality<realNeq>(vertexState, nTimeStep,
                                        triangleWantRefine);
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
      vertexSoundSpeed->SetSize(nVertex);
      CalcPotential();
      SetInitial(0.0, 0);
      ReplaceEnergyWithPressure();
    }


    while (ret > 0) {
      FillWantRefine();

      try {
       ret = mesh->ImproveQuality<realNeq>(vertexState, nTimeStep,
                                            triangleWantRefine);
      }
      catch (...) {
        std::cout << "Error refining mesh" << std::endl;
        throw;
      }

      // Reset initial condition (override interpolation done in mesh->Refine
      if (simulationTime == 0.0) {
        int nVertex = mesh->GetNVertex();
        vertexPotential->SetSize(nVertex);
        vertexSoundSpeed->SetSize(nVertex);
        CalcPotential();
        SetInitial(0.0, 0);
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
    vertexSource->SetSize(nVertex);

    triangleResidueN->SetSize(nTriangle);
    triangleResidueLDA->SetSize(nTriangle);
    triangleResidueTotal->SetSize(nTriangle);
    if (simulationParameter->intScheme == SCHEME_BX)
      triangleShockSensor->SetSize(nTriangle);
    triangleResidueSource->SetSize(nTriangle);

    CalcPotential();
    // Calculate source residual to make sure it contains sensible values
    CalcSource(vertexState);
  }
}

//##############################################################################
/*! Coarsen the mesh according to an estimate of the truncation error
\param maxCycle Maximum number of coarsening cycles. If < 0, coarsen until no more triangles to do*/
//##############################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::Coarsen(int maxCycle)
{
  int nCycle = 0;
  int finishedFlag = 0;


  while (finishedFlag == 0) {
    FillWantRefine();

    try {
      if (mesh->RemoveVertices<realNeq>(vertexState, nTimeStep,
                                        triangleWantRefine) == 0)
        finishedFlag = 1;
    }
    catch (...) {
      std::cout << "Error in coarsening step" << std::endl;
      throw;
    }

    nCycle++;
    if (nCycle >= maxCycle && maxCycle >= 0)
      finishedFlag = 1;

    if (simulationTime == 0.0) {
      int nVertex = mesh->GetNVertex();
      vertexPotential->SetSize(nVertex);
      CalcPotential();
      SetInitial(0.0, 0);
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
  if (simulationParameter->intScheme == SCHEME_BX)
    triangleShockSensor->SetSize(nTriangle);
  triangleResidueSource->SetSize(nTriangle);

  // Calculate source residual to make sure it contains sensible values
  CalcSource(vertexState);
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::Refine();
template void Simulation<real, CL_BURGERS>::Refine();
template void Simulation<real3, CL_CART_ISO>::Refine();
template void Simulation<real3, CL_CYL_ISO>::Refine();
template void Simulation<real4, CL_CART_EULER>::Refine();

//##############################################################################

template void Simulation<real, CL_ADVECT>::Coarsen(int maxCycle);
template void Simulation<real, CL_BURGERS>::Coarsen(int maxCycle);
template void Simulation<real3, CL_CART_ISO>::Coarsen(int maxCycle);
template void Simulation<real3, CL_CYL_ISO>::Coarsen(int maxCycle);
template void Simulation<real4, CL_CART_EULER>::Coarsen(int maxCycle);

}  // namespace astrix
