/*! \file update.cpp
\brief File containing function for updating state using residuals*/
#include <iostream>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"

namespace astrix {
  
//##############################################################################
/*! After computing all residuals we are in a position to update the state at the vertices. First we calculate the blend parameter (if using the B scheme) to combine N and LDA residuals. Then we try an update and check if this leads to an unphysical state. Wherever we find an unphysical state we force a first order update using the N-scheme.*/
//##############################################################################

void Simulation::UpdateState(real dt, int RKStep)    
{
  int transformFlag = 0;
  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexState->TransformToDevice();
      vertexStateOld->TransformToDevice();
      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleBlendFactor->TransformToDevice();
      triangleShockSensor->TransformToDevice();
   
      cudaFlag = 1;
    } else {
      vertexState->TransformToHost();
      vertexStateOld->TransformToHost();
      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleBlendFactor->TransformToHost();
      triangleShockSensor->TransformToHost();
      
      cudaFlag = 0;
    }
  }
  
  int nVertex = mesh->GetNVertex();

  // Flag whether state at vertex is unphysical
  Array<int> *vertexUnphysicalFlag = new Array<int>(1, cudaFlag, nVertex);

  // Calculate blend parameter if necessary
  if (intScheme == SCHEME_B) CalcBlend();

  if (intScheme == SCHEME_BX) CalcShockSensor();
  
  int nCycle = 0;
  int maxCycle = mesh->GetNTriangle();
  
  int failFlag = 1;
  while (failFlag > 0) {
    nCycle++;
    // Do not try indefinitely
    if (nCycle > maxCycle) {
      std::cout << "Unphysical state after " << maxCycle
		<< " cycles, exiting" << std::endl;

      if (cudaFlag == 0) {
	int *pVu = vertexUnphysicalFlag->GetHostPointer();
	const real2 *pVc = mesh->VertexCoordinatesData();
	realNeq *pVs = vertexState->GetHostPointer();
	
	for (int i = 0; i < nVertex; i++) {
	  if (pVu[i] != 0) {
	    std::cout << "Unphysical state at x = "
		      << pVc[i].x << ", y = " << pVc[i].y << std::endl;
	    real dens = pVs[i].x;
	    real momx = pVs[i].y;
	    real momy = pVs[i].z;
	    real ener = pVs[i].w;
	    real pres = (specificHeatRatio - 1.0)*
	      (ener - 0.5*(momx*momx + momy*momy)/dens);
	    
	    std::cout << pVs[i].x << " " << pVs[i].y << " "
		      << pVs[i].z << " " << pVs[i].w << " "
		      << pres << std::endl;
	    //int qq; std::cin >> qq;
	  }
	}
      }
      
      throw std::runtime_error("");
    }

    // Distribute residue over vertices
    AddResidue(dt);

    // Check for unphysical states
    FlagUnphysical(vertexUnphysicalFlag);
    
    // Check if unphysical state anywhere
    failFlag = vertexUnphysicalFlag->Maximum();
    
    if (failFlag > 0) {
      if (intScheme == SCHEME_N) {
	nCycle = maxCycle;
      } else { 
	if (verboseLevel > 1) {
	  if (nCycle == 1) std::cout << std::endl;
	  std::cout << "Found unphysical state at "
		    << vertexUnphysicalFlag->Sum()
		    << " vertices in cycle " << nCycle
		    << std::endl;
	}

	// Replace LDA residue with N residue for all unphysical states
	ReplaceLDA(vertexUnphysicalFlag, RKStep);
      
	// Return to old state so that we can update again
	vertexState->SetEqual(vertexStateOld);
      }
    }
  }
  
  delete vertexUnphysicalFlag;

  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexState->TransformToDevice();
      vertexStateOld->TransformToDevice();
      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleBlendFactor->TransformToDevice();
      triangleShockSensor->TransformToDevice();
      
      cudaFlag = 1;
    } else {
      vertexState->TransformToHost();
      vertexStateOld->TransformToHost();
      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleBlendFactor->TransformToHost();
      triangleShockSensor->TransformToHost();
      
      cudaFlag = 0;
    }
  }

}

}
