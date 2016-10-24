// -*-c++-*-
/*! \file coarsen.cpp
\brief Top-level function for coarsening Mesh
*/
#include <iostream>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "../Common/nvtxEvent.h"
#include "./Delaunay/delaunay.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"
#include "./Coarsen/coarsen.h"

namespace astrix {
  
//#########################################################################
/*! Coarsen mesh. First calculate an estimate of the discretization error and flag triangles that can be coarsened. Then remove as many vertices as possible from mesh.

\param *vertexState Pointer to state vector at vertices
\param specificHeatRatio Ratio of specific heats
\param nTimeStep Number of time steps taken so far. Used in combination with \a nStepSkipCoarsen to possibly avoid coarsening every timestep*/
//#########################################################################

int Mesh::RemoveVertices(Array<realNeq> *vertexState,
			 real specificHeatRatio, int nTimeStep)
{
  // Return if skipping this time step
  if (nTimeStep % meshParameter->nStepSkipCoarsen != 0) return 0;

  int nTriangle = connectivity->triangleVertices->GetSize();

  triangleWantRefine->SetSize(nTriangle);
    
  // Flag triangles if refinement / coarsening is needed
  FillWantRefine(vertexState, specificHeatRatio);

  int nRemove = coarsen->RemoveVertices(connectivity,
					predicates,
					vertexState,
					specificHeatRatio,
					triangleWantRefine,
					meshParameter,
					delaunay, 1,
					randomVector);

  if (nRemove > 0) {
    CalcNormalEdge();
    CalcVertexArea();
    FindBoundaryVertices();
  }

  return nRemove;
}

}
