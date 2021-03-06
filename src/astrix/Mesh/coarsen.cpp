// -*-c++-*-
/*! \file coarsen.cpp
\brief Top-level function for coarsening Mesh

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include <iostream>

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

template<class realNeq, ConservationLaw CL>
int Mesh::RemoveVertices(Array<realNeq> *vertexState,
                         real specificHeatRatio, int nTimeStep)
{
  // Return if skipping this time step
  if (nTimeStep % meshParameter->nStepSkipCoarsen != 0) return 0;

  int nTriangle = connectivity->triangleVertices->GetSize();

  triangleWantRefine->SetSize(nTriangle);

  // Flag triangles if refinement / coarsening is needed
  FillWantRefine<realNeq, CL>(vertexState, specificHeatRatio);

  int nRemove =
    coarsen->RemoveVertices<realNeq, CL>(connectivity,
                                         predicates,
                                         vertexState,
                                         specificHeatRatio,
                                         triangleWantRefine,
                                         meshParameter,
                                         delaunay, 1);

  if (nRemove > 0) {
    CalcNormalEdge();
    connectivity->CalcVertexArea(GetPx(), GetPy());
    FindBoundaryVertices();
  }

  return nRemove;
}

//##############################################################################
// Instantiate
//##############################################################################

template int
Mesh::RemoveVertices<real, CL_ADVECT>(Array<real> *vertexState,
                                      real specificHeatRatio,
                                      int nTimeStep);
template int
Mesh::RemoveVertices<real, CL_BURGERS>(Array<real> *vertexState,
                                      real specificHeatRatio,
                                      int nTimeStep);
template int
Mesh::RemoveVertices<real3, CL_CART_ISO>(Array<real3> *vertexState,
                                      real specificHeatRatio,
                                      int nTimeStep);
template int
Mesh::RemoveVertices<real4, CL_CART_EULER>(Array<real4> *vertexState,
                                      real specificHeatRatio,
                                      int nTimeStep);

}  // namespace astrix
