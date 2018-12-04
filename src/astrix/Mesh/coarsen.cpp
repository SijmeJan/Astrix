// -*-c++-*-
/*! \file coarsen.cpp
\brief Top-level function for coarsening Mesh

*/ /* \section LICENSE
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
\param nTimeStep Number of time steps taken so far. Used in combination with \a nStepSkipCoarsen to possibly avoid coarsening every timestep
\param *triangleWantRefine Pointer to triangles that need to be refined/coarsened*/
//#########################################################################

template<class realNeq>
int Mesh::RemoveVertices(Array<realNeq> *vertexState,
                         int nTimeStep,
                         Array<int> *triangleWantRefine)
{
  // Return if skipping this time step
  if (nTimeStep % meshParameter->nStepSkipCoarsen != 0) return 0;

  int nRemove = 0;
  try {
    nRemove = coarsen->RemoveVertices<realNeq>(connectivity,
                                               predicates,
                                               vertexState,
                                               triangleWantRefine,
                                               meshParameter,
                                               delaunay, 1);
  }
  catch (...) {
    std::cout << "Error coarsening Mesh" << std::endl;
    throw;
  }

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
Mesh::RemoveVertices<real>(Array<real> *vertexState,
                           int nTimeStep,
                           Array<int> *triangleWantRefine);
template int
Mesh::RemoveVertices<real3>(Array<real3> *vertexState,
                            int nTimeStep,
                            Array<int> *triangleWantRefine);
template int
Mesh::RemoveVertices<real4>(Array<real4> *vertexState,
                            int nTimeStep,
                            Array<int> *triangleWantRefine);

}  // namespace astrix
