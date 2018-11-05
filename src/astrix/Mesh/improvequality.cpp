// -*-c++-*-
/*! \file improvequality.cpp
\brief Function to refine Mesh

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include <cmath>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./Morton/morton.h"
#include "./mesh.h"
#include "./Refine/refine.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"
#include "./triangleLow.h"

namespace astrix {

//#########################################################################
/*! Refine Delaunay mesh, depending on the state vector (i.e. density, momentum, etc). If \a *vertexState != 0, we first calculate an estimate of the discretization error and flag triangles to be refined; otherwise, all triangles can be refined, necessary for example when building the mesh for the first time. It returns the number of vertices added.

\param *vertexState Pointer to state vector at vertices
\param nTimeStep Number of time steps taken so far. Used in combination with \a nStepSkipRefine to possibly avoid refining every timestep
\param *triangleWantRefine Pointer to Array of flags whether triangles need to be refined*/
//#########################################################################

template<class realNeq>
int Mesh::ImproveQuality(Array<realNeq> *vertexState,
                         int nTimeStep,
                         Array<int> *triangleWantRefine)
{
  if (nTimeStep % meshParameter->nStepSkipRefine != 0) return 0;

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int nAdded = 0;

  // Flag triangles if refinement is needed
  if (vertexState != 0) {
    // We are in the middle of a simulation: refine flagged triangles
    nAdded = refine->ImproveQuality<realNeq>(connectivity,
                                             meshParameter,
                                             predicates,
                                             morton,
                                             delaunay,
                                             vertexState,
                                             triangleWantRefine);
  } else {
    try {
      // Start of simulation
      nAdded = refine->ImproveQuality<realNeq>(connectivity,
                                               meshParameter,
                                               predicates,
                                               morton,
                                               delaunay,
                                               vertexState,
                                               0);
    }
    catch (...) {
      std::cout << "Error improving Mesh, saving Mesh" << std::endl;
      Save(1000);
      throw;
    }
  }

  nVertex = connectivity->vertexCoordinates->GetSize();
  nTriangle = connectivity->triangleVertices->GetSize();
  nEdge = connectivity->edgeTriangles->GetSize();

  // Minimum edge length

  if (cudaFlag == 1) {
    connectivity->vertexCoordinates->CopyToHost();
    connectivity->triangleVertices->CopyToHost();
  }

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  real minEdgeLength = Px;

  for (int i = 0; i < nTriangle; i++) {
    int a = pTv[i].x;
    int b = pTv[i].y;
    int c = pTv[i].z;

    real ax, bx, cx, ay, by, cy;
    GetTriangleCoordinates(pVc, a, b, c,
                           nVertex, Px, Py,
                           ax, bx, cx, ay, by, cy);

    // Three edges
    real l1 = sqrt((ax - bx)*(ax - bx) + (ay - by)*(ay - by));
    real l2 = sqrt((ax - cx)*(ax - cx) + (ay - cy)*(ay - cy));
    real l3 = sqrt((cx - bx)*(cx - bx) + (cy - by)*(cy - by));

    real lmin = std::min(l1, std::min(l2, l3));

    minEdgeLength = std::min(minEdgeLength, lmin);
  }

  //std::cout << "L/lmin = " << Px/minEdgeLength << std::endl;

  if (nAdded > 0) {
    // Calculate triangle normals and areas
    CalcNormalEdge();
    connectivity->CalcVertexArea(GetPx(), GetPy());
    FindBoundaryVertices();
  }

  return nAdded;
}

//##############################################################################
// Instantiate
//##############################################################################

template int
Mesh::ImproveQuality<real>(Array<real> *vertexState,
                           int nTimeStep,
                           Array<int> *triangleWantRefine);
template int
Mesh::ImproveQuality<real3>(Array<real3> *vertexState,
                            int nTimeStep,
                            Array<int> *triangleWantRefine);
template int
Mesh::ImproveQuality<real4>(Array<real4> *vertexState,
                            int nTimeStep,
                            Array<int> *triangleWantRefine);

}  // namespace astrix
