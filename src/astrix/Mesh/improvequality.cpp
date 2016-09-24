// -*-c++-*-
/*! \file refine.cpp
\brief Function to refine Mesh*/
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
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
\param specificHeatRatio Ratio of specific heats
\param nTimeStep Number of time steps taken so far. Used in combination with \a nStepSkipRefine to possibly avoid refining every timestep*/
//#########################################################################
  
int Mesh::ImproveQuality(Array<realNeq> *vertexState,
			 real specificHeatRatio, int nTimeStep)
{
  if (nTimeStep % meshParameter->nStepSkipRefine != 0) return 0;

  int nAdded = 0;
  
  // Flag triangles if refinement is needed
  if (vertexState != 0) {
    triangleWantRefine->SetSize(nTriangle);
    FillWantRefine(vertexState, specificHeatRatio);
    //dmax = dMaxBase/((real)(maxRefineFactor*maxRefineFactor));
    
    nAdded = refine->ImproveQuality(connectivity,
				    meshParameter,
				    predicates,
				    morton, delaunay,
				    vertexState,
				    specificHeatRatio,
				    triangleWantRefine);
  } else {
    try {
      nAdded = refine->ImproveQuality(connectivity,
				      meshParameter,
				      predicates,
				      morton, delaunay,
				      vertexState,
				      specificHeatRatio, 0);
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

  std::cout << "L/lmin = " << Px/minEdgeLength << std::endl;
  
  
  if (nAdded > 0) {
    if (debugLevel > 0)
      morton->Order(connectivity, triangleWantRefine, vertexState);
  
    // Calculate triangle normals and areas
    CalcNormalEdge();
    CalcVertexArea();
    FindBoundaryVertices();
  }
  
  return nAdded;
}

}
