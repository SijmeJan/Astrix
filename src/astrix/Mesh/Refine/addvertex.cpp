// -*-c++-*-
/*! \file addvertices.cpp
\brief Add specific list of vertices to Mesh*/
#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../Delaunay/delaunay.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {
  
//#########################################################################
/*! Add a list of vertices to the Mesh. Used in cases where vertex locations are known in advance, i.e. when inserting boundaries. Can only be used at t = 0, so no interpolation of the state is necessary. 

\param *connectivity Pointer to basic Mesh data: vertices, triangles, edges
\param *meshParameter Pointer to Mesh parameters, read from input file
\param *predicates Exact geometric predicates
\param *delaunay Pointer to Delaunay object, used to maintain Delaunay triangulation
\param *vertexBoundaryCoordinates Pointer to coordinates of vertices to be added
\param *vertexOrder Output: the order in which vertices were inserted.*/
//#########################################################################

int Refine::AddVertices(Connectivity * const connectivity,
			const MeshParameter *meshParameter,
			const Predicates *predicates,
			Delaunay * const delaunay,
			Array<real2> * const vertexBoundaryCoordinates,
			Array<int> * const vertexOrder)
{
  nvtxEvent *nvtxRefine = new nvtxEvent("Refine", 1);

  // Going to add all boundary vertices
  int nRefine = vertexBoundaryCoordinates->GetSize();
  int nAdded = 0;
    
  Array<int> *vertexOrderInsert = new Array<int>(1, cudaFlag);
  Array<int> *vertexBoundaryOrder =
    new Array<int>(1, cudaFlag, (unsigned int) nRefine);
  vertexBoundaryOrder->SetToSeries();
  vertexBoundaryOrder->AddValue(4, 0, nRefine);

  // While there are still points to add
  while (nRefine > 0) {
    vertexOrderInsert->SetSize(nRefine);
    vertexOrderInsert->SetEqual(vertexBoundaryOrder);
    
    elementAdd->SetSize(nRefine);
    vertexCoordinatesAdd->SetSize(nRefine);
    vertexCoordinatesAdd->SetEqual(vertexBoundaryCoordinates);

    badTriangles->SetSize(nRefine);
    badTriangles->SetToValue(-1);
    
    // Find triangles for all new vertices
    try {
      FindTriangles(connectivity, meshParameter, predicates);
    }
    catch (...) {
      std::cout << "Error finding triangles" << std::endl;
      throw;
    }

    // Find unique triangle set
    FindParallelInsertionSet(connectivity,
			     vertexBoundaryOrder,
			     vertexOrderInsert,
			     vertexBoundaryCoordinates,
			     predicates, meshParameter); 

    // Add inserted vertices to vertexOrder
    vertexOrder->Concat(vertexOrderInsert);

    nRefine = elementAdd->GetSize();
    nAdded += nRefine;
    
    AddToPeriodic(connectivity, nRefine);

    // Insert vertices into Mesh
    InsertVertices(connectivity, meshParameter, predicates, 0, 0);

    nRefine = vertexBoundaryCoordinates->GetSize();

    // Maintain Delaunay triangulation
    delaunay->MakeDelaunay(connectivity, 0, predicates,
			   meshParameter, 0, 0, 0, 0);
    
    if (verboseLevel > 1) std::cout << std::endl;
  }

  delete vertexOrderInsert;
  delete vertexBoundaryOrder;
 
  delete nvtxRefine;

  return nAdded;
}

}
