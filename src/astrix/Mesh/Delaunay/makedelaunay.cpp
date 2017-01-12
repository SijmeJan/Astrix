// -*-c++-*-
/*! \file makedelaunay.cpp
\brief Main routine transforming Mesh into Delaunay Mesh */
#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./delaunay.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {
  
//#########################################################################
/*! Transform triangulated Mesh into Delaunay Mesh. This is achieved by flipping edges that do not have the Delaunay property. First, we make a list of edges that are not Delaunay, then we select those that can be flipped in parallel, we adjust the state vector in order to conserve mass, momentum and energy, and finally we flip the edges. A repair step ensures all edges have the correct neighbouring triangles. This is repeated until all edges are Delaunay.

\param *connectivity Pointer to basic Mesh data
\param *vertexState Pointer to state vector
\param *predicates Pointer to Predicates object, used to check Delaunay property without roundoff error
\param *meshParameter Pointer to Mesh parameters
\param maxCycle Maximum number of cycles. If <= 0, cycle until all edges are Delaunay*/
//#########################################################################

void Delaunay::MakeDelaunay(Connectivity * const connectivity,
			    Array<realNeq> * const vertexState,
			    const Predicates *predicates,
			    const MeshParameter *meshParameter,
			    const int maxCycle,
			    Array<int> * const edgeNeedsChecking,
			    const int nEdgeCheck,
			    const int flopFlag)
{  
  nvtxEvent *nvtxDelaunay = new nvtxEvent("Delaunay", 6);

  int nEdge = connectivity->edgeTriangles->GetSize();
  int nTriangle = connectivity->triangleVertices->GetSize();

  triangleSubstitute->SetSize(nTriangle);
  edgeNonDelaunay->SetSize(nEdge);

  triangleAffected->SetSize(2*nEdge);
  triangleAffectedEdge->SetSize(2*nEdge);

  int finished = 0;
  int nCycle = 0;
  while (!finished) {
    nvtxEvent *nvtxTemp = new nvtxEvent("CheckEdge", 1);

    if (flopFlag != 1) {
      // Check all edges for Delaunay property
      CheckEdges(connectivity, predicates, meshParameter,
		 edgeNeedsChecking, nEdgeCheck);
    } else {
      // Check all edges if can be flopped
      CheckEdgesFlop(connectivity, predicates, meshParameter,
		     edgeNeedsChecking, nEdgeCheck);
      finished = 1;
    }
    
    delete nvtxTemp;
    nvtxTemp = new nvtxEvent("Compact", 2);

    // Select edges that are not Delaunay (note: size of Array not changed!)
    int nNonDel = edgeNonDelaunay->RemoveValue(-1);

    delete nvtxTemp;

    if (nNonDel == 0) {
      // No more edges to flip: done
      finished = 1;
    } else {
      nvtxTemp = new nvtxEvent("Parallel", 3);

      // Find edges that can be flipped in parallel
      nNonDel = FindParallelFlipSet(connectivity, nNonDel);

      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Sub", 4);

      // Fill substitution triangles for repair step
      FillTriangleSubstitute(connectivity, nNonDel);

      // Adjust state for conservation
      if (vertexState != 0)
	AdjustState(connectivity, vertexState, predicates,
		    meshParameter, nNonDel);
      
      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Flip", 5);

      // Flip edges
      FlipEdge(connectivity, nNonDel);
      
      delete nvtxTemp;
      nvtxTemp = new nvtxEvent("Repair", 6);

      // Repair
      EdgeRepair(connectivity, edgeNeedsChecking, nEdgeCheck);
      //EdgeRepair(connectivity, 0, nEdgeCheck);

      delete nvtxTemp;
    }

    nCycle++;
    if (maxCycle > 0)
      if (nCycle >= maxCycle) finished = 1;
  }

  delete nvtxDelaunay;
}

}
