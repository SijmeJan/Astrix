// -*-c++-*-
/*! \file splitsegment.cpp
\brief File containing function to split additional segments*/
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//##############################################################################
/*! If a vertex encroaches a segment, we have moved the vertex onto the encroached segment. Occasionally, this vertex will encroach another segment. This function finds these segments and splits them one by one (not in parallel).

\param *vertexState State vector at vertices
\param specificHeatRatio Ratio of specific heats*/
//##############################################################################
  
void Refine::SplitSegment(Connectivity * const connectivity,
			  const MeshParameter *meshParameter,
			  const Predicates *predicates,
			  Array<real4> * const vertexState,
			  Array<int> * const triangleWantRefine,
			  const real specificHeatRatio,
			  const int nTriangleOld)
{  
  nvtxEvent *nvtxSplitSegment = new nvtxEvent("SplitSegment", 3);

  int nTriangle = connectivity->triangleVertices->GetSize();

  // We only need to consider vertices that were inserted on segments
  
  // Compact to points with elementAdd >= nTriangleOld
  int nRefineGlobal = elementAdd->SelectLargerThan(nTriangleOld - 1,
						   vertexCoordinatesAdd);
  if (nRefineGlobal == 0) {
    delete nvtxSplitSegment;
    return;
  }

  elementAdd->AddValue(nTriangle - nTriangleOld, 0, nRefineGlobal);
  
  // Edges onto which vertices were inserted
  Array<int> *elementAddOld = new Array<int>(1, cudaFlag, 
					     (unsigned int) nRefineGlobal);
  elementAddOld->SetEqual(elementAdd);

  // Test for encroached segments; if encroached move vertex onto segment 
  TestEncroach(connectivity, meshParameter, nRefineGlobal);

  // Select where edge has changed; those vertices encroach another segment
  nRefineGlobal =
    elementAdd->SelectWhereDifferent(elementAddOld, vertexCoordinatesAdd);
  if (nRefineGlobal == 0) {
    delete elementAddOld;
    delete nvtxSplitSegment;
    
    return;
  }
  
  elementAddOld->SetEqual(elementAdd);

  Array<real2> *vertexCoordinatesAddOld =
    new Array<real2>(1, cudaFlag, (unsigned int) nRefineGlobal);
  vertexCoordinatesAddOld->SetEqual(vertexCoordinatesAdd);
  
  // Insert vertices one by one
  for (int i = 0; i < nRefineGlobal; i++) {
    real x, y;
    real2 X;
    vertexCoordinatesAddOld->GetSingleValue(&X, i);
    x = X.x;
    y = X.y;
    
    int e;
    elementAddOld->GetSingleValue(&e, i);
    
    elementAdd->SetSize(1);
    vertexCoordinatesAdd->SetSize(1);
  
    elementAdd->SetToValue(e);

    X.x = x;
    X.y = y;
    vertexCoordinatesAdd->SetSingleValue(X, 0);
    
    if (vertexState != 0) 
      InterpolateState(connectivity,
		       meshParameter,
		       vertexState,
		       triangleWantRefine,
		       specificHeatRatio);
      
    AddToPeriodic(connectivity, 1);
      
    InsertVertices(connectivity,
		   meshParameter,
		   predicates,
		   vertexState,
		   triangleWantRefine);
  }

  delete elementAddOld;
  delete vertexCoordinatesAddOld;
  
  delete nvtxSplitSegment;
}

}
