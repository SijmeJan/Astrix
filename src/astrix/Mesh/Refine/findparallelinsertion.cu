// -*-c++-*-
/*! \file findparallelinsertion.cu
\brief File containing function to find parallel insertion set.*/

#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./refine.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Predicates/predicates.h"
#include "../Param/meshparameter.h"
#include "./../triangleLow.h"
#include "../../Common/atomic.h"

namespace astrix {
  
//#########################################################################
/*! \brief Find independent set of insertion points

Upon return, \a elementAdd and \a vertexCoordinatesAdd are compacted to a set that can be inserted in one parallel step. This set is found by finding the cavities of the insertion points, keeping an insertion point \i only if none of the triangles in its cavity are needed by points > \a i. To optimize the set, we randomize the order by assigning a unique random number to each insertion point, and keep insertion point \a i with associated random number \a r only if none of the triangles in its cavity are needed by points with associated random number > \a r.    

\param *connectivity Pointer to basic Mesh data
\param *vertexOrder The order in which vertices were inserted. All entries relating to the independent set will be removed
\param *vertexOrderInsert Will be compacted with \a elementAdd and \a vertexCoordinatesAdd
\param *vOBCoordinates Coordinates of vertices to be inserted. All entries relating to the independent set will be removed
\param *predicates Pointer to Predicates object
\param *meshParameter Pointer to Mesh parameters*/
//#########################################################################
  
void Refine::FindParallelInsertionSet(Connectivity * const connectivity,
				      Array<int> * const vertexOrder,
				      Array<int> * const vertexOrderInsert,
				      Array<real2> * const vOBCoordinates,
				      const Predicates *predicates,
				      const MeshParameter *meshParameter)
{
  nvtxEvent *nvtxUnique = new nvtxEvent("unique", 3);

  // Number of triangles and number of insertion points
  unsigned int nTriangle = connectivity->triangleVertices->GetSize();
  unsigned int nRefine = elementAdd->GetSize();

  Array <int> *triangleInCavity = new Array<int>(1, cudaFlag, nTriangle);
  Array <int> *uniqueFlag = new Array<int>(1, cudaFlag, nRefine);
  Array <int> *uniqueFlagScan = new Array<int>(1, cudaFlag, nRefine);
  int *pUniqueFlag = uniqueFlag->GetPointer();

  // Shuffle points to add to maximise parallelisation
  unsigned int *pRandomPermutation = randomUnique->GetPointer();
  
  unsigned int nVertex = connectivity->vertexCoordinates->GetSize();
  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);
  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();
  int *pElementAdd = elementAdd->GetPointer();

  LockTriangles(connectivity, predicates, meshParameter, triangleInCavity);
  FindIndependentCavities(connectivity, predicates, meshParameter,
			  triangleInCavity, uniqueFlag);
 
  // Compact arrays to new nRefine
  nRefine = uniqueFlag->ExclusiveScan(uniqueFlagScan, nRefine);
  elementAdd->Compact(nRefine, uniqueFlag, uniqueFlagScan);
  vertexCoordinatesAdd->Compact(nRefine, uniqueFlag, uniqueFlagScan);
  pVcAdd = vertexCoordinatesAdd->GetPointer();
  pElementAdd = elementAdd->GetPointer();

  if (vertexOrder != 0) {
    // Compact insertion order
    vertexOrderInsert->Compact(nRefine, uniqueFlag, uniqueFlagScan);
    // Remove inserted vertices from list of boundary vertices to be inserted
    uniqueFlag->Invert();
    int nIgnore = uniqueFlag->ExclusiveScan(uniqueFlagScan,
					    uniqueFlag->GetSize());
    vOBCoordinates->Compact(nIgnore, uniqueFlag, uniqueFlagScan);
    vertexOrder->Compact(nIgnore, uniqueFlag, uniqueFlagScan);
  }

  FlagEdgesForChecking(connectivity, predicates, meshParameter);

  delete triangleInCavity;
  
  delete uniqueFlag;
  delete uniqueFlagScan;

  delete nvtxUnique;
}

}
