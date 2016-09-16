/*! \file delaunay.h
\brief Header file for Delaunay class*/
#ifndef ASTRIX_DELAUNAY_H
#define ASTRIX_DELAUNAY_H

namespace astrix {
  
// Forward declarations
template <class T> class Array;
class Predicates;
class Connectivity;
class MeshParameter;
 
//! Class for turning Mesh into Delaunay mesh
/*! The public function of this class turns the Mesh into a Delaunay mesh by edge flipping.*/
class Delaunay
{
 public:
  //! Constructor
  Delaunay(int _cudaFlag, int _debugLevel);
  //! Destructor; releases memory.
  ~Delaunay();  

  //! Turn Mesh into Delaunay nesh
  void MakeDelaunay(Connectivity * const connectivity,
		    Array<real4> * const vertexState,
		    const Predicates *predicates,
		    const MeshParameter *meshParameter,
		    const int maxCycle,
		    Array<int> * const edgeNeedsChecking,
		    const int nEdgeCheck,
		    const int flopFlag);

 private:
  //! Flag whether to use device or host
  int cudaFlag;
  //! Level of extra checks
  int debugLevel;
  
  //! List of edges that are not Delaunay
  Array <int> *edgeNonDelaunay;
  //! List of triangle subs, possibly necessary after flip
  Array <int> *triangleSubstitute;
  //! Area associated with vertex (Voronoi cell)
  Array <real> *vertexArea;
  //! List of triangles affected by flips
  Array <int> *triangleAffected;
  //! Indices in edgeNonDelaunay for affected triangles
  Array <int> *triangleAffectedEdge;

  //! Check if any edges are not Delaunay
  void CheckEdges(Connectivity * const connectivity,
		  const Predicates *predicates,
		  const MeshParameter *meshParameter,
		  Array<int> * const edgeNeedsChecking,
		  const int nEdgeCheck);
  //! Check if edges can be flopped
  void CheckEdgesFlop(Connectivity * const connectivity,
		      const Predicates *predicates,
		      const MeshParameter *meshParameter,
		      Array<int> * const edgeNeedsChecking,
		      const int nEdgeCheck);
  //! Find set of edges that can be flipped in parallel
  int FindParallelFlipSet(Connectivity * const connectivity,
			  const int nNonDel);
   //! Flip edges in parallel
  void FlipEdge(Connectivity * const connectivity,
		const int nNonDel);
  //! Fill triangle substitute Array for repairing edges
  void FillTriangleSubstitute(Connectivity * const connectivity,
			      const int nNonDel);
  //! Repair damaged edges after flipping
  void EdgeRepair(Connectivity * const connectivity);
  //! Adjust state in order to remain conservative
  void AdjustState(Connectivity * const connectivity,
		   Array<real4> * const vertexState,
		   const Predicates *predicates,
		   const MeshParameter *meshParameter,
		   const int nNonDel);
  //! Calculate areas associates with vertices (Voronoi cells)
  void CalcVertexArea(Connectivity * const connectivity,
		      const MeshParameter *meshParameter);
};

}

#endif
