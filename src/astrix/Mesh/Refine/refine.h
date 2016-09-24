/*! \file refine.h
\brief Header file for Refine class*/
#ifndef ASTRIX_REFINE_H
#define ASTRIX_REFINE_H

namespace astrix {
  
// Forward declarations
template <class T> class Array;
class Predicates;
class Morton;
class Delaunay;
class Connectivity;
class MeshParameter;
 
class Refine
{
 public:
  //! Constructor
  Refine(int _cudaFlag, int _debugLevel, int _verboseLevel);
  //! Destructor; releases memory.
  ~Refine();  

  //! Add vertices to Mesh until quality constraints met
  int ImproveQuality(Connectivity * const connectivity,		
		     const MeshParameter *meshParameter,
		     const Predicates *predicates,
		     Morton * const morton,
		     Delaunay * const delaunay,
		     Array<realNeq> * const vertexState,
		     const real specificHeatRatio,
		     Array<int> * const triangleWantRefine);

  //! Add list of vertices to Mesh
  int AddVertices(Connectivity * const connectivity,
		  const MeshParameter *meshParameter,
		  const Predicates *predicates,
		  Delaunay * const delaunay,
		  Array<real2> * const vertexBoundaryCoordinates,
		  Array<int> * const vertexOrder);

 private:
  //! Flag whether to use device or host
  int cudaFlag;
  //! Level of extra checks
  int debugLevel;
  //! Level of output to stdout
  int verboseLevel;

  //! List of low-quality triangles
  Array<int> *badTriangles;
  //! Coordinates of vertices to be added
  Array <real2> *vertexCoordinatesAdd;
  //! List of triangles/edges to put new vertices in/on
  Array <int> *elementAdd;
  //! Triangles affected by inserting vertices
  Array <int> *triangleAffected;
  //! Vertices to be inserted affecting triangles (0...nRefine) 
  Array <int> *triangleAffectedIndex;
  //! Flag if edge needs checking for Delaunay-hood
  Array<int> *edgeNeedsChecking;
  //! Unique random numbers
  Array<unsigned int> *randomUnique;
  
  //! Find low-quality triangles
  int TestTrianglesQuality(Connectivity * const connectivity,
			   const MeshParameter *meshParameter,
                           const Array<int> *triangleWantRefine);
  //! Find circumcentres of bad triangles
  void FindCircum(Connectivity * const connectivity,
		  const MeshParameter *meshParameter,
		  const int nRefine);
  //! Find triangles or edges to put new vertices in/on
  void FindTriangles(Connectivity * const connectivity,
		     const MeshParameter *meshParameter,
		     const Predicates *predicates);
  //! Adjust indices of periodic vertices for new \a nVertex
  void AddToPeriodic(Connectivity * const connectivity,
		     const int nAdd);
  void FindParallelInsertionSet(Connectivity * const connectivity,
				Array<int> * const vertexOrder,
				Array<int> * const vertexOrderInsert,
				Array<real2> * const vOBCoordinates,
				const Predicates *predicates,
				const MeshParameter *meshParameter);
  
  //! Flag points to be inserted on segments
  int FlagSegment(Connectivity * const connectivity,
		  Array<unsigned int> * const onSegmentFlagScan);
  //! Insert new vertices into Mesh
  void InsertVertices(Connectivity * const connectivity,
		      const MeshParameter *meshParameter,
		      const Predicates *predicates,
		      Array<realNeq> * const vertexState,
		      Array<int> * const triangleWantRefine);
  //! Interpolate state at new vertices
  void InterpolateState(Connectivity * const connectivity,
			const MeshParameter *meshParameter,
			Array<realNeq> * const vertexState,
			Array<int> * const triangleWantRefine,
			const real specificHeatRatio);
  //! Split any additional segments
  void SplitSegment(Connectivity * const connectivity,
		    const MeshParameter *meshParameter,
		    const Predicates *predicates,
		    Array<realNeq> * const vertexState,
		    Array<int> * const triangleWantRefine,
		    const real specificHeatRatio,
		    const int nTriangleOld);
  //! Test whether any new vertices encroach a segment
  void TestEncroach(Connectivity * const connectivity,
		    const MeshParameter *meshParameter,
		    const int nRefine);

};

}

#endif
