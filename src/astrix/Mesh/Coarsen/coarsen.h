/*! \file coarsen.h
\brief Header file for Coarsen class*/
#ifndef ASTRIX_COARSEN_H
#define ASTRIX_COARSEN_H

namespace astrix {
  
// Forward declarations
template <class T> class Array;
class Predicates;
class Morton;
class Delaunay;
class Connectivity;
class MeshParameter;
 
class Coarsen
{
 public:
  //! Constructor
  Coarsen(int _cudaFlag, int _debugLevel, int _verboseLevel);
  //! Destructor; releases memory.
  ~Coarsen();  

  int RemoveVertices(Connectivity *connectivity,
		     Predicates *predicates,
		     Array<real4> *vertexState,
		     real specificHeatRatio,
		     Array<int> *triangleWantRefine,
		     const MeshParameter *meshParameter,
		     Delaunay *delaunay,
		     int maxCycle,
		     Array<unsigned int> *randomVector);

 private:
  //! Flag whether to use device or host
  int cudaFlag;
  //! Level of extra checks
  int debugLevel;
  int verboseLevel;

  //! Indices of vertices to be removed
  Array <int> *vertexRemove;
  //! Every vertex has at least one triangle associated with it
  Array <int> *vertexTriangle;
  //! Vector of random numbers to insert points randomly for efficiency
  //Array <unsigned int> *randomVector;
  //! Area associated with vertex (Voronoi cell)
  Array<real> *vertexArea;
  
  //! Check if removing vertices leads to encroached segment
  void CheckEncroach(Connectivity *connectivity,
		     Predicates *predicates,
		     Array<int> *vertexRemoveFlag,
		     const MeshParameter *mp);
  //! Remove vertices from mesh
  void Remove(Connectivity *connectivity,
	      Array<int> *triangleWantRefine,
	      Array<int> *vertexTriangleList,
	      int maxTriPerVert,
	      Array<int> *triangleTarget,
	      Array<real4> *vertexState);
  //! Adjust state conservatively after coarsening
  void AdjustState(Connectivity *connectivity,
		   int maxTriPerVert,
		   Array<int> *vertexTriangleList,
		   Array<int> *triangleTarget,
		   Array<real4> *vertexState,
		   real G, const MeshParameter *mp,
		   Array<int> *vertexNeighbour);
  //! Find single triangle for every vertex 
  void FillVertexTriangle(Connectivity *connectivity);
  //! Maximum number of triangles per vertex
  int MaxTriPerVert(Connectivity *connectivity);
  //! Flag vertices for removal
  void FlagVertexRemove(Connectivity *connectivity,
			Array<int> *vertexRemoveFlag,
			Array<int> *triangleWantRefine);
  //! Create list of triangles sharing every vertex
  void CreateVertexTriangleList(Connectivity *connectivity,
				Array<int> *vertexTriangleList,
				int maxTriPerVert);
  //! Find allowed 'target' triangles for vertex removal
  void FindAllowedTargetTriangles(Connectivity *connectivity,
				  Predicates *predicates,
				  Array<int> *vertexTriangleAllowed,
				  Array<int> *vertexTriangleList,
				  int maxTriPerVert,
				  const MeshParameter *mp);
  //! Find 'target' triangles for vertex removal
  void FindTargetTriangles(Connectivity *connectivity,
			   Array<int> *triangleWantRefine,
			   Array<int> *triangleTarget,
			   Array<int> *vertexTriangleAllowed,
			   Array<int> *vertexTriangleList,
			   int maxTriPerVert);
  //! Find vertices neighbouring vertex
  void FindVertexNeighbours(Connectivity *connectivity,
			    Array<int> *vertexNeighbour,
			    Array<int> *triangleTarget,
			    Array<int> *vertexTriangleList,
			    int maxTriPerVert);
  //! Reject triangles that are too large for removal
  void RejectLargeTriangles(Connectivity *connectivity,
			    const MeshParameter *mp,
			    Array<int> *triangleWantRefine);
  //! Find set of points that can be removed in parallel
  void FindParallelDeletionSet(Connectivity *connectivity,
			       int maxTriPerVert,
			       Array<unsigned int> *randomVector);
  void CalcVertexArea(Connectivity *connectivity,
		      const MeshParameter *meshParameter);

};

}

#endif
