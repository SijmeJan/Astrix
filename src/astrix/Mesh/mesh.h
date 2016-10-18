/*! \file mesh.h
\brief Header file for Mesh class
*/
#ifndef ASTRIX_MESH_H
#define ASTRIX_MESH_H

#include <string>
#include <cuda_runtime_api.h>

namespace astrix {
  
// Forward declarations
class Predicates;
template <class T> class Array;
class Device;
class Morton;
class Delaunay;
class Refine;
class Coarsen;
class Connectivity;
class MeshParameter;
 
//! Class containing functions acting on the whole mesh 
/*! A Mesh object can be thought of representing the Delaunay mesh on which the computations are performed.*/
class Mesh
{
 public:
  //! Constructor
  Mesh(int meshVerboseLevel, int meshDebugLevel, int meshCudaFlag,
       const char *fileName, Device *device, int restartNumber,
       int extraFlag);
  //! Destructor; releases memory.
  ~Mesh();  
 
  //! Refine mesh
  int ImproveQuality(Array<realNeq> *vertexState,
		     real specificHeatRatio,
		     int nTimeStep);
  //! Coarsen mesh
  int RemoveVertices(Array<realNeq> *vertexState,
		     real specificHeatRatio,
		     int nTimeStep);
  
  //! Save mesh to disk
  void Save(int nSave);
  //! Read previously created mesh from disk
  void ReadFromDisk(int nSave);

  //! Return number of vertices
  int GetNVertex();
  //! Return number of triangles
  int GetNTriangle();
  //! Return number of edges
  int GetNEdge();
  //! Return if mesh is adaptive
  int IsAdaptive();
  //! Return size of domain in x
  real GetPx();
  //! Return size of domain in y
  real GetPy();
  
  // Give access to Mesh data through constant pointers to the data
  
  //! Return pointer to vertex coordinates data
  const real2* VertexCoordinatesData();
  //! Return pointer to vertex boundary flag data
  const int* VertexBoundaryFlagData();
  //! Return pointer to vertex area data
  const real* VertexAreaData();

  //! Return pointer to triangle vertices data
  const int3* TriangleVerticesData();
  //! Return pointer to triangle edges data
  const int3* TriangleEdgesData();
  //! Return pointer to triangle edge normals data
  const real2* TriangleEdgeNormalsData(int dim);
  //! Return triangle edge length data
  const real3* TriangleEdgeLengthData();

  //! Return edge triangles data
  const int2* EdgeTrianglesData();

  // Allow switch between host and device memory
  
  //! Transform all Arrays
  void Transform();
  
 private:
  // General Mesh Arrays

  //! Basic Mesh structure
  Connectivity *connectivity;

  MeshParameter *meshParameter;
  
  //! Flag whether vertex is part of boundary
  Array <int> *vertexBoundaryFlag;
  //! Vertex area (area of Voronoi cell)
  Array <real> *vertexArea;

  //! Flag whether triangle needs to be refined
  Array <int> *triangleWantRefine;
  //! Normal vector to triangle edges (normalized)
  Array <real2> *triangleEdgeNormals;
  //! Triangle edge lengths
  Array <real3> *triangleEdgeLength;
  //! Estimate of discretization error
  Array <real> *triangleErrorEstimate;

  Array<unsigned int> *randomVector;
  
  // Runtime flags
  
  //! Flag whether running on CUDA device
  int cudaFlag;
  //! How much to output to screen
  int verboseLevel;
  //! Level of extra checks on mesh
  int debugLevel;
  
  // Derived Mesh properties
  
  //! Number of vertices
  int nVertex;
  //! Number of triangles
  int nTriangle;
  //! Number of edges
  int nEdge;

  //! Use exact geometric predicates
  Predicates *predicates;
  //! GPU properties
  cudaDeviceProp deviceProp;

  //! Set up the mesh
  void Init(const char *fileName, int restartNumber, int extraFlag);
  //! Output mesh statistics to stdout
  void OutputStat();

  // Functions calculating important Mesh properties
  
  //! Calculate triangle normals and edge lengths
  void CalcNormalEdge();
  //! Calculate areas associated with vertices (Voronoi cells)
  void CalcVertexArea();
  //! Flag vertices where boundary conditions need to be applied
  void FindBoundaryVertices();

  //! Construct mesh boundaries
  void ConstructBoundaries();
  //! Create structured mesh
  void CreateStructuredMesh();
 
  // Functions for adaptive Mesh
  
  //! Calculate estimate of discretization error
  void CalcErrorEstimate(Array<realNeq> *vertexState, real G);
  //! Check which triangles want refining based of state
  void FillWantRefine(Array<realNeq> *vertexState, real specificHeatRatio);

  // Debugging functions
  
  //! Return maximum edge length for triangle \a i
  real MaxEdgeLengthTriangle(int i);
  //! Return maximum edge length for whole grid
  real MaximumEdgeLength();
  //! Check if any vertex encroaches upon segment (slow, used for debugging)
  void CheckEncroachSlow();
  //! Check if all triangles are legal (used for debugging)
  void CheckLegalTriangle();
  //! Check if any edge is larger than \a maxEdgeLength
  void CheckEdgeLength(real maxEdgeLength);

  //! Delaunay object to transform mesh into Delaunay mesh
  Delaunay *delaunay;

  //! Morton object to improve data locality
  Morton *morton;

  Refine *refine;
  Coarsen *coarsen;
};

}
#endif
