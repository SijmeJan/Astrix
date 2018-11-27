/*! \file mesh.h
\brief Header file for Mesh class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_MESH_H
#define ASTRIX_MESH_H

#include <cuda_runtime_api.h>
#include <string>

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
/*! A Mesh object can be thought of representing the Delaunay mesh on which the
computations are performed.*/
class Mesh
{
 public:
  //! Constructor
  Mesh(int meshVerboseLevel, int meshDebugLevel, int meshCudaFlag,
       const char *fileName, Device *device, int restartNumber);
  //! Destructor; releases memory.
  ~Mesh();

  //! Refine mesh
  template<class realNeq>
  int ImproveQuality(Array<realNeq> *vertexState,
                     int nTimeStep,
                     Array<int> *triangleWantRefine);
  //! Coarsen mesh
  template<class realNeq>
  int RemoveVertices(Array<realNeq> *vertexState,
                     int nTimeStep,
                     Array<int> *triangleWantRefine);

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
  //! Return minimum x
  real GetMinX();
  //! Return maximum x
  real GetMaxX();
  //! Return minimum y
  real GetMinY();
  //! Return maximum y
  real GetMaxY();
  //! Return total vertex area
  real GetTotalArea();

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
  //! Return triangle average X data
  const real* TriangleAverageXData();

  //! Return edge triangles data
  const int2* EdgeTrianglesData();

  // Allow switch between host and device memory

  //! Transform all Arrays
  void Transform();

 private:
  //! Basic Mesh structure
  Connectivity *connectivity;
  //! Class holding parameters for the mesh
  MeshParameter *meshParameter;
  //! Delaunay object to transform mesh into Delaunay mesh
  Delaunay *delaunay;
  //! Morton object to improve data locality
  Morton *morton;
  //! Object for refining mesh
  Refine *refine;
  //! Object for coarsening mesh
  Coarsen *coarsen;
  //! Use exact geometric predicates
  Predicates *predicates;

  //! Flag whether vertex is part of boundary
  Array <int> *vertexBoundaryFlag;

  //! Normal vector to triangle edges (normalized)
  Array <real2> *triangleEdgeNormals;
  //! Triangle edge lengths
  Array <real3> *triangleEdgeLength;

  //! Average x coordinate triangle (needed in cylindrical geometry)
  Array <real> *triangleAverageX;

  // Runtime flags

  //! Flag whether running on CUDA device
  int cudaFlag;
  //! How much to output to screen
  int verboseLevel;
  //! Level of extra checks on mesh
  int debugLevel;

  //! Set up the mesh
  void Init(const char *fileName, int restartNumber);
  //! Output mesh statistics to stdout
  void OutputStat();

  // Functions calculating important Mesh properties

  //! Calculate triangle normals and edge lengths
  void CalcNormalEdge();
  //! Flag vertices where boundary conditions need to be applied
  void FindBoundaryVertices();

  //! Construct mesh boundaries
  void ConstructBoundaries(Array<real2> *vertexBoundaryCoordinates);
  //! Make initial mesh periodic
  void MakePeriodic();
  //! Remove redundant triangles from initial mesh
  void RemoveRedundant(Array<int> *vertexOrder, int nVertexOuterBoundary);

  //! Create structured mesh
  void CreateStructuredMesh();

  // Debugging functions

  //! Return maximum edge length for triangle \a i
  real MaxEdgeLengthTriangle(int i);
  //! Return maximum edge length for whole grid
  real MaximumEdgeLength();
  //! Check if any vertex encroaches upon segment (slow, used for debugging)
  void CheckEncroachSlow();
  //! Check if any edge is larger than \a maxEdgeLength
  void CheckEdgeLength(real maxEdgeLength);

  //! Find convex hull of set of points
  Array<real2>* ConvexHull(Array<real2> *pointCoordinates);

};

}  // namespace astrix

#endif  // ASTRIX_MESH_H
