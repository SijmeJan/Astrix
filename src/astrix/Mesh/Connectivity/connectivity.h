/*! \file connectivity.h
\brief Header file for Connectivity class*/
#ifndef ASTRIX_CONNECTIVITY_H
#define ASTRIX_CONNECTIVITY_H

namespace astrix {
  
// Forward declaration
template <class T> class Array;
 
//! Class containing Mesh data structure
/*! Class containing coordinates and connectivity of Mesh; data needed by all Mesh-related classes. The class is essentially data-only, plus a few functions to move data between host and device. All data members are public, which can be unsafe. It is assumed that at the end of any function modifying the Mesh, the Connectivity represents a valid triangulation (not necessarily Delaunay), and that the sizes of the Arrays are properly set, i.e. the size of \a vertexCoordinates is the number of vertices, the size of both \a triangleVertices and \a triangleEdges equals the number of triangles, and the size of \a edgeTriangles equals the number of edges.*/
class Connectivity
{
 public:
  //! Constructor
  Connectivity(int _cudaFlag);
  //! Destructor; releases memory.
  ~Connectivity();  

  //! Coordinates of vertices
  Array <real2> *vertexCoordinates;
  //! Vertices belonging to triangles
  Array <int3> *triangleVertices;
  //! Edges belonging to triangles
  Array <int3> *triangleEdges;
  //! Triangles associated with edges
  Array <int2> *edgeTriangles;

  //! Transform from device to host or vice versa
  void Transform();
  //! Copy data to host
  void CopyToHost();
  //! Copy data to device
  void CopyToDevice();
 private:
  //! Flag whether date resides on host (0) or device (1)
  int cudaFlag;
};

}

#endif

