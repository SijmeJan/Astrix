/*! \file refine.h
\brief Header file for Refine class

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
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

//! Class containing functions for refining Mesh
/*! Refine the Mesh (add new vertices) either until a quality bound is met or add vertices manually.*/
class Refine
{
 public:
  //! Constructor
  Refine(int _cudaFlag, int _debugLevel, int _verboseLevel);
  //! Destructor; releases memory.
  ~Refine();

  //! Add vertices to Mesh until quality constraints met
  template<class realNeq>
    int ImproveQuality(Connectivity * const connectivity,
                       const MeshParameter *meshParameter,
                       const Predicates *predicates,
                       Morton * const morton,
                       Delaunay * const delaunay,
                       Array<realNeq> * const vertexState,
                       Array<int> * const triangleWantRefine);

  //! Add list of vertices to Mesh
  int AddVertices(Connectivity * const connectivity,
                  const MeshParameter *meshParameter,
                  const Predicates *predicates,
                  Delaunay * const delaunay,
                  Array<real2> * const vertexCoordinatesToAdd,
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
  //! Determine which vertices can be inserted in parallel
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
  template<class realNeq>
    void InsertVertices(Connectivity * const connectivity,
                        const MeshParameter *meshParameter,
                        const Predicates *predicates,
                        Array<realNeq> * const vertexState,
                        Array<int> * const triangleWantRefine);
  //! Interpolate state at new vertices
  template<class realNeq>
    void InterpolateState(Connectivity * const connectivity,
                          const MeshParameter *meshParameter,
                          Array<realNeq> * const vertexState,
                          Array<int> * const triangleWantRefine);
  //! Split any additional segments
  template<class realNeq>
    void SplitSegment(Connectivity * const connectivity,
                      const MeshParameter *meshParameter,
                      const Predicates *predicates,
                      Array<realNeq> * const vertexState,
                      Array<int> * const triangleWantRefine,
                      const int nTriangleOld);
  //! Test whether any new vertices encroach a segment
  void TestEncroach(Connectivity * const connectivity,
                    const MeshParameter *meshParameter,
                    const int nRefine);

  //! Lock triangles that are in use
  void LockTriangles(Connectivity * const connectivity,
                     const Predicates *predicates,
                     const MeshParameter *meshParameter,
                     Array<int> *triangleInCavity);
  //! Find non-overlapping cavities
  void FindIndependentCavities(Connectivity * const connectivity,
                               const Predicates *predicates,
                               const MeshParameter *meshParameter,
                               Array<int> * const triangleInCavity,
                               Array<int> *uniqueFlag);
  //! Flag suspect edges for checking Delaunay-hood later
  void FlagEdgesForChecking(Connectivity * const connectivity,
                            const Predicates *predicates,
                            const MeshParameter *meshParameter);

};

}  // namespace astrix

#endif  // ASTRIX_REFINE_H
