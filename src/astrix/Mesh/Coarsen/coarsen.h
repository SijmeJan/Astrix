/*! \file coarsen.h
\brief Header file for Coarsen class

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
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

  //! Remove vertices from Mesh, maintaining quality
  template<class realNeq>
    int RemoveVertices(Connectivity *connectivity,
                       Predicates *predicates,
                       Array<realNeq> *vertexState,
                       Array<int> *triangleWantRefine,
                       const MeshParameter *meshParameter,
                       Delaunay *delaunay,
                       int maxCycle);

 private:
  //! Flag whether to use device or host
  int cudaFlag;
  //! Level of extra checks
  int debugLevel;
  //! Level of screen output
  int verboseLevel;

  //! Vector of random numbers to insert points randomly for efficiency
  Array<unsigned int> *randomUnique;
  //! Flag whether edge needs to be checked for Delaunay-hood
  Array<int> *edgeNeedsChecking;

  Array<int> *edgeCollapseList;
  Array<real2> *edgeCoordinates;

  //! Remove vertices from mesh
  template<class realNeq>
    void Remove(Connectivity *connectivity,
                Array<int> *triangleWantRefine,
                Array<realNeq> *vertexState);
  //! Adjust state conservatively after coarsening
  template<class realNeq>
    void AdjustState(Connectivity *connectivity,
                     Array<realNeq> *vertexState,
                     const MeshParameter *mp);
  //! Reject triangles that are too large for removal
  void RejectLargeTriangles(Connectivity *connectivity,
                            const MeshParameter *mp,
                            Array<int> *triangleWantRefine);
  //! Find set of points that can be removed in parallel
  void FindParallelDeletionSet(Connectivity *connectivity);
  //! Lock 'cavities' of deletion points
  void LockTriangles(Connectivity *connectivity,
                     Array<int> *triangleLock);
  //! Find independent 'cavities' of deletion points
  void FindIndependent(Connectivity *connectivity,
                       Array<int> *triangleLock);
  void FlagEdges(Connectivity *connectivity);


  void FillEdgeCollapseList(Connectivity *connectivity,
                            Array<int> *triangleWantRefine);
  void TestEdgeCollapse(Connectivity *connectivity,
                        Predicates *predicates,
                        const MeshParameter *mp);
};

}

#endif
