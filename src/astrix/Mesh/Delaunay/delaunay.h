/*! \file delaunay.h
\brief Header file for Delaunay class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
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
  template<class realNeq, ConservationLaw CL>
    void MakeDelaunay(Connectivity * const connectivity,
                      Array<realNeq> * const vertexState,
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
  //Array <real> *vertexArea;
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
  void EdgeRepair(Connectivity * const connectivity,
                  Array<int> * const edgeNeedsChecking,
                  const int nEdgeCheck);
  //! Adjust state in order to remain conservative
  template<class realNeq, ConservationLaw CL>
    void AdjustState(Connectivity * const connectivity,
                     Array<realNeq> * const vertexState,
                     const Predicates *predicates,
                     const MeshParameter *meshParameter,
                     const int nNonDel);
};

}

#endif
