/*! \file morton.h
\brief Header file for Morton class

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_MORTON_H
#define ASTRIX_MORTON_H

namespace astrix {

// Forward declarations
template <class T> class Array;
class Connectivity;

//! Class containing functions for Morton ordering
/*! A Morton object can be used to reorder vertices, triangles and edges to
improve data locality.*/
class Morton
{
 public:
  //! Constructor
  explicit Morton(int _cudaFlag);
  //! Destructor; releases memory.
  ~Morton();

  //! Reorder Arrays for maximum locality
  template<class realNeq>
    void Order(Connectivity * const connectivity,
               Array<int> * const triangleWantRefine,
               Array<realNeq> * const vertexState);

 private:
  //! Flag whether to use device or host
  int cudaFlag;

  //! Minimum vertex x coordinate
  real minx;
  //! Maximum vertex x coordinate
  real maxx;
  //! Minimum vertex y coordinate
  real miny;
  //! Maximum vertex y coordinate
  real maxy;

  //! Morton values for every vertex
  Array<unsigned int> *vertexMorton;
  //! Morton values for every object (vertex, triangle or edge)
  Array<unsigned int> *mortValues;
  //! Indexing array
  Array<unsigned int> *index;
  //! Array for inverse reindexing
  Array<unsigned int> *inverseIndex;

  //! Calculate Morton values for every vertex
  void CalcValues(Connectivity * const connectivity);

  //! Reorder vertices, adjusting triangleVertices
  template<class realNeq>
    void OrderVertex(Connectivity * const connectivity,
                     Array<realNeq> * const vertexState);
  //! Reorder triangles, adjusting edgeTriangles
  void OrderTriangle(Connectivity * const connectivity,
                     Array<int> * const triangleWantRefine);
  //! Reorder edges, adjusting triangleEdges
  void OrderEdge(Connectivity * const connectivity);
};

}  // namespace astrix

#endif  // ASTRIX_MORTON_H
