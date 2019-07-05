/*! \file parameter_groups.h
\brief Header file for classes grouping various parameters

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_PARAMETER_GROUPS_H
#define ASTRIX_PARAMETER_GROUPS_H

namespace astrix {

class Mesh;

//! Class containing pointers to triangle normals
class TriangleNormals
{
 public:
  //! Constructor, set data
  TriangleNormals(Mesh *mesh) {
    pTn1 = mesh->TriangleEdgeNormalsData(0);
    pTn2 = mesh->TriangleEdgeNormalsData(1);
    pTn3 = mesh->TriangleEdgeNormalsData(2);

    pTl = mesh->TriangleEdgeLengthData();
  }

  //! Destructor.
  __host__ __device__
  ~TriangleNormals(){};

  const real2 *pTn1, *pTn2, *pTn3;
  const real3 *pTl;
};

//! Class containing mesh size (number of triangles etc)
class MeshSize
{
 public:
  //! Constructor, set data
  MeshSize(Mesh *mesh) {
    nVertex = mesh->GetNVertex();
    nTriangle = mesh->GetNTriangle();
    nEdge = mesh->GetNEdge();
  }

  //! Destructor.
  __host__ __device__
  ~MeshSize(){};

  int nVertex, nTriangle, nEdge;
};

}

#endif  // ASTRIX_PARAMETER_GROUPS_H
