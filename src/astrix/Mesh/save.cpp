// -*-c++-*-
/*! \file save.cpp
\brief File containing functions to save and restore mesh to and from disk.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "./Connectivity/connectivity.h"

namespace astrix {

//#########################################################################
/*! Save vertexCoordinates in a vertex file, triangleVertices and triangleEdges in a triangle file, and edgeTriangles in an edge file.

\param nSave Number of save, used to generate file names*/
//#########################################################################

void Mesh::Save(int nSave)
{
  connectivity->Save(nSave);
}

//#########################################################################
/*! Read mesh from disk as it was saved under number \a nSave. In addition, calculate triangle normals, vertex areas and find boundary vertices.

  \param nSave Number of save to restore*/
//#########################################################################

void Mesh::ReadFromDisk(int nSave)
{
  connectivity->ReadFromDisk(nSave);

  CalcNormalEdge();
  connectivity->CalcVertexArea(GetPx(), GetPy());
  FindBoundaryVertices();

  std::cout << "Done reading mesh from disk" << std::endl;
}

}  // namespace astrix
