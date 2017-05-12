// -*-c++-*-
/*! \file connectivity.cpp
\brief Functions for creating Connectivity object

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./connectivity.h"

namespace astrix {

//#########################################################################
/*! Constructor for Connectivity class. Memory is allocated in large chunks to
minimise any further calls to cudaMalloc when improving the Mesh.

\param _cudaFlag Flag whether Arrays reside on host (0) or device (1)
*/
//#########################################################################

Connectivity::Connectivity(int _cudaFlag)
{
  cudaFlag = _cudaFlag;

  // Allocate arrays (note large size to minimise additional allocation calls)
  vertexCoordinates = new Array<real2>(1, cudaFlag, 0, 128*8192);
  triangleVertices = new Array<int3>(1, cudaFlag, 0, 128*8192);
  triangleEdges = new Array<int3>(1, cudaFlag, 0, 128*8192);
  edgeTriangles = new Array<int2>(1, cudaFlag, 0, 128*8192);
  vertexArea = new Array<real>(1, cudaFlag, 0, 128*8192);
}

//#########################################################################
// Destructor, releases memory
//#########################################################################

Connectivity::~Connectivity()
{
  // Release memory
  delete vertexCoordinates;
  delete triangleVertices;
  delete triangleEdges;
  delete edgeTriangles;
  delete vertexArea;
}

//#########################################################################
/*! Move whole class to device or host, depending on current value of \a
cudaFlag*/
//#########################################################################

void Connectivity::Transform()
{
  if (cudaFlag == 1) {
    vertexCoordinates->TransformToHost();
    triangleVertices->TransformToHost();
    triangleEdges->TransformToHost();
    edgeTriangles->TransformToHost();
    vertexArea->TransformToHost();
    cudaFlag = 0;
  } else {
    vertexCoordinates->TransformToDevice();
    triangleVertices->TransformToDevice();
    triangleEdges->TransformToDevice();
    edgeTriangles->TransformToDevice();
    vertexArea->TransformToDevice();
    cudaFlag = 1;
  }
}

//#########################################################################
/*! Copy all data currently residing on device to host. Unlike during a
transform, \a cudaFlag is not changed*/
//#########################################################################

void Connectivity::CopyToHost()
{
  vertexCoordinates->CopyToHost();
  triangleVertices->CopyToHost();
  triangleEdges->CopyToHost();
  edgeTriangles->CopyToHost();
  vertexArea->CopyToHost();
}

//#########################################################################
/*! Copy all data currently residing on the host to the device. Unlike during a
transform, \a cudaFlag is not changed*/
//#########################################################################

void Connectivity::CopyToDevice()
{
  vertexCoordinates->CopyToDevice();
  triangleVertices->CopyToDevice();
  triangleEdges->CopyToDevice();
  edgeTriangles->CopyToDevice();
  vertexArea->CopyToDevice();
}

}  // namespace astrix
