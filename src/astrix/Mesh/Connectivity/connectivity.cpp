// -*-c++-*-
/*! \file connectivity.cpp
\brief Functions for creating Connectivity object*/

#include <iostream>
#include <cuda_runtime_api.h>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./connectivity.h"

namespace astrix {

//#########################################################################
/*! Constructor for Connectivity class. Memory is allocated in large chunks to minimise any further calls to cudaMalloc when improving the Mesh.*/
//#########################################################################

Connectivity::Connectivity(int _cudaFlag)
{
  cudaFlag = _cudaFlag;

  // Allocate arrays (note large size to minimise additional allocation calls)
  vertexCoordinates = new Array<real2>(1, cudaFlag, 0, 128*8192);
  triangleVertices = new Array<int3>(1, cudaFlag, 0, 128*8192);
  triangleEdges = new Array<int3>(1, cudaFlag, 0, 128*8192);
  edgeTriangles = new Array<int2>(1, cudaFlag, 0, 128*8192);
}

//#########################################################################
//#########################################################################

Connectivity::~Connectivity()
{
  // Release memory
  delete vertexCoordinates;
  delete triangleVertices;
  delete triangleEdges;
  delete edgeTriangles;
}

//#########################################################################
/*! Move whole class to device or host, depending on current value of \a cudaFlag*/
//#########################################################################

void Connectivity::Transform()
{
  if (cudaFlag == 1) {
    vertexCoordinates->TransformToHost();
    triangleVertices->TransformToHost();
    triangleEdges->TransformToHost();
    edgeTriangles->TransformToHost();
    cudaFlag = 0;
  } else {
    vertexCoordinates->TransformToDevice();
    triangleVertices->TransformToDevice();
    triangleEdges->TransformToDevice();
    edgeTriangles->TransformToDevice();
    cudaFlag = 1;
  }
}

//#########################################################################
//#########################################################################

void Connectivity::CopyToHost()
{
  vertexCoordinates->CopyToHost();
  triangleVertices->CopyToHost();
  triangleEdges->CopyToHost();
  edgeTriangles->CopyToHost();
}

//#########################################################################
//#########################################################################

void Connectivity::CopyToDevice()
{
  vertexCoordinates->CopyToDevice();
  triangleVertices->CopyToDevice();
  triangleEdges->CopyToDevice();
  edgeTriangles->CopyToDevice();
}

}
