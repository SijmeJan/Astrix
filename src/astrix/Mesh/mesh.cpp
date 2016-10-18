// -*-c++-*-
/*! \file mesh.cpp
\brief Functions for creating Mesh object*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "Predicates/predicates.h"
#include "./Morton/morton.h"
#include "./Delaunay/delaunay.h"
#include "./Refine/refine.h"
#include "./Coarsen/coarsen.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"
#include "./mesh.h"

namespace astrix {
  
//#########################################################################
/*! Create Mesh with parameters specified in file \a fileName. 

\param meshVerboseLevel Level of output to stdout
\param meshDebugLevel Level of debugging in Mesh construction and maintenance
\param meshCudaFlag Flag whether to use CUDA device
\param *fileName Input file name
\param *device Pointer to Device class containing information about any CUDA device present
\param restartNumber Number of save file to restore from*/
//#########################################################################

Mesh::Mesh(int meshVerboseLevel, int meshDebugLevel, int meshCudaFlag,
	   const char *fileName, Device *device, int restartNumber,
	   int extraFlag)
{
  verboseLevel = meshVerboseLevel;
  debugLevel = meshDebugLevel;
  cudaFlag = meshCudaFlag;

  meshParameter = new MeshParameter;
  connectivity = new Connectivity(cudaFlag);
  predicates = new Predicates(device);
  morton = new Morton(cudaFlag);
  delaunay = new Delaunay(cudaFlag, debugLevel);
  refine = new Refine(cudaFlag, debugLevel, verboseLevel);
  coarsen = new Coarsen(cudaFlag, debugLevel, verboseLevel);
  
  // Define arrays
  vertexBoundaryFlag = new Array<int>(1, cudaFlag);
  vertexArea = new Array<real>(1, cudaFlag);

  triangleWantRefine = new Array<int>(1, cudaFlag);
  triangleEdgeNormals = new Array<real2>(3, cudaFlag);
  triangleEdgeLength = new Array<real3>(1, cudaFlag);
  triangleErrorEstimate = new Array<real>(1, cudaFlag);

  randomVector = new Array<unsigned int>(1, cudaFlag);
  if (cudaFlag == 1)
    randomVector->TransformToHost();
  randomVector->SetSize(10000000);
  randomVector->SetToRandom();
  if (cudaFlag == 1)
    randomVector->TransformToDevice();

  try {
    Init(fileName, restartNumber, extraFlag);
  }
  catch (...) {
    std::cout << "Error initializing mesh" << std::endl;
    
    // Clean up; destructor won't be called
    delete vertexBoundaryFlag;
    delete vertexArea;
    
    delete triangleWantRefine;
    delete triangleEdgeNormals;
    delete triangleEdgeLength;
    delete triangleErrorEstimate;

    delete randomVector;

    delete predicates;
    delete morton;
    delete delaunay;
    delete refine;
    delete coarsen;
    delete connectivity;
    delete meshParameter;
    
    throw;
  }
}

//#########################################################################
// Mesh destructor
//#########################################################################

Mesh::~Mesh()
{
  delete vertexBoundaryFlag;
  delete vertexArea;

  delete triangleWantRefine;
  delete triangleEdgeNormals;
  delete triangleEdgeLength;
  delete triangleErrorEstimate;

  delete randomVector;
  
  delete predicates;
  delete morton;
  delete delaunay;
  delete refine;
  delete coarsen;
  delete connectivity;
  delete meshParameter;
}

//#########################################################################
/*! Set up the initial Mesh according to parameters specified in \a *fileName. After this input file has been read, we construct the Mesh boundaries and then refine the Mesh until the desired initial resolution is reached.

\param *fileName Input file name
\param restartNumber Number of save file to restore from*/
//#########################################################################

  void Mesh::Init(const char *fileName, int restartNumber, int extraFlag)
{
  //-------------------------------------------------------------------
  // Read mesh input file
  //-------------------------------------------------------------------

  try {
    meshParameter->ReadFromFile(fileName);
  }
  catch (...) {
    std::cout << "Error reading data into MeshParameter" << std::endl;    
    throw;
  }

  if (extraFlag > 0) {
    real nx = (real) (extraFlag - 1);
  
    // Convert nx into base resolution requirement
    meshParameter->baseResolution =
      0.565*((meshParameter->maxx - meshParameter->minx)/nx)*
      ((meshParameter->maxx - meshParameter->minx)/nx);
  }
  
  //--------------------------------------------------------------------
  // Output some mesh parameters to stdout
  //--------------------------------------------------------------------

  if (verboseLevel > 0) {
    std::cout << "Creating new mesh:" << std::endl;
    std::cout << "  Domain: " << meshParameter->minx << " < x < "
	      << meshParameter->maxx << ", " 
	      << meshParameter->miny << " < y < "
	      << meshParameter->maxy << std::endl;
    if(meshParameter->periodicFlagX) 
      std::cout << "  Periodic in x" << std::endl;
    else
      std::cout << "  Not periodic in x" << std::endl;
    if(meshParameter->periodicFlagY) 
      std::cout << "  Periodic in y" << std::endl;
    else
      std::cout << "  Not periodic in y" << std::endl;
  }

  //--------------------------------------------------------------------
  // Create mesh
  //--------------------------------------------------------------------

  // Construct mesh if not restarting
  if (restartNumber == 0) {
    if (meshParameter->structuredFlag == 1 ||
	meshParameter->structuredFlag == 2) {
      // Structured mesh
      CreateStructuredMesh();
    } else {
      // Create unstructured mesh
      try {
	// Set up mesh boundaries
	ConstructBoundaries();
      }
      catch (...) {
	std::cout << "Error constructing mesh boundaries" << std::endl;
	throw;
      }

      try {
	// Refine mesh to base resolution
	ImproveQuality(0, 0.0, 0);
      }
      catch (...) {
	std::cout << "Error refining initial mesh" << std::endl;
	throw;
      }    
    }
  } else {
    try {
      // Read mesh from disk
      ReadFromDisk(restartNumber);
    }
    catch (...) {
      std::cout << "Error reading mesh from disk" << std::endl;
      throw;
    }        
  }

  //--------------------------------------------------------------------
  // Output stats to screen
  //--------------------------------------------------------------------

  if (verboseLevel > 0) {
    OutputStat();
    std::cout << "Done creating mesh." << std::endl;
    std::cout << "Memory allocated on host: "
	      << ((real)(Array<real>::memAllocatedHost) +
		  (real)(Array<int>::memAllocatedHost) +
		  (real)(Array<unsigned int>::memAllocatedHost))/
      (real) (1073741824) << " Gb, on device: "
	      << ((real)(Array<real>::memAllocatedDevice) +
		  (real)(Array<int>::memAllocatedDevice) +
		  (real)(Array<unsigned int>::memAllocatedDevice))/
      (real) (1073741824) << " Gb" << std::endl;
  }

}

//#########################################################################
// Return number of vertices
//#########################################################################

int Mesh::GetNVertex()
{
  return nVertex;
}

//#########################################################################
// Return number of triangles
//#########################################################################

int Mesh::GetNTriangle()
{
  return nTriangle;
}

//#########################################################################
// Return number of edges
//#########################################################################

int Mesh::GetNEdge()
{
  return nEdge;
}

//#########################################################################
// Return if mesh is adaptive
//#########################################################################

int Mesh::IsAdaptive()
{
  return meshParameter->adaptiveMeshFlag;
}


const real2* Mesh::VertexCoordinatesData()
{
  return connectivity->vertexCoordinates->GetPointer();
}

const int* Mesh::VertexBoundaryFlagData()
{
  return vertexBoundaryFlag->GetPointer();
}
  
const real* Mesh::VertexAreaData()
{
  return vertexArea->GetPointer();
}
  
const int3* Mesh::TriangleEdgesData()
{
  return connectivity->triangleEdges->GetPointer();
}

const int3* Mesh::TriangleVerticesData()
{
  return connectivity->triangleVertices->GetPointer();
}

const real2* Mesh::TriangleEdgeNormalsData(int dim)
{
  return triangleEdgeNormals->GetPointer(dim);
}
  
const real3* Mesh::TriangleEdgeLengthData()
{
  return triangleEdgeLength->GetPointer();
}

const int2* Mesh::EdgeTrianglesData()
{
  return connectivity->edgeTriangles->GetPointer();
}

void Mesh::Transform()
{
  connectivity->Transform();

  if (cudaFlag == 1) {
    vertexBoundaryFlag->TransformToHost();
    vertexArea->TransformToHost();
    triangleWantRefine->TransformToHost();
    triangleEdgeNormals->TransformToHost();
    triangleEdgeLength->TransformToHost();
    triangleErrorEstimate->TransformToHost();
    
    //randomVector->TransformToHost();
    
    //vertexRemove->TransformToHost();
    //vertexTriangle->TransformToHost();
    cudaFlag = 0;
  } else {
    vertexBoundaryFlag->TransformToDevice();
    vertexArea->TransformToDevice();
    triangleWantRefine->TransformToDevice();
    triangleEdgeNormals->TransformToDevice();
    triangleEdgeLength->TransformToDevice();
    triangleErrorEstimate->TransformToDevice();
    
    //randomVector->TransformToDevice();
    
    //vertexRemove->TransformToDevice();
    //vertexTriangle->TransformToDevice();
    cudaFlag = 1;
  }
}

real Mesh::GetPx()
{
  return meshParameter->maxx - meshParameter->minx;
}

real Mesh::GetPy()
{
  return meshParameter->maxy - meshParameter->miny;
}

  
}
