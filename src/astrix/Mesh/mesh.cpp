// -*-c++-*-
/*! \file mesh.cpp
\brief Functions for creating Mesh object

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./Predicates/predicates.h"
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
           const char *fileName, Device *device, int restartNumber)
{
  verboseLevel = meshVerboseLevel;
  debugLevel = meshDebugLevel;
  cudaFlag = meshCudaFlag;

  meshParameter = new MeshParameter;
  connectivity = new Connectivity(cudaFlag);
  predicates = new Predicates(device);
  morton = new Morton(cudaFlag);
  delaunay = new Delaunay(cudaFlag, debugLevel, verboseLevel);
  refine = new Refine(cudaFlag, std::max(1, debugLevel), verboseLevel);
  coarsen = new Coarsen(cudaFlag, std::max(1, debugLevel), verboseLevel);

  // Define arrays
  vertexBoundaryFlag = new Array<int>(1, cudaFlag);

  triangleEdgeNormals = new Array<real2>(3, cudaFlag);
  triangleEdgeLength = new Array<real3>(1, cudaFlag);

  try {
    Init(fileName, restartNumber);
  }
  catch (...) {
    std::cout << "Error initializing mesh" << std::endl;

    // Clean up; destructor won't be called
    delete vertexBoundaryFlag;

    delete triangleEdgeNormals;
    delete triangleEdgeLength;

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

  delete triangleEdgeNormals;
  delete triangleEdgeLength;

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

void Mesh::Init(const char *fileName, int restartNumber)
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

  //--------------------------------------------------------------------
  // Output some mesh parameters to stdout
  //--------------------------------------------------------------------

  if (verboseLevel > 0) {
    std::cout << "Creating new mesh:" << std::endl;
    std::cout << "  Domain: " << meshParameter->minx << " < x < "
              << meshParameter->maxx << ", "
              << meshParameter->miny << " < y < "
              << meshParameter->maxy << std::endl;
    if (meshParameter->periodicFlagX)
      std::cout << "  Periodic in x" << std::endl;
    else
      std::cout << "  Not periodic in x" << std::endl;
    if (meshParameter->periodicFlagY)
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
      Array <real2> *vertexCoordinatesToAdd = 0;
      Array <real2> *vertexOuterBoundary = 0;

      // If boundary vertex list available...
      if (!meshParameter->vertexBoundaryInputFile.empty()) {
        if (verboseLevel > 0)
          std::cout << "Reading boundary vertex file..." << std::endl;

        // Read vertex coordinates into Array
        Array <real> *temp;
        try {
          temp = new Array <real> (meshParameter->vertexBoundaryInputFile);
        }
        catch (...) {
          throw;
        }

        // Need array to be 2D
        if (temp->GetDimension() != 2) {
          std::cout << "Error: vertex input file should have 2 columns"
                    << std::endl;
          throw std::runtime_error("");
        }

        // Convert to real2
        vertexCoordinatesToAdd = new Array <real2>;
        temp->MakeIntrinsic2D(vertexCoordinatesToAdd);
        delete temp;

        real2 *pVc = vertexCoordinatesToAdd->GetPointer();
        int nAdd = vertexCoordinatesToAdd->GetSize();
        for (int i = 0; i < nAdd; i++) {
          real2 a = pVc[i];
          int next = i + 1;
          if (next >= nAdd) next -= nAdd;

          real2 b = pVc[next];

          for (int j = 0; j < nAdd; j++) {
            real2 v = pVc[j];

            if ((a.x - v.x)*(b.x - v.x) + (a.y - v.y)*(b.y - v.y) < 0.0) {
              std::cout << "Error: encroached segment detected in input boundary vertices. Possibly due to small input angle." << std::endl;
              throw std::runtime_error("");
            }
          }
        }

        // If boundary vertices are given, pass these to constructBoundaries
        vertexOuterBoundary =
          new Array<real2>(1, 0, vertexCoordinatesToAdd->GetSize());
        vertexOuterBoundary->SetEqual(vertexCoordinatesToAdd);
      } else {
        // If internal vertex list available...
        if (!meshParameter->vertexInputFile.empty()) {
          if (verboseLevel > 0)
            std::cout << "Reading internal vertex file..." << std::endl;

          // Read vertex coordinates into Array
          Array <real> *temp;
          try {
            temp = new Array <real> (meshParameter->vertexInputFile);
          }
          catch (...) {
            throw;
          }

          // Need array to be 2D
          if (temp->GetDimension() != 2) {
            std::cout << "Error: vertex input file should have 2 columns"
                      << std::endl;
            throw std::runtime_error("");
          }

          // Convert to real2
          vertexCoordinatesToAdd = new Array <real2>;
          temp->MakeIntrinsic2D(vertexCoordinatesToAdd);
          delete temp;

          /*
          real2 origin;
          origin.x = 1.0;//0.5*(meshParameter->maxx + meshParameter->minx);
          origin.y = 1.0;//0.5*(meshParameter->maxy + meshParameter->miny);

          vertexCoordinatesToAdd->SortCounterClock<real>(origin);
          */

          // If internal vertices are given, but no boundary vertices,
          // need convex hull of point set to determine initial triangulation
          if (verboseLevel > 0)
            std::cout << "Calculating convex hull..." << std::endl;

          // Gift wrap to find convex hull:
          // convex hull vertices are removed from vertexCoordinatesAdd!!
          vertexOuterBoundary = ConvexHull(vertexCoordinatesToAdd);
        }
      }

      // Create unstructured mesh. Note that vertexOuterboundary is either 0
      // or contains either the boundary vertices read from file or the convex
      // hull of internal vertices read from file.
      try {
        // Set up mesh boundaries
        ConstructBoundaries(vertexOuterBoundary);
      }
      catch (...) {
        std::cout << "Error constructing mesh boundaries" << std::endl;
        throw;
      }

      // If internal vertex list available...
      if (!meshParameter->vertexInputFile.empty()) {
        // If also boundary vertices available from file, we need
        if (!meshParameter->vertexBoundaryInputFile.empty()) {

          if (verboseLevel > 0)
            std::cout << "Reading internal vertex file..." << std::endl;

          // Read vertex coordinates into Array
          Array <real> *temp;
          try {
            temp = new Array <real> (meshParameter->vertexInputFile);
          }
          catch (...) {
            throw;
          }

          // Need array to be 2D
          if (temp->GetDimension() != 2) {
            std::cout << "Error: vertex input file should have 2 columns"
                      << std::endl;
            throw std::runtime_error("");
          }

          // Convert to real2
          vertexCoordinatesToAdd = new Array <real2>;
          temp->MakeIntrinsic2D(vertexCoordinatesToAdd);
          delete temp;
        }

        // Add vertices into mesh
        refine->AddVertices(connectivity,
                            meshParameter,
                            predicates,
                            delaunay,
                            vertexCoordinatesToAdd,
                            0);

        // Calculate triangle normals and areas
        CalcNormalEdge();
        connectivity->CalcVertexArea(GetPx(), GetPy());
        FindBoundaryVertices();

        delete vertexCoordinatesToAdd;

      } else {
        try {
          // Refine mesh to base resolution
          ImproveQuality<real>((Array<real> *)0, 0, 0);
        }
        catch (...) {
          std::cout << "Error refining initial mesh" << std::endl;
          throw;
        }
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
                  (real)(Array<real2>::memAllocatedHost) +
                  (real)(Array<real3>::memAllocatedHost) +
                  (real)(Array<real4>::memAllocatedHost) +
                  (real)(Array<int>::memAllocatedHost) +
                  (real)(Array<int2>::memAllocatedHost) +
                  (real)(Array<int3>::memAllocatedHost) +
                  (real)(Array<int4>::memAllocatedHost) +
                  (real)(Array<unsigned int>::memAllocatedHost))/
      (real) (1073741824) << " Gb, on device: "
              << ((real)(Array<real>::memAllocatedDevice) +
                  (real)(Array<real2>::memAllocatedDevice) +
                  (real)(Array<real3>::memAllocatedDevice) +
                  (real)(Array<real4>::memAllocatedDevice) +
                  (real)(Array<int>::memAllocatedDevice) +
                  (real)(Array<int2>::memAllocatedDevice) +
                  (real)(Array<int3>::memAllocatedDevice) +
                  (real)(Array<int4>::memAllocatedDevice) +
                  (real)(Array<unsigned int>::memAllocatedDevice))/
      (real) (1073741824) << " Gb" << std::endl;
  }
}

//#########################################################################
// Return number of vertices
//#########################################################################

int Mesh::GetNVertex()
{
  return connectivity->vertexCoordinates->GetSize();
}

//#########################################################################
// Return number of triangles
//#########################################################################

int Mesh::GetNTriangle()
{
  return connectivity->triangleVertices->GetSize();
}

//#########################################################################
// Return number of edges
//#########################################################################

int Mesh::GetNEdge()
{
  return connectivity->edgeTriangles->GetSize();
}

//#########################################################################
// Return if mesh is adaptive
//#########################################################################

int Mesh::IsAdaptive()
{
  return meshParameter->adaptiveMeshFlag;
}

//#########################################################################
// Return total vertex area
//#########################################################################

real Mesh::GetTotalArea()
{
  return connectivity->vertexArea->Sum();
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
  return connectivity->vertexArea->GetPointer();
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
    triangleEdgeNormals->TransformToHost();
    triangleEdgeLength->TransformToHost();

    cudaFlag = 0;
  } else {
    vertexBoundaryFlag->TransformToDevice();
    triangleEdgeNormals->TransformToDevice();
    triangleEdgeLength->TransformToDevice();

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

real Mesh::GetMinX()
{
  return meshParameter->minx;
}

real Mesh::GetMaxX()
{
  return meshParameter->maxx;
}

real Mesh::GetMinY()
{
  return meshParameter->miny;
}

real Mesh::GetMaxY()
{
  return meshParameter->maxy;
}

}  // namespace astrix
