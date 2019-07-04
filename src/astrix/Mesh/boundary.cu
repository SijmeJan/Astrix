// -*-c++-*-
/*! \file boundary.cu
\brief Functions for creating mesh boundaries

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <stdexcept>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "../Common/cudaLow.h"
#include "./Delaunay/delaunay.h"
#include "./Refine/refine.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//##############################################################################
/*! \brief Set coordinates for nth outer boundary vertex

Set coordinates for the nth outer boundary vertex on host or device.

\param n Index of outer boundary vertex
\param problemDef Problem definition
\param *pVc Pointer to vertex coordinates (output)
\param nVertexOuterBoundary Total number of vertices on outer boundary vertices
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary*/
//##############################################################################

__host__ __device__
void SetVertexOuterBoundarySingle(int n, ProblemDefinition problemDef,
                                  real2 *pVc, int nVertexOuterBoundary,
                                  int periodicFlagX, int periodicFlagY,
                                  real minx, real maxx, real miny, real maxy)
{
  if (problemDef == PROBLEM_RIEMANN) {
    if (n == 0) {
      pVc[n].x = maxx;
      pVc[n].y = maxy;
    }
    if (n == 1) {
      pVc[n].x = minx;
      pVc[n].y = maxy;
    }
    if (n == 2) {
      pVc[n].x = minx;
      pVc[n].y = miny;
    }
    if (n == 3) {
      pVc[n].x = maxx;
      pVc[n].y = miny;
    }
  }

  if (problemDef == PROBLEM_SOD ||
      problemDef == PROBLEM_BLAST ||
      problemDef == PROBLEM_GAUSS) {
    if (periodicFlagY == 1) {
      if (n == 0) {
        pVc[n].x = maxx;
        pVc[n].y = 0.612345543*(maxy - miny) + miny;
      }
      if (n == 1) {
        pVc[n].x = minx;
        pVc[n].y = 0.7564387543*(maxy - miny) + miny;
      }
      if (n == 2) {
        pVc[n].x = minx;
        pVc[n].y = 0.45576498332*(maxy - miny) + miny;
      }
      if (n == 3) {
        pVc[n].x = maxx;
        pVc[n].y = 0.41766758765*(maxy - miny) + miny;
      }
    } else {
      if (n == 0) {
        pVc[n].x = maxx;
        pVc[n].y = maxy;
      }
      if (n == 1) {
        pVc[n].x = minx;
        pVc[n].y = maxy;
      }
      if (n == 2) {
        pVc[n].x = minx;
        pVc[n].y = miny;
      }
      if (n == 3) {
        pVc[n].x = maxx;
        pVc[n].y = miny;
      }
    }
  }

  if (problemDef == PROBLEM_KH ||
      problemDef == PROBLEM_LINEAR ||
      problemDef == PROBLEM_VORTEX ||
      problemDef == PROBLEM_NOH ||
      problemDef == PROBLEM_SOURCE ||
      problemDef == PROBLEM_DISC ||
      problemDef == PROBLEM_PLANET) {
    if (n == 0) {
      pVc[n].x = 0.63689475*(maxx - minx) + minx;
      pVc[n].y = 0.60565906*(maxy - miny) + miny;
    }
    if (n == 1) {
      pVc[n].x = 0.38659404*(maxx - minx) + minx;
      pVc[n].y = 0.62345630*(maxy - miny) + miny;
    }
    if (n == 2) {
      pVc[n].x = 0.43564807*(maxx - minx) + minx;
      pVc[n].y = 0.41564785*(maxy - miny) + miny;
    }
    if (n == 3) {
      pVc[n].x = 0.61678539*(maxx - minx) + minx;
      pVc[n].y = 0.43436738*(maxy - miny) + miny;
    }

    if (periodicFlagX == 0) {
      if (n == 0) pVc[n].x = maxx;
      if (n == 1) pVc[n].x = minx;
      if (n == 2) pVc[n].x = minx;
      if (n == 3) pVc[n].x = maxx;
    }
    if (periodicFlagY == 0) {
      if (n == 0) pVc[n].y = maxy;
      if (n == 1) pVc[n].y = maxy;
      if (n == 2) pVc[n].y = miny;
      if (n == 3) pVc[n].y = miny;
    }
  }

  if (problemDef == PROBLEM_CYL) {
    if (n == 0) {
      pVc[n].x = 0.9340376*(maxx - minx) + minx;
      pVc[n].y = 0.9743867*(maxy - miny) + miny;
    }
    if (n == 1) {
      pVc[n].x = 0.06438064*(maxx - minx) + minx;
      pVc[n].y = 0.92467259*(maxy - miny) + miny;
    }
    if (n == 2) {
      pVc[n].x = 0.050386*(maxx - minx) + minx;
      pVc[n].y = 0.075728*(maxy - miny) + miny;
    }
    if (n == 3) {
      pVc[n].x = 0.91567*(maxx - minx) + minx;
      pVc[n].y = 0.04357*(maxy - miny) + miny;
    }

    if (periodicFlagX == 0) {
      if (n == 0) pVc[n].x = maxx;
      if (n == 1) pVc[n].x = minx;
      if (n == 2) pVc[n].x = minx;
      if (n == 3) pVc[n].x = maxx;
    }
    if (periodicFlagY == 0) {
      if (n == 0) pVc[n].y = maxy;
      if (n == 1) pVc[n].y = maxy;
      if (n == 2) pVc[n].y = miny;
      if (n == 3) pVc[n].y = miny;
    }
  }
}

//##############################################################################
/*! \brief Set coordinates for nth inner boundary vertex

Set coordinates for the nth inner boundary vertex on host or device. Note that these should be inserted in clockwise order to facilitate easy redundant triangle removsl later.

\param n Index of inner boundary vertex
\param problemDef Problem definition
\param *pVc Pointer to vertex coordinates (output)
\param nVertexInnerBoundary Total number of vertices on inner boundary
\param nVertexOuterBoundary Total number of vertices on outer boundary
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary*/
//##############################################################################

__host__ __device__
void SetVertexInnerBoundarySingle(int n, ProblemDefinition problemDef,
                                  real2 *pVc, int nVertexInnerBoundary,
                                  int nVertexOuterBoundary,
                                  real minx, real maxx, real miny, real maxy)
{
  if (problemDef == PROBLEM_CYL) {
    real phi = -2.0*M_PI*(real) n/(real) nVertexInnerBoundary;

    pVc[n + nVertexOuterBoundary].x = 0.5*cos(phi);
    pVc[n + nVertexOuterBoundary].y = 0.5*sin(phi);
  }

  if (problemDef == PROBLEM_PLANET) {
    real phi = -2.0*M_PI*(real) n/(real) nVertexInnerBoundary;

    pVc[n + nVertexOuterBoundary].x = 0.0005*cos(phi);
    pVc[n + nVertexOuterBoundary].y = M_PI + 0.0005*sin(phi);
  }

  return;
}

//######################################################################
/*! \brief Kernel for setting outer boundary vertices.

\param problemDef Problem definition
\param *pVc Pointer to vertex coordinates (output)
\param nVertexOuterBoundary Total number of vertices on outer boundary vertices
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary*/
//######################################################################

__global__ void
devSetVertexOuterBoundary(ProblemDefinition problemDef,
                          real2 *pVc, int nVertexOuterBoundary,
                          int periodicFlagX, int periodicFlagY,
                          real minx, real maxx, real miny, real maxy)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertexOuterBoundary) {
    SetVertexOuterBoundarySingle(i, problemDef, pVc,
                                 nVertexOuterBoundary,
                                 periodicFlagX, periodicFlagY,
                                 minx, maxx, miny, maxy);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Kernel for setting inner boundary vertices

\param problemDef Problem definition
\param *pVc Pointer to vertex coordinates (output)
\param nVertexInnerBoundary Total number of vertices on inner boundary
\param nVertexOuterBoundary Total number of vertices on outer boundary
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary*/
//######################################################################

__global__ void
devSetVertexInnerBoundary(ProblemDefinition problemDef, real2 *pVc,
                          int nVertexInnerBoundary, int nVertexOuterBoundary,
                          real minx, real maxx, real miny, real maxy)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertexInnerBoundary) {
    SetVertexInnerBoundarySingle(i, problemDef, pVc,
                                 nVertexInnerBoundary, nVertexOuterBoundary,
                                 minx, maxx, miny, maxy);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}



//#########################################################################
/*! Construct domain edges. For problems with a simple domain only 4 outer boundary vertices are needed. If on the other hand the outer boundary is for example a circle, all vertices on this circle need to be specified and inserted first. In addition any completely internal boundaries need to be inserted at this point as well.*/
//#########################################################################

void Mesh::ConstructBoundaries(Array <real2> *vertexBoundaryCoordinates)
{
  // Flag whether vertex positions are read from file
  int fixedVerticesFlag = 0;
  if (vertexBoundaryFlag != 0)
    fixedVerticesFlag = 1;

  if (verboseLevel > 0)
    std::cout << "Constructing boundaries..." << std::endl;

  int nVertexOuterBoundary = 0;
  int nVertexInnerBoundary = 0;

  if (vertexBoundaryCoordinates == 0) {
    // Not reading vertex positions from file: create
    vertexBoundaryCoordinates = new Array<real2>(1, cudaFlag);

    if (meshParameter->problemDef == PROBLEM_RIEMANN ||
        meshParameter->problemDef == PROBLEM_SOD ||
        meshParameter->problemDef == PROBLEM_BLAST ||
        meshParameter->problemDef == PROBLEM_KH ||
        meshParameter->problemDef == PROBLEM_LINEAR ||
        meshParameter->problemDef == PROBLEM_VORTEX ||
        meshParameter->problemDef == PROBLEM_NOH ||
        meshParameter->problemDef == PROBLEM_SOURCE ||
        meshParameter->problemDef == PROBLEM_GAUSS ||
        meshParameter->problemDef == PROBLEM_DISC) {
      nVertexOuterBoundary = 4;
      nVertexInnerBoundary = 0;
    }
    if (meshParameter->problemDef == PROBLEM_CYL ||
        meshParameter->problemDef == PROBLEM_PLANET) {
      nVertexOuterBoundary = 4;
      nVertexInnerBoundary = 0;
    }

    // Happens if problem is not defined
    if (nVertexOuterBoundary == 0) {
      std::cout << "Error: no outer boundary vertices!" << std::endl;
      throw std::runtime_error("");
    }

    vertexBoundaryCoordinates->SetSize(nVertexOuterBoundary +
                                       nVertexInnerBoundary);
    real2 *pVbc = vertexBoundaryCoordinates->GetPointer();

    // Fill coordinates of outer boundary vertices
    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devSetVertexOuterBoundary,
                                         (size_t) 0, 0);

      devSetVertexOuterBoundary<<<nBlocks, nThreads>>>
        (meshParameter->problemDef, pVbc, nVertexOuterBoundary,
         meshParameter->periodicFlagX, meshParameter->periodicFlagY,
         meshParameter->minx, meshParameter->maxx, meshParameter->miny,
         meshParameter->maxy);

      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int n = 0; n < nVertexOuterBoundary; n++)
        SetVertexOuterBoundarySingle(n, meshParameter->problemDef, pVbc,
                                     nVertexOuterBoundary,
                                     meshParameter->periodicFlagX,
                                     meshParameter->periodicFlagY,
                                     meshParameter->minx,
                                     meshParameter->maxx,
                                     meshParameter->miny,
                                     meshParameter->maxy);
    }

    // Fill coordinates of inner boundary
    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                         devSetVertexInnerBoundary,
                                         (size_t) 0, 0);

      devSetVertexInnerBoundary<<<nBlocks, nThreads>>>
        (meshParameter->problemDef, pVbc,
         nVertexInnerBoundary, nVertexOuterBoundary,
         meshParameter->minx, meshParameter->maxx,
         meshParameter->miny, meshParameter->maxy);

      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int n = 0; n < nVertexInnerBoundary; n++)
        SetVertexInnerBoundarySingle(n, meshParameter->problemDef, pVbc,
                                     nVertexInnerBoundary, nVertexOuterBoundary,
                                     meshParameter->minx, meshParameter->maxx,
                                     meshParameter->miny, meshParameter->maxy);
    }
  } else {
    nVertexOuterBoundary = vertexBoundaryCoordinates->GetSize();
    nVertexInnerBoundary = 0;
  }

  // Find minimum/maximum x and y
  real minxBoundary = vertexBoundaryCoordinates->MinimumComb<real>(0);
  real minyBoundary = vertexBoundaryCoordinates->MinimumComb<real>(1);
  real maxxBoundary = vertexBoundaryCoordinates->MaximumComb<real>(0);
  real maxyBoundary = vertexBoundaryCoordinates->MaximumComb<real>(1);

  real minx = minxBoundary - (maxxBoundary - minxBoundary);
  real maxx = maxxBoundary + (maxxBoundary - minxBoundary);
  real miny = minyBoundary - (maxyBoundary - minyBoundary);
  real maxy = maxyBoundary + (maxyBoundary - minyBoundary);

  //-----------------------------------------------------------------
  // Create initial mesh, consisting of 4 vertices making up a square,
  // large enough to contain all boundary vertices. These 4 vertices
  // will be removed later.
  //-----------------------------------------------------------------

  // Starting number of vertices, edges and triangles
  int nVertex   = 4;
  int nEdge     = 5;
  int nTriangle = 2;

  // We do this step on the host!
  connectivity->vertexCoordinates->SetSizeHost(nVertex);
  connectivity->triangleVertices->SetSizeHost(nTriangle);
  connectivity->triangleEdges->SetSizeHost(nTriangle);
  connectivity->edgeTriangles->SetSizeHost(nEdge);

  real2 *pVc = connectivity->vertexCoordinates->GetHostPointer();
  int3 *pTv = connectivity->triangleVertices->GetHostPointer();
  int3 *pTe = connectivity->triangleEdges->GetHostPointer();
  int2 *pEt = connectivity->edgeTriangles->GetHostPointer();

  // First four vertices
  pVc[0].x = minx;     // Upper left
  pVc[1].x = maxx;     // Upper right
  pVc[2].x = minx;     // Lower left
  pVc[3].x = maxx;     // Lower right

  pVc[0].y = maxy;
  pVc[1].y = maxy;
  pVc[2].y = miny;
  pVc[3].y = miny;

  // Vertices for triangles
  pTv[0].x = 2;
  pTv[0].y = 1;
  pTv[0].z = 0;
  pTv[1].x = 2;
  pTv[1].y = 3;
  pTv[1].z = 1;

  // Edges for every triangle
  pTe[0].x = 1;
  pTe[0].y = 0;
  pTe[0].z = 4;
  pTe[1].x = 2;
  pTe[1].y = 3;
  pTe[1].z = 1;

  // Triangles for every edge
  pEt[0].x = -1;
  pEt[0].y = 0;
  pEt[1].x = 1;
  pEt[1].y = 0;
  pEt[2].x = 1;
  pEt[2].y = -1;
  pEt[3].x = -1;
  pEt[3].y = 1;
  pEt[4].x = 0;
  pEt[4].y = -1;

  // Copy data to device
  if (cudaFlag == 1) connectivity->CopyToDevice();

  //-----------------------------------------------------------------
  // Now add all boundary vertices to the Mesh
  //-----------------------------------------------------------------

  // All triangles can be refined in principle
  //triangleWantRefine->SetSize(nTriangle);
  //triangleWantRefine->SetToValue(1);

  Array<int> *vertexOrder = new Array<int>(1, cudaFlag, (unsigned int) 4);
  vertexOrder->SetToSeries();

  int nAdded = refine->AddVertices(connectivity,
                                   meshParameter,
                                   predicates,
                                   delaunay,
                                   vertexBoundaryCoordinates,
                                   vertexOrder);

  if (nAdded != nVertexOuterBoundary + nVertexInnerBoundary) {
    std::cout << "Error in Mesh::ConstructBoundaries(): number of added vertices does not match number of boundary vertices" << std::endl;
    throw std::runtime_error("");
  }

  delete vertexBoundaryCoordinates;

  //-----------------------------------------------------------------
  // Now remove the 4 initial vertices and the associated triangles
  // and edges.
  //-----------------------------------------------------------------

  delaunay->RecoverSegments(connectivity, predicates,
                            meshParameter, vertexOrder);

  RemoveRedundant(vertexOrder, nVertexOuterBoundary);
  delaunay->MakeDelaunay<real>(connectivity, 0,
                               predicates, meshParameter,
                               0,
                               0,
                               0,
                               0);
  delete vertexOrder;

  //-----------------------------------------------------------------
  // Extra work for periodic domains: connect left of domain to the
  // right and/or top of domain to the bottom.
  //-----------------------------------------------------------------

  MakePeriodic();

  // Add extra vertices to break symmetry
  if (meshParameter->periodicFlagX == 0 &&
      meshParameter->periodicFlagY == 0 &&
      fixedVerticesFlag == 0) {
    Array<real2> *vertexExtra = new Array<real2>(1, cudaFlag, 2);
    Array<int> *vertexExtraOrder = new Array<int>(1, cudaFlag, 2);

    real2 temp;
    temp.y = (meshParameter->maxy - meshParameter->miny)/sqrt(2.0) +
      meshParameter->miny;
    temp.x = meshParameter->minx;

    vertexExtra->SetSingleValue(temp, 0);

    temp.x = (meshParameter->maxx - meshParameter->minx)/M_PI +
      meshParameter->minx;
    temp.y = meshParameter->maxy;

    vertexExtra->SetSingleValue(temp, 1);

    nAdded = refine->AddVertices(connectivity,
                                 meshParameter,
                                 predicates,
                                 delaunay,
                                 vertexExtra,
                                 vertexExtraOrder);

    if (verboseLevel > 0)
      std::cout << "Added " << nAdded << " extra vertices" << std::endl;

    delete vertexExtra;
    delete vertexExtraOrder;
  }

  if (verboseLevel > 0)
    std::cout << "Boundaries done" << std::endl;
}

}  // namespace astrix
