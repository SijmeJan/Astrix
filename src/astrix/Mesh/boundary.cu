// -*-c++-*-
/*! \file boundary.cu
\brief Functions for creating mesh boundaries
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
  if (problemDef == PROBLEM_SOD || problemDef == PROBLEM_BLAST) {
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
      problemDef == PROBLEM_YEE ||
      problemDef == PROBLEM_ADVECT) {
    if (n == 0) {
      pVc[n].x = 0.63*(maxx - minx) + minx;
      pVc[n].y = 0.60*(maxy - miny) + miny;
    }
    if (n == 1) {
      pVc[n].x = 0.38*(maxx - minx) + minx;
      pVc[n].y = 0.62*(maxy - miny) + miny;
    }
    if (n == 2) {
      pVc[n].x = 0.43*(maxx - minx) + minx;
      pVc[n].y = 0.41*(maxy - miny) + miny;
    }
    if (n == 3) {
      pVc[n].x = 0.61*(maxx - minx) + minx;
      pVc[n].y = 0.40*(maxy - miny) + miny;
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

  if (problemDef == PROBLEM_RT) {
    if (periodicFlagX == 1) {
      if (n == 0) {
	pVc[n].x = 0.72*(maxx - minx) + minx;
	pVc[n].y = maxy;
      }
      if (n == 1) {
	pVc[n].x = 0.3*(maxx - minx) + minx;
	pVc[n].y = maxy;
      }
      if (n == 2) {
	pVc[n].x = 0.3*(maxx - minx) + minx;
	pVc[n].y = miny;
      }
      if (n == 3) {
	pVc[n].x = 0.71*(maxx - minx) + minx;
	pVc[n].y = miny;
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
}
 
//##############################################################################
/*! \brief Set coordinates for nth inner boundary vertex

Set coordinates for the nth inner boundary vertex on host or device. At the moment, this function does nothing.

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
  return;
}
 
//######################################################################
/*! \brief Kernel for setting outer boundary vertices

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
  
//######################################################################
/*! \brief Flag vertices for removal

Flag whether vertex \a i needs to be removed. Only the first four inserted vertices are flagged for removal 

\param i Index in vertex array
\param pVertexOrder Pointer to vertex insertion order array
\param pVertexRemoveFlag Pointer to output array*/
//######################################################################

__host__ __device__
void FlagVertexRemoveSingle(int i, int *pVertexOrder, int *pVertexRemoveFlag)
{
  int ret = 0;
  // Only flag first four inserted vertices
  if (pVertexOrder[i] < 4) ret = 1;
  pVertexRemoveFlag[i] = ret;
}

//######################################################################
/*! \brief Kernel for flagging vertices for removal. 

Only the first four inserted vertices are flagged for removal 

\param nVertex Total number of vertices in Mesh
\param pVertexOrder Pointer to vertex insertion order array
\param pVertexRemoveFlag Pointer to output array*/
//######################################################################

__global__ void
devFlagVertexRemove(int nVertex, int *pVertexOrder, int *pVertexRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nVertex) {
    FlagVertexRemoveSingle(i, pVertexOrder, pVertexRemoveFlag);

    // Next vertex
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Flag triangles for removal

Flag any triangle located outside the outer boundary for removal. 

\param i Triangle to be checked
\param *pTv Pointer to triangle vertices 
\param *pVertexRemoveFlag Pointer to flags whether vertices will be removed
\param *pVertexOrder Pointer to vertex insertion order
\param nVertexOuterBoundary Number of vertices making up outer boundary
\param *pTriangleRemoveFlag Pointer to output array*/
//######################################################################

__host__ __device__
void FlagTriangleRemoveSingle(int i, int3 *pTv, int *pVertexRemoveFlag,
			      int *pVertexOrder, int nVertexOuterBoundary,
			      int *pTriangleRemoveFlag)
{
  int ret = 0;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  
  // Remove any triangle for which at least one vertex will be removed
  if (pVertexRemoveFlag[a] == 1 ||
      pVertexRemoveFlag[b] == 1 ||
      pVertexRemoveFlag[c] == 1) ret = 1;
  
  int nBad = 0;

  // Remove any triangle for which all vertices are part of the outer boundary
  // and for which vertices occur in wrong order
  if (pVertexOrder[a] < nVertexOuterBoundary + 4 &&
      pVertexOrder[b] < nVertexOuterBoundary + 4 &&
      pVertexOrder[c] < nVertexOuterBoundary + 4)
    nBad = (pVertexOrder[a] > pVertexOrder[b]) +
			(pVertexOrder[b] > pVertexOrder[c]) +
      (pVertexOrder[c] > pVertexOrder[a]);
  
  if (pVertexOrder[a] >= nVertexOuterBoundary + 4 &&
      pVertexOrder[b] >= nVertexOuterBoundary + 4 &&
      pVertexOrder[c] >= nVertexOuterBoundary + 4)
    nBad = (pVertexOrder[a] > pVertexOrder[b]) +
      (pVertexOrder[b] > pVertexOrder[c]) +
      (pVertexOrder[c] > pVertexOrder[a]);
  
  if (nBad > 1) ret = 1;
  
  pTriangleRemoveFlag[i] = ret;
}
  
//######################################################################
/*! \brief Kernel flagging triangles for removal

Flag any triangle for which one of its vertices will be removed for removal. 

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVertexRemoveFlag Pointer to flags whether vertices will be removed
\param *pVertexOrder Pointer to vertex insertion order
\param nVertexOuterBoundary Number of vertices making up outer boundary
\param *pTriangleRemoveFlag Pointer to output array*/
//######################################################################

__global__ void
devFlagTriangleRemove(int nTriangle, int3 *pTv, int *pVertexRemoveFlag,
		      int *pVertexOrder, int nVertexOuterBoundary,
		      int *pTriangleRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FlagTriangleRemoveSingle(i, pTv, pVertexRemoveFlag,
			     pVertexOrder, nVertexOuterBoundary,
			     pTriangleRemoveFlag);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}
    
//######################################################################
/*! \brief Flag unnecessary edges for removal

Flag any edge for which both neighbouring triangles will be removed

\param i Index in edge array
\param *pEt Pointer to array containing edge triangles
\param *pTriangleRemoveFlag Pointer to flags whether triangles will be removed
\param *pEdgeRemoveFlag Pointer to output array*/
//######################################################################

__host__ __device__
void FlagEdgeRemoveSingle(int i, int2 *pEt,
			  int *pTriangleRemoveFlag, int *pEdgeRemoveFlag)
{
  int t1 = pEt[i].x;
  int t2 = pEt[i].y;
  
  if (t1 != -1)
    if (pTriangleRemoveFlag[t1] == 1) t1 = -1;
  if (t2 != -1)
    if (pTriangleRemoveFlag[t2] == 1) t2 = -1;
  int ret = 0;
  if (t1 == -1 && t2 == -1) ret = 1;
  pEdgeRemoveFlag[i] = ret;

  pEt[i].x = t1;
  pEt[i].y = t2;
}
  
//###################################################################### 
/*! \brief Kernel to flag unnecessary edges for removal

Flag any edge for which both neighbouring triangles will be removed

\param nEdge Total number of edges in Mesh
\param *pEt Pointer to array containing edge triangles
\param *pTriangleRemoveFlag Pointer to flags whether triangles will be removed
\param *pEdgeRemoveFlag Pointer to output array*/
//######################################################################

__global__ void
devFlagEdgeRemove(int nEdge, int2 *pEt,
		  int *pTriangleRemoveFlag, int *pEdgeRemoveFlag)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    FlagEdgeRemoveSingle(i, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! \brief Adjust triangle vertices for vertex removal

\param i Index of triangle
\param *pTv Pointer to triangle vertices
\param *pVertexFlagScan Pointer to scanned array of flags whether vertices will be removed*/
//######################################################################

__host__ __device__
void AdjustTriangleVerticesSingle(int i, int3 *pTv, int *pVertexFlagScan)
{
  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  
  pTv[i].x -= pVertexFlagScan[a];
  pTv[i].y -= pVertexFlagScan[b];
  pTv[i].z -= pVertexFlagScan[c];
}
  
//######################################################################
/*! \brief Kernel adjusting triangle vertices for vertex removal

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVertexFlagScan Pointer to scanned array of flags whether vertices will be removed*/
//######################################################################

__global__ void
devAdjustTriangleVertices(int nTriangle, int3 *pTv, int *pVertexFlagScan)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    AdjustTriangleVerticesSingle(i, pTv, pVertexFlagScan);

    // Next triangle
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Adjust edge triangles for triangle removal

\param i Index of edge
\param *pEt Pointer to edge triangles
\param *pTriangleFlagScan Pointer to scanned array of flags whether triangles will be removed*/
//######################################################################

__host__ __device__
void AdjustEdgeTrianglesSingle(int i, int2 *pEt, int *pTriangleFlagScan)
{
  int t1 = pEt[i].x;
  int t2 = pEt[i].y;
  
  if (t1 != -1) t1 -= pTriangleFlagScan[t1];
  if (t2 != -1) t2 -= pTriangleFlagScan[t2];

  pEt[i].x = t1;
  pEt[i].y = t2;
}
  
//######################################################################
/*! \brief Kernel adjusting edge triangles for triangle removal

\param nEdge Total number of edges in Mesh
\param *pEt Pointer to edge triangles
\param *pTriangleFlagScan Pointer to scanned array of flags whether triangles will be removed*/
//######################################################################

__global__ void
devAdjustEdgeTriangles(int nEdge, int2 *pEt, int *pTriangleFlagScan)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nEdge) {
    AdjustEdgeTrianglesSingle(i, pEt, pTriangleFlagScan);

    i += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! \brief Adjust triangle edges for edge removal

\param i Index of triangle
\param *pTe Pointer to triangle edges
\param *pEdgeFlagScan Pointer to scanned array of flags whether edges will be removed*/
//######################################################################

__host__ __device__
void AdjustTriangleEdgesSingle(int i, int3 *pTe, int *pEdgeFlagScan)
{
  pTe[i].x -= pEdgeFlagScan[pTe[i].x];
  pTe[i].y -= pEdgeFlagScan[pTe[i].y];
  pTe[i].z -= pEdgeFlagScan[pTe[i].z];
}
  
//######################################################################
/*! \brief Kernel adjusting triangle edges for edge removal

\param nTriangle Total number of triangles in Mesh
\param *pTe Pointer to triangle edges
\param *pEdgeFlagScan Pointer to scanned array of flags whether edges will be removed*/ 
//######################################################################

__global__ void
devAdjustTriangleEdges(int nTriangle, int3 *pTe, int *pEdgeFlagScan)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    AdjustTriangleEdgesSingle(i, pTe, pEdgeFlagScan);

    i += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! \brief Check if edge between v1 and v2 lies in triangle i

\param i Triangle to be checked
\param v1 Vertex 1
\param v2 Vertex 2
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *eBetween Pointer to output*/
//######################################################################

__host__ __device__
void FindEdgeBetween(int i, unsigned int v1, unsigned int v2,
		     int3 *pTv, int3 *pTe, int *eBetween)
{
  if (pTv[i].y == (int) v1 && pTv[i].x == (int) v2) eBetween[0] = pTe[i].x;  
  if (pTv[i].z == (int) v1 && pTv[i].y == (int) v2) eBetween[0] = pTe[i].y;   
  if (pTv[i].x == (int) v1 && pTv[i].z == (int) v2) eBetween[0] = pTe[i].z;
}
  
//######################################################################
/*! \brief Kernel finding edge between v1 and v2

\param nTriangle Total number of triangles in Mesh
\param v1 Vertex 1
\param v2 Vertex 2
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEdgeBetween Pointer to output*/
//######################################################################

__global__ void
devFindEdgeBetween(int nTriangle, unsigned int v1, unsigned int v2,
		   int3 *pTv, int3 *pTe, int *pEdgeBetween)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    FindEdgeBetween(i, v1, v2, pTv, pTe, pEdgeBetween);

    i += blockDim.x*gridDim.x;
  }
}
  
//#########################################################################
/*! Construct domain edges. For problems with a simple domain only 4 outer boundary vertices are needed. If on the other hand the outer boundary is for example a circle, all vertices on this circle need to be specified and inserted first. In addition any completely internal boundaries need to be inserted at this point as well.  

\param problemDef Problem definition*/
//#########################################################################

void Mesh::ConstructBoundaries()
{
  if (verboseLevel > 0)
    std::cout << "Constructing boundaries..." << std::endl;

  // Coordinates of boundary vertices
  Array<real2> *vertexBoundaryCoordinates = new Array<real2>(1, cudaFlag);

  int nVertexOuterBoundary = 0;
  int nVertexInnerBoundary = 0;

  if (meshParameter->problemDef == PROBLEM_RIEMANN ||
      meshParameter->problemDef == PROBLEM_SOD ||
      meshParameter->problemDef == PROBLEM_BLAST ||
      meshParameter->problemDef == PROBLEM_RT ||
      meshParameter->problemDef == PROBLEM_VORTEX ||
      meshParameter->problemDef == PROBLEM_KH ||
      meshParameter->problemDef == PROBLEM_LINEAR ||
      meshParameter->problemDef == PROBLEM_YEE ||
      meshParameter->problemDef == PROBLEM_ADVECT) {
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

  // Find minimum/maximum x and y
  //real minxOld = meshParameter->minx;
  //real maxxOld = meshParameter->maxx;
  //real minyOld = meshParameter->miny;
  //real maxyOld = meshParameter->maxy;
  
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
  nVertex   = 4;
  nEdge     = 5;
  nTriangle = 2;

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
  triangleWantRefine->SetSize(nTriangle);
  triangleWantRefine->SetToValue(1);
  
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
  
  nVertex = connectivity->vertexCoordinates->GetSize();
  nTriangle = connectivity->triangleVertices->GetSize();
  nEdge = connectivity->edgeTriangles->GetSize();
  
  delete vertexBoundaryCoordinates;

  //-----------------------------------------------------------------
  // Now remove the 4 initial vertices and the associated triangles
  // and edges.
  //-----------------------------------------------------------------

  pTv = connectivity->triangleVertices->GetPointer();
  pTe = connectivity->triangleEdges->GetPointer();
  pEt = connectivity->edgeTriangles->GetPointer();
  
  Array<int> *vertexRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  int *pVertexRemoveFlag = vertexRemoveFlag->GetPointer();
  Array<int> *vertexFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nVertex);
  int *pVertexFlagScan = vertexFlagScan->GetPointer();
  Array<int> *triangleRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  int *pTriangleRemoveFlag = triangleRemoveFlag->GetPointer();
  Array<int> *triangleFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nTriangle);
  int *pTriangleFlagScan = triangleFlagScan->GetPointer();
  Array<int> *edgeRemoveFlag =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);
  int *pEdgeRemoveFlag = edgeRemoveFlag->GetPointer();
  Array<int> *edgeFlagScan =
    new Array<int>(1, cudaFlag, (unsigned int) nEdge);
  int *pEdgeFlagScan = edgeFlagScan->GetPointer();
  
  int *pVertexOrder = vertexOrder->GetPointer();

  // Flag first four vertices to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlagVertexRemove, 
				       (size_t) 0, 0);

    devFlagVertexRemove<<<nBlocks, nThreads>>>
      (nVertex, pVertexOrder, pVertexRemoveFlag);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++) 
      FlagVertexRemoveSingle(i, pVertexOrder, pVertexRemoveFlag);
  }

  // Flag triangles to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlagTriangleRemove, 
				       (size_t) 0, 0);

    devFlagTriangleRemove<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVertexRemoveFlag,
       pVertexOrder, nVertexOuterBoundary, pTriangleRemoveFlag);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++) 
      FlagTriangleRemoveSingle(i, pTv, pVertexRemoveFlag,
			       pVertexOrder, nVertexOuterBoundary,
			       pTriangleRemoveFlag);
  }
  
  // Flag edges to be removed
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devFlagEdgeRemove, 
				       (size_t) 0, 0);

    devFlagEdgeRemove<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++) 
      FlagEdgeRemoveSingle(i, pEt, pTriangleRemoveFlag, pEdgeRemoveFlag);
  }
  
  vertexRemoveFlag->ExclusiveScan(vertexFlagScan, nVertex);

  // Adjust tv's for removed vertices
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devAdjustTriangleVertices, 
       (size_t) 0, 0);

    devAdjustTriangleVertices<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVertexFlagScan);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++) 
      AdjustTriangleVerticesSingle(i, pTv, pVertexFlagScan);
  }
  
  triangleRemoveFlag->ExclusiveScan(triangleFlagScan, nTriangle);

  // Adjust et's for removed triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devAdjustEdgeTriangles, 
				       (size_t) 0, 0);

    devAdjustEdgeTriangles<<<nBlocks, nThreads>>>
      (nEdge, pEt, pTriangleFlagScan);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nEdge; i++) 
      AdjustEdgeTrianglesSingle(i, pEt, pTriangleFlagScan);
  }
  
  edgeRemoveFlag->ExclusiveScan(edgeFlagScan, nEdge);

  // Adjust te's for removed edges
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;
    
    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devAdjustTriangleEdges, 
				       (size_t) 0, 0);

    devAdjustTriangleEdges<<<nBlocks, nThreads>>>
      (nTriangle, pTe, pEdgeFlagScan);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      AdjustTriangleEdgesSingle(i, pTe, pEdgeFlagScan);
  }
  
  vertexRemoveFlag->Invert();
  int nvKeep = vertexRemoveFlag->ExclusiveScan(vertexFlagScan, nVertex);
  connectivity->vertexCoordinates->Compact(nvKeep, vertexRemoveFlag,
					   vertexFlagScan);

  triangleRemoveFlag->Invert();
  int ntKeep = triangleRemoveFlag->ExclusiveScan(triangleFlagScan, nTriangle);
  connectivity->triangleVertices->Compact(ntKeep, triangleRemoveFlag,
					  triangleFlagScan);
  connectivity->triangleEdges->Compact(ntKeep, triangleRemoveFlag,
				       triangleFlagScan);

  edgeRemoveFlag->Invert();
  int neKeep =edgeRemoveFlag->ExclusiveScan(edgeFlagScan, nEdge);
  connectivity->edgeTriangles->Compact(neKeep, edgeRemoveFlag, edgeFlagScan);

  nVertex = nvKeep;
  nTriangle = ntKeep;
  nEdge = neKeep;

  //meshParameter->minx = minxOld;
  //meshParameter->maxx = maxxOld;
  //meshParameter->miny = minyOld;
  //meshParameter->maxy = maxyOld;

  //-----------------------------------------------------------------
  // Extra work for periodic domains: connect left of domain to the
  // right and/or top of domain to the bottom. 
  //-----------------------------------------------------------------

  // Make periodic in x
  if (meshParameter->periodicFlagX) {
    Array<real> *vxTemp =
      new Array<real>(1, cudaFlag, (unsigned int) nVertex);
    Array<unsigned int> *indexArray =
      new Array<unsigned int>(1, cudaFlag, (unsigned int) nVertex);
    
    // Set equal to first dimension of vertexCoordinates
    vxTemp->SetEqualComb(connectivity->vertexCoordinates, 0, 0);
    indexArray->SetToSeries();
    vxTemp->SortByKey(indexArray);

    // Two leftmost and rightmost vertices
    unsigned int vLeft1, vLeft2, vRight1, vRight2;
    indexArray->GetSingleValue(&vLeft1, 0);
    indexArray->GetSingleValue(&vLeft2, 1);
    indexArray->GetSingleValue(&vRight1, nVertex - 1);
    indexArray->GetSingleValue(&vRight2, nVertex - 2);

    delete vxTemp;
    delete indexArray;

    // y-coordinates of leftmost vertices
    real vy1, vy2;
    real2 Vy;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vLeft1);
    vy1 = Vy.y;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vLeft2);
    vy2 = Vy.y;
    
    if (vy1 > vy2) {
      int temp = vLeft1;
      vLeft1 = vLeft2;
      vLeft2 = temp;
    }
    
    // y-coordinates of rightmost vertices
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vRight1);
    vy1 = Vy.y;
    connectivity->vertexCoordinates->GetSingleValue(&Vy, vRight2);
    vy2 = Vy.y;
    
    if (vy1 > vy2) {
      int temp = vRight1;
      vRight1 = vRight2;
      vRight2 = temp;
    }
    
    // Find edge between two leftmost and to rightmost vertices
    Array<int> *edgeLeftRight = new Array<int>(1, cudaFlag, (unsigned int) 2);
    int *pEdgeLeftRight = edgeLeftRight->GetPointer();

    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;

      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devFindEdgeBetween, 
					 (size_t) 0, 0);
      
      devFindEdgeBetween<<<nBlocks, nThreads>>>
	(nTriangle, vLeft1, vLeft2, pTv, pTe, &(pEdgeLeftRight[0]));
      devFindEdgeBetween<<<nBlocks, nThreads>>>
	(nTriangle, vRight2, vRight1, pTv, pTe, &(pEdgeLeftRight[1]));
    
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int i = 0; i < nTriangle; i++) {
	FindEdgeBetween(i, vLeft1, vLeft2, pTv, pTe, &(pEdgeLeftRight[0]));
	FindEdgeBetween(i, vRight2, vRight1, pTv, pTe, &(pEdgeLeftRight[1]));
      }
    }
    
    if (cudaFlag == 1)
      edgeLeftRight->TransformToHost();
    pEdgeLeftRight = edgeLeftRight->GetPointer();
    
    int eleft = pEdgeLeftRight[0];
    int eright = pEdgeLeftRight[1];

    delete edgeLeftRight;
    
    if (eleft == -1 || eright == -1) {
      std::cout << "Could not find edges to make domain periodic! "
		<< eleft << " " << eright << std::endl;

      std::cout << vLeft1 << " " << vLeft2 << std::endl;
      std::cout << vRight1 << " " << vRight2 << std::endl;
      throw std::runtime_error("");
    }
 
    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 3);

    int3 V;
    V.x = (int) vLeft1;
    V.y = (int) vLeft2;
    V.z = (int) (vRight2 - nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);
    V.x = (int) vRight2;
    V.y = (int) vRight1;
    V.z = (int) (vLeft1 + nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);
    
    V.x = (int) eleft;
    V.y = nEdge + 2;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle);
    V.x = (int) eright;
    V.y = nEdge;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle + 1);
    
    int et1Left, et2Left, et1Right, et2Right;
    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, eleft);
    et1Left = T.x;
    et2Left = T.y;
    if (et1Left == -1) T.x = nTriangle;
    if (et2Left == -1) T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, eleft);
    
    connectivity->edgeTriangles->GetSingleValue(&T, eright);
    et1Right = T.x;
    et2Right = T.y;
    if (et1Right == -1) T.x = nTriangle + 1;
    if (et2Right == -1) T.y = nTriangle + 1;
    connectivity->edgeTriangles->SetSingleValue(T, eright);

    T.x = nTriangle + 1;
    T.y = -1;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);
    T.x = nTriangle + 1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 1);
    T.x = -1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 2);
    
    nTriangle += 2;
    nEdge += 3;
  }
  
  // Make periodic in y
  if (meshParameter->periodicFlagY) {
    Array<real> *vyTemp =
      new Array<real>(1, cudaFlag, (unsigned int) nVertex);
    Array<unsigned int> *indexArray =
      new Array<unsigned int>(1, cudaFlag, (unsigned int) nVertex);
    
    // Set equal to second dimension of vertexCoordinates
    vyTemp->SetEqualComb(connectivity->vertexCoordinates, 0, 1);
    indexArray->SetToSeries();
    vyTemp->SortByKey(indexArray);

    // Two topmost and bottommost vertices
    unsigned int vBottom1, vBottom2, vTop1, vTop2;
    indexArray->GetSingleValue(&vBottom1, 0);
    indexArray->GetSingleValue(&vBottom2, 1);
    indexArray->GetSingleValue(&vTop1, nVertex - 1);
    indexArray->GetSingleValue(&vTop2, nVertex - 2);

    delete vyTemp;
    delete indexArray;

    // x-coordinates of bottommost vertices
    real vx1, vx2;
    real2 Vx;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vBottom1);
    vx1 = Vx.x;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vBottom2);
    vx2 = Vx.x;
    
    if (vx1 > vx2) {
      int temp = vBottom1;
      vBottom1 = vBottom2;
      vBottom2 = temp;
    }
    
    // x-coordinates of topmost vertices
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vTop1);
    vx1 = Vx.x;
    connectivity->vertexCoordinates->GetSingleValue(&Vx, vTop2);
    vx2 = Vx.x;

    if (vx1 > vx2) {
      int temp = vTop1;
      vTop1 = vTop2;
      vTop2 = temp;
    }

    // Find edge between two bottommost and to topmost vertices
    Array<int> *edgeBottomTop = new Array<int>(1, cudaFlag, (unsigned int) 2);
    edgeBottomTop->SetToValue(-1);
    int *pEdgeBottomTop = edgeBottomTop->GetPointer();

    if (cudaFlag == 1) {
      int nBlocks = 128;
      int nThreads = 128;
      
      // Base nThreads and nBlocks on maximum occupancy
      cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
					 devFindEdgeBetween, 
					 (size_t) 0, 0);
      
      devFindEdgeBetween<<<nBlocks, nThreads>>>
	(nTriangle, vBottom2, vBottom1, pTv, pTe, &(pEdgeBottomTop[0]));
      devFindEdgeBetween<<<nBlocks, nThreads>>>
	(nTriangle, vTop1, vTop2, pTv, pTe, &(pEdgeBottomTop[1]));
      
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
    } else {
      for (int i = 0; i < nTriangle; i++) {
	FindEdgeBetween(i, vBottom2, vBottom1, pTv, pTe, &(pEdgeBottomTop[0]));
	FindEdgeBetween(i, vTop1, vTop2, pTv, pTe, &(pEdgeBottomTop[1]));
      }
    }
    
    if (cudaFlag == 1)
      edgeBottomTop->TransformToHost();
    pEdgeBottomTop = edgeBottomTop->GetPointer();
    
    int eBottom = pEdgeBottomTop[0];
    int eTop = pEdgeBottomTop[1];

    delete edgeBottomTop;
    
    if (eBottom == -1 || eTop == -1) {
      std::cout << "Could not find y edges to make domain periodic! "
		<< eBottom << " " << eTop << std::endl;

      std::cout << vBottom1 << " " << vBottom2 << std::endl;
      std::cout << vTop1 << " " << vTop2 << std::endl;
      throw std::runtime_error("");
    }
 
    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 3);

    int3 V;
    V.x = (int) vBottom2;
    V.y = (int) vBottom1;
    V.z = (int) (vTop2 - 3*nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);

    V.x = (int) vTop1;
    V.y = (int) vTop2;
    V.z = (int) (vBottom1 + 3*nVertex);
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);
   
    V.x = (int) eBottom;
    V.y = nEdge + 2;
    V.z = nEdge + 1;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle);
    
    V.x = (int) eTop;
    V.y = nEdge + 2;
    V.z = nEdge;
    connectivity->triangleEdges->SetSingleValue(V, nTriangle + 1);
    
    int et1Bottom, et2Bottom, et1Top, et2Top;
    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, eBottom);
    et1Bottom = T.x;
    et2Bottom = T.y;
    if (et1Bottom == -1) T.x = nTriangle;
    if (et2Bottom == -1) T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, eBottom);
    
    connectivity->edgeTriangles->GetSingleValue(&T, eTop);
    et1Top = T.x;
    et2Top = T.y;
    
    if (et1Top == -1) T.x = nTriangle + 1;
    if (et2Top == -1) T.y = nTriangle + 1;
    connectivity->edgeTriangles->SetSingleValue(T, eTop);

    T.x = nTriangle + 1;
    T.y = -1;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);
    T.x = -1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 1);
    T.x = nTriangle + 1;
    T.y = nTriangle;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge + 2);
    
    nTriangle += 2;
    nEdge += 3;
  }  

  if (meshParameter->periodicFlagX && meshParameter->periodicFlagY) {
    connectivity->triangleVertices->SetSize(nTriangle + 2);
    connectivity->triangleEdges->SetSize(nTriangle + 2);
    connectivity->edgeTriangles->SetSize(nEdge + 1);
    
    int e1, v1, v2;
    int3 V;
    int3 E;
    connectivity->triangleEdges->GetSingleValue(&E, 5);
    e1 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, 5);
    v1 = V.x;
    v2 = V.y;
    
    int2 T;
    connectivity->edgeTriangles->GetSingleValue(&T, e1);
    int et1e1 = T.x;
    int et2e1 = T.y;
    
    if (et1e1 != -1 && et2e1 != -1) {
      e1 = E.y;
      v1 = V.y;
      v2 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e1);
    et1e1 = T.x;
    et2e1 = T.y;
    if (et1e1 != -1 && et2e1 != -1) {
      e1 = E.z;
      v1 = V.z;
      v2 = V.x;
    }

    int e2, v4;
    connectivity->triangleEdges->GetSingleValue(&E, 4);
    e2 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, 4);
    v4 = V.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e2);
    int et1e2 = T.x;
    int et2e2 = T.y;
    if (et1e2 != -1 && et2e2 != -1) {
      e2 = E.y;
      v4 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e2);
    et1e2 = T.x;
    et2e2 = T.y;
    if (et1e2 != -1 && et2e2 != -1) {
      e2 = E.z;
      v4 = V.x;
    }

    int e3, v5;
    connectivity->triangleEdges->GetSingleValue(&E, 2);
    e3 = E.x;
    connectivity->triangleVertices->GetSingleValue(&V, 2);
    v5 = V.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e3);
    int et1e3 = T.x;
    int et2e3 = T.y;
    
    if (et1e3 != -1 && et2e3 != -1) {
      e3 = E.y;
      v5 = V.z;
    }
    connectivity->edgeTriangles->GetSingleValue(&T, e3);
    et1e3 = T.x;
    et2e3 = T.y;
    if (et1e3 != -1 && et2e3 != -1) {
      e3 = E.z;
      v5 = V.x;
    }

    int e4;
    connectivity->triangleEdges->GetSingleValue(&E, 3);
    e4 = E.x;
    connectivity->edgeTriangles->GetSingleValue(&T, e4);
    int et1e4 = T.x;
    int et2e4 = T.y;
    if (et1e4 != -1 && et2e4 != -1) e4 = E.y;
    connectivity->edgeTriangles->GetSingleValue(&T, e4);
    et1e4 = T.x;
    et2e4 = T.y;
    if (et1e4 != -1 && et2e4 != -1) e4 = E.z;

    V.x = v2;
    V.y = v1;
    V.z = v4 + 2*nVertex;
    connectivity->triangleVertices->SetSingleValue(V, nTriangle);    

    E.x = e1;
    E.y = e4;
    E.z = nEdge;
    connectivity->triangleEdges->SetSingleValue(E, nTriangle);

    V.x = v5;
    V.y = v2;
    V.z = v4 + 2*nVertex;
    connectivity->triangleVertices->SetSingleValue(V, nTriangle + 1);

    E.x = e3;
    E.y = nEdge;
    E.z = e2;
    connectivity->triangleEdges->SetSingleValue(E, nTriangle + 1);

    T.x = 6;
    T.y = 7;
    connectivity->edgeTriangles->SetSingleValue(T, nEdge);

    T.x = 4;
    T.y = 7;
    connectivity->edgeTriangles->SetSingleValue(T, e2);

    T.x = 5;
    T.y = 6;
    connectivity->edgeTriangles->SetSingleValue(T, e1);

    T.x = 2;
    T.y = 7;
    connectivity->edgeTriangles->SetSingleValue(T, e3);

    T.x = 3;
    T.y = 6;
    connectivity->edgeTriangles->SetSingleValue(T, e4);

    nTriangle += 2;
    nEdge += 1;

  }
  
  delete vertexRemoveFlag;
  delete vertexFlagScan;
  delete triangleRemoveFlag;
  delete triangleFlagScan;
  delete edgeRemoveFlag;
  delete edgeFlagScan;
  
  delete vertexOrder;

  if (verboseLevel > 0)
    std::cout << "Boundaries done" << std::endl;

  // Add extra vertices to break symmetry
  Array<real2> *vertexExtra = new Array<real2>(1, cudaFlag, 2);
  Array<int> *vertexExtraOrder = new Array<int>(1, cudaFlag, 2);

  real2 temp;
  temp.x = 1.0/sqrt(2.0);
  temp.y = meshParameter->miny;
  vertexExtra->SetSingleValue(temp, 0);

  temp.x = 1.0/M_PI;
  temp.y = meshParameter->maxy; 
  vertexExtra->SetSingleValue(temp, 1);
    
  nAdded = refine->AddVertices(connectivity,
			       meshParameter,
			       predicates,
			       delaunay,
			       vertexExtra,
			       vertexExtraOrder);

  std::cout << "Added " << nAdded << " extra vertices" << std::endl;

  delete vertexExtra;
  delete vertexExtraOrder;
}

}
