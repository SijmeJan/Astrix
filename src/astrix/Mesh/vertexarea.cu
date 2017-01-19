// -*-c++-*-
/*! \file vertexarea.cu
\brief Functions for calculating vertex areas*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./mesh.h"
#include "./triangleLow.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"
#include "./Connectivity/connectivity.h"
#include "./Param/meshparameter.h"

namespace astrix {

//##############################################################################
/*! \brief Add contribution of area of triangle \a n to the area of its vertices

  Every triangle contributes one third of its area to the area of the Voronoi cell associated with its vertices. Atomically add this contribution to \a vertexArea

\param n Index of triangle to consider
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *pVertexArea Pointer to output array containing vertex areas
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *triL Pointer to array containing triangle edge lengths
\param nTriangle Total number of triangles*/
//##############################################################################

__host__ __device__
void VertexAreaSingle(int n, int3 *pTv,
                      real *pVertexArea, int nVertex,
                      real2 *pVc, real Px, real Py, int nTriangle)
{
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  real Ax, Bx, Cx, Ay, By, Cy;
  GetTriangleCoordinates(pVc, a, b, c,
                         nVertex, Px, Py,
                         Ax, Bx, Cx, Ay, By, Cy);

  real A = half*((Ax - Cx)*(By - Cy) - (Ay - Cy)*(Bx - Cx))*onethird;

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  AtomicAdd(&pVertexArea[a], A);
  AtomicAdd(&pVertexArea[b], A);
  AtomicAdd(&pVertexArea[c], A);
}

//######################################################################
/*! \brief Kernel calculating vertex areas (Voronoi cells)

  Every triangle contributes one third of its area to the area of the Voronoi cell associated with its vertices. Atomically add this contribution to \a vertexArea

\param nVertex Total number of vertices in Mesh
\param nTriangle Total number of triangles
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *pVertexArea Pointer to output array containing vertex areas
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *triL Pointer to array containing triangle edge lengths*/
//######################################################################

__global__ void
devCalcVertexArea(int nVertex, int nTriangle,
                  int3 *pTv, real *pVertexArea,
                  real2 *pVc, real Px, real Py)
{
  // n = triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    VertexAreaSingle(n, pTv, pVertexArea,
                     nVertex, pVc, Px, Py, nTriangle);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Every triangle contributes one third of its area to the area of the Voronoi cell associated with its vertices. Atomically add this contribution to \a vertexArea*/
//#########################################################################

void Mesh::CalcVertexArea()
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();

  // Vertex area (= volume Voronoi cell)
  vertexArea->SetSize(nVertex);
  vertexArea->SetToValue(0.0);
  real *pVertexArea = vertexArea->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcVertexArea,
                                       (size_t) 0, 0);

    devCalcVertexArea<<<nBlocks, nThreads>>>
      (nVertex, nTriangle, pTv, pVertexArea, pVc, Px, Py);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      VertexAreaSingle(n, pTv, pVertexArea, nVertex, pVc, Px, Py, nTriangle);
  }
}

}
