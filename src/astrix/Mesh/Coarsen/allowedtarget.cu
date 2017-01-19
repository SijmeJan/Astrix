// -*-c++-*-
/*! \file allowedtarget.cu
\brief Functions for determining allowed target triangles
*/
#include <cmath>
#include <iostream>
#include <iomanip>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "./coarsen.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Predicates/predicates.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! \brief Select target triangle containing vertex to move \a vRemove on when removing

  Vertices are removed by moving them on top of a neighbouring vertex and subsequently adjusting the connections. However, not all vertices are suited to move another vertex on top of; it may lead to illegal triangles. This function will select a suitable target triangle.

\param vRemove Vertex to be removed
\param *vTri Pointer to list of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *tAllowed Pointer to output vector. For every triangle sharing vertex \a vRemove, output will either be 1 (allowed as target) or 0 (not allowed as target)*/
//#########################################################################

__host__ __device__
void FindAllowedTargetTriangleSingle(int vRemove, int *vTri, int maxTriPerVert,
                                     int3 *pTv, int3 *pTe, int2 *pEt,
                                     int nVertex, real2 *pVc,
                                     real Px, real Py,
                                     Predicates *pred, real *pParam,
                                     int *tAllowed)
{
  const real zero  = (real) 0.0;

  // Reject triangles that would generate illegal triangles
  for (int i = 0; i < maxTriPerVert; i++) {
    // Assume this triangle is suitable
    int ret = 1;
    int t = vTri[i];

    if (t != -1) {
      int segmentFlag = 0;

      int a = pTv[t].x;
      int b = pTv[t].y;
      int c = pTv[t].z;

      real Ax, Bx, Cx, Ay, By, Cy;
      GetTriangleCoordinates(pVc, a, b, c,
                             nVertex, Px, Py,
                             Ax, Bx, Cx, Ay, By, Cy);


      // Translate triangle
      // pERIODIC
      TranslateTriangleToVertex(vRemove, Px, Py, nVertex,
                                a, b, c, Ax, Ay, Bx, By, Cx, Cy);

      while (a >= nVertex) a -= nVertex;
      while (b >= nVertex) b -= nVertex;
      while (c >= nVertex) c -= nVertex;
      while (a < 0) a += nVertex;
      while (b < 0) b += nVertex;
      while (c < 0) c += nVertex;

      real x = zero, y = zero;

      int e1 = pTe[t].x;
      int e2 = pTe[t].y;
      int e3 = pTe[t].z;

      int t11 = pEt[e1].x;
      int t21 = pEt[e1].y;
      int t12 = pEt[e2].x;
      int t22 = pEt[e2].y;
      int t13 = pEt[e3].x;
      int t23 = pEt[e3].y;

      if (a == vRemove) {
        x = Bx;
        y = By;
        if (t11 == -1 || t21 == -1) segmentFlag = 1;
        if (t13 == -1 || t23 == -1) {
          segmentFlag = 1;
          x = Cx;
          y = Cy;
        }
      }
      if (b == vRemove) {
        x = Cx;
        y = Cy;
        if (t12 == -1 || t22 == -1) segmentFlag = 1;
        if (t11 == -1 || t21 == -1) {
          segmentFlag = 1;
          x = Ax;
          y = Ay;
        }
      }
      if (c == vRemove) {
        x = Ax;
        y = Ay;
        if (t13 == -1 || t23 == -1) segmentFlag = 1;
        if (t12 == -1 || t22 == -1) {
          segmentFlag = 1;
          x = Bx;
          y = By;
        }
      }

      // Count how many illegal triangles are created by moving vertex
      int nBad = 0;
      for (int j = 0; j < maxTriPerVert; j++) {
        int t2 = vTri[j];
        if (t2 != -1) {
          int a = pTv[t2].x;
          int b = pTv[t2].y;
          int c = pTv[t2].z;

          real ax, bx, cx, ay, by, cy;
          GetTriangleCoordinates(pVc, a, b, c,
                                 nVertex, Px, Py,
                                 ax, bx, cx, ay, by, cy);

          // Translate triangle
          // pERIODIC
          TranslateTriangleToVertex(vRemove, Px, Py, nVertex,
                                    a, b, c, ax, ay, bx, by, cx, cy);

          while (a >= nVertex) a -= nVertex;
          while (b >= nVertex) b -= nVertex;
          while (c >= nVertex) c -= nVertex;
          while (a < 0) a += nVertex;
          while (b < 0) b += nVertex;
          while (c < 0) c += nVertex;

          // Replace vRemove coord's with those of vTarget
          if (a == vRemove) {
            ax = x;
            ay = y;
          }
          if (b == vRemove) {
            bx = x;
            by = y;
          }
          if (c == vRemove) {
            cx = x;
            cy = y;
          }

          real det = pred->orient2d(ax, ay, bx, by, cx, cy, pParam);
          if (det <= zero) nBad++;
        }
      }
      // Too many bad triangles: reject triangle and vTarget
      if (nBad > 2 - segmentFlag) ret = 0;
    }

    tAllowed[i] = ret;
  }
}

//#########################################################################
/*! \brief Kernel selecting target triangle containing vertex when removing

  Vertices are removed by moving them on top of a neighbouring vertex and subsequently adjusting the connections. However, not all vertices are suited to move another vertex on top of; it may lead to illegal triangles. This kernel function will select suitable target triangles.

\param *pVertexRemove Pointer to list of vertices to be removed
\param nRemove Number of vertices to be removed
\param *pVertexTriangleList Pointer to list of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *te1 Pointer to first edge of triangle
\param *te2 Pointer to second edge of triangle
\param *te3 Pointer to third edge of triangle
\param *et1 Pointer to first triangle neighbouring edge
\param *et2 Pointer to second triangle neighbouring edge
\param nVertex Total number of vertices in Mesh
\param *pVertX Pointer to x-coordinates of vertices
\param *pVertY Pointer to y-coordinates of vertices
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pred Pointer to initialised Predicates objest
\param pParam Pointer to initialised Predicates parameter vector
\param *pAllowed Pointer to output vector. For every triangle sharing vertex \a pVertexRemove[i], output will either be 1 (allowed as target) or 0 (not allowed as target)*/
//#########################################################################

__global__
void devFindAllowedTargetTriangle(int *pVertexRemove, int nRemove,
                                  int *pVertexTriangleList,
                                  int maxTriPerVert,
                                  int3 *pTv, int3 *pTe, int2 *pEt,
                                  int nVertex, real2 *pVc,
                                  real Px, real Py,
                                  Predicates *pred, real *pParam,
                                  int *pAllowed)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nRemove) {
    FindAllowedTargetTriangleSingle(pVertexRemove[n],
                                    &(pVertexTriangleList[n*maxTriPerVert]),
                                    maxTriPerVert,
                                    pTv, pTe, pEt,
                                    nVertex, pVc, Px, Py,
                                    pred, pParam,
                                    &(pAllowed[n*maxTriPerVert]));

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Vertices are removed by moving them on top of a neighbouring vertex and subsequently adjusting the connections. However, not all vertices are suited to move another vertex on top of; it may lead to illegal triangles. This function will select suitable target triangles for all vertices to be removed.

\param *vertexTriangleAllowed Pointer to output array flagging whether the corresponding triangle in \a vertexTriangleList is suited as target triangle
\param *vertexTriangleList Pointer to Array of triangles sharing vertex
\param maxTriPerVert Maximum number of triangles sharing single vertex*/
//#########################################################################

void Coarsen::FindAllowedTargetTriangles(Connectivity *connectivity,
                                         Predicates *predicates,
                                         Array<int> *vertexTriangleAllowed,
                                         Array<int> *vertexTriangleList,
                                         int maxTriPerVert,
                                         const MeshParameter *mp)
{
  int transformFlag = 0;

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();
      vertexTriangleAllowed->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();
      vertexTriangleAllowed->TransformToDevice();

      cudaFlag = 1;
    }
  }

  int *pAllowed = vertexTriangleAllowed->GetPointer();

  int *pVertexTriangleList = vertexTriangleList->GetPointer();
  int *pVertexRemove = vertexRemove->GetPointer();
  int nRemove = vertexRemove->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  real Px = mp->maxx - mp->minx;
  real Py = mp->maxy - mp->miny;

  // Find allowed target triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads,
       devFindAllowedTargetTriangle,
       (size_t) 0, 0);

    devFindAllowedTargetTriangle<<<nBlocks, nThreads>>>
      (pVertexRemove, nRemove, pVertexTriangleList,
       maxTriPerVert, pTv, pTe, pEt,
       nVertex, pVc, Px, Py,
       predicates, pParam, pAllowed);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRemove; n++)
      FindAllowedTargetTriangleSingle(pVertexRemove[n],
                                      &(pVertexTriangleList[n*maxTriPerVert]),
                                      maxTriPerVert,
                                      pTv, pTe, pEt,
                                      nVertex, pVc,
                                      Px, Py,
                                      predicates, pParam,
                                      &(pAllowed[n*maxTriPerVert]));
  }

  if (transformFlag == 1) {
    connectivity->Transform();
    if (cudaFlag == 1) {
      vertexTriangleList->TransformToHost();
      vertexRemove->TransformToHost();
      vertexTriangleAllowed->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexTriangleList->TransformToDevice();
      vertexRemove->TransformToDevice();
      vertexTriangleAllowed->TransformToDevice();

      cudaFlag = 1;
    }
  }

}

}
