// -*-c++-*-
/*! \file insert.cu
\brief Functions to insert new vertices in Mesh

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../Predicates/predicates.h"
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/nvtxEvent.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/profile.h"

namespace astrix {

//#########################################################################
/*! \brief Insert vertex \a n into Mesh

\param n Index of vertex to insert
\param t Triangle to insert vertex into (-1 if inserting on edge)
\param e Edge onto which to insert vertex (-1 if inserting in triangle)
\param nVertex Total number of vertices in Mesh
\param nEdge Total number of edges in Mesh
\param nTriangle Total number of triangles in Mesh
\param x x-coordinate of vertex to insert
\param y y-coordinate of vertex to insert
\param indexInEdgeArray Index to start adding edges in \a edgeTriangles
\param indexInTriangleArray Index to start adding triangles in \a triangleVertices and \a triangleEdges
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param nvAdd Total number of vertices to be added
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pWantRefine Pointer to array of flags whether to refine triangles. Value of the initial triangle is set to that of the new triangles (either 1 if creating a new Mesh or 0 if refining during a simulation) */
//#########################################################################

__host__ __device__
void InsertVertex(int n, int t, int e, int nVertex, int nEdge,
                  int nTriangle, real x, real y,
                  int indexInEdgeArray, int indexInTriangleArray,
                  real2 *pVc, int3 *pTv, int3 *pTe, int2 *pEt,
                  real minx, real maxx, real miny, real maxy,
                  int nvAdd, int periodicFlagX, int periodicFlagY,
                  const Predicates *pred, real *pParam,
                  int *pWantRefine)
{
  // Index of new vertex in vertex array
  int i = n + nVertex;

  // i + translateVertex = valid triangle vertex
  int translateVertex = 0;

  // If vertex coordinates outside domain, insert periodic copy
  if (periodicFlagX) {
    if (x <= minx) translateVertex -= (nVertex + nvAdd);
    if (x > maxx) translateVertex += (nVertex + nvAdd);
  }
  if (periodicFlagY) {
    if (y <= miny) translateVertex -= 3*(nVertex + nvAdd);
    if (y > maxy) translateVertex += 3*(nVertex + nvAdd);
  }

  //#########################################
  // Case 1: Insert vertex inside triangle t
  //#########################################

  if (t != -1) {
    // Copy flag whether to refine
    if (pWantRefine != 0)
      pWantRefine[t] = pWantRefine[indexInTriangleArray];

    int a = pTv[t].x;
    int b = pTv[t].y;
    int c = pTv[t].z;

    // Create 2 new triangles
    pTv[indexInTriangleArray].x = c;
    pTv[indexInTriangleArray].y = a;
    pTv[indexInTriangleArray].z = i + translateVertex;
    pTv[indexInTriangleArray + 1].x = b;
    pTv[indexInTriangleArray + 1].y = c;
    pTv[indexInTriangleArray + 1].z = i + translateVertex;
    // Replace triangle t
    pTv[t].z = i + translateVertex;

    // Make valid triangles in case of periodic domain
    MakeValidIndices(pTv[indexInTriangleArray].x,
                     pTv[indexInTriangleArray].y,
                     pTv[indexInTriangleArray].z,
                     nVertex + nvAdd);
    MakeValidIndices(pTv[indexInTriangleArray + 1].x,
                     pTv[indexInTriangleArray + 1].y,
                     pTv[indexInTriangleArray + 1].z,
                     nVertex + nvAdd);
    MakeValidIndices(pTv[t].x, pTv[t].y, pTv[t].z, nVertex + nvAdd);

    // Give new edges triangles
    pEt[indexInEdgeArray].x = indexInTriangleArray;
    pEt[indexInEdgeArray].y = t;
    pEt[indexInEdgeArray + 1].x = t;
    pEt[indexInEdgeArray + 1].y = indexInTriangleArray + 1;
    pEt[indexInEdgeArray + 2].x = indexInTriangleArray + 1;
    pEt[indexInEdgeArray + 2].y = indexInTriangleArray;

    int e2 = pTe[t].y;
    int e3 = pTe[t].z;

    // Update triangle info for old edges
    if (pEt[e2].x == t) pEt[e2].x = indexInTriangleArray + 1;
    if (pEt[e2].y == t) pEt[e2].y = indexInTriangleArray + 1;
    if (pEt[e3].x == t) pEt[e3].x = indexInTriangleArray;
    if (pEt[e3].y == t) pEt[e3].y = indexInTriangleArray;

    // Give new triangles edges
    pTe[indexInTriangleArray].x = e3;
    pTe[indexInTriangleArray].y = indexInEdgeArray;
    pTe[indexInTriangleArray].z = indexInEdgeArray + 2;
    pTe[indexInTriangleArray + 1].x = e2;
    pTe[indexInTriangleArray + 1].y = indexInEdgeArray + 2;
    pTe[indexInTriangleArray + 1].z = indexInEdgeArray + 1;
    pTe[t].y = indexInEdgeArray + 1;
    pTe[t].z = indexInEdgeArray;
  } else {
    //#################################
    // Case 2: Insert vertex on edge e
    //#################################

    // Neighbouring triangles
    int t1 = pEt[e].x;
    int t2 = pEt[e].y;

    // Check if edge is segment
    if (t1 != -1 && t2 != -1) {
      //##############################################
      // Case 2a: Insert vertex on non-segment edge e
      //##############################################

      // Copy flag whether to refine
      if (pWantRefine != 0) {
        pWantRefine[t1] = pWantRefine[indexInTriangleArray];
        pWantRefine[t2] = pWantRefine[indexInTriangleArray];
      }

      int tv11 = pTv[t1].x;
      int tv21 = pTv[t1].y;
      int tv31 = pTv[t1].z;
      int tv12 = pTv[t2].x;
      int tv22 = pTv[t2].y;
      int tv32 = pTv[t2].z;

      int te11 = pTe[t1].x;
      int te21 = pTe[t1].y;
      int te31 = pTe[t1].z;
      int te12 = pTe[t2].x;
      int te22 = pTe[t2].y;
      int te32 = pTe[t2].z;

      // Make sure edge e is between B and C in t1
      int B = tv11;
      int C = tv21;
      if (e == te21) {
        B = tv21;
        C = tv31;
      }
      if (e == te31) {
        B = tv31;
        C = tv11;
      }

      // Make sure edge e is between E and F in t2
      int E = tv12;
      int F = tv22;
      if (e == te22) {
        E = tv22;
        F = tv32;
      }
      if (e == te32) {
        E = tv32;
        F = tv12;
      }

      // Check if one triangle needs translating
      if (B != F || C != E) {
        // Coordinates of t1
        real ax, bx, cx, ay, by, cy;
        GetTriangleCoordinates(pVc, tv11, tv21, tv31,
                               nVertex + nvAdd, maxx - minx, maxy - miny,
                               ax, bx, cx, ay, by, cy);

        // A1: vertex *exactly* on (extended) edge of t1?
        real A1 = 0.0;
        if (e == te11) A1 = pred->orient2d(ax, ay, bx, by, x, y, pParam);
        if (e == te21) A1 = pred->orient2d(bx, by, cx, cy, x, y, pParam);
        if (e == te31) A1 = pred->orient2d(cx, cy, ax, ay, x, y, pParam);

        // Coordinates of t2
        GetTriangleCoordinates(pVc, tv12, tv22, tv32,
                               nVertex + nvAdd, maxx - minx, maxy - miny,
                               ax, bx, cx, ay, by, cy);

        // A2: vertex *exactly* on (extended) edge of t2?
        real A2 = 0.0;
        if (e == te12) A2 = pred->orient2d(ax, ay, bx, by, x, y, pParam);
        if (e == te22) A2 = pred->orient2d(bx, by, cx, cy, x, y, pParam);
        if (e == te32) A2 = pred->orient2d(cx, cy, ax, ay, x, y, pParam);

        // Vertex *exactly* on (extended) edges of both t1 and t2
        if (A1 == 0.0 && A2 == 0.0) {
          // Check whether vertex is in between vertices of t2
          real xmax = max(ax, bx);
          real xmin = min(ax, bx);
          real ymax = max(ay, by);
          real ymin = min(ay, by);

          if (e == te22) {
            xmax = max(bx, cx);
            xmin = min(bx, cx);
            ymax = max(by, cy);
            ymin = min(by, cy);
          }

          if (e == te32) {
            xmax = max(cx, ax);
            xmin = min(cx, ax);
            ymax = max(cy, ay);
            ymin = min(cy, ay);
          }

          // Assume vertex is on t1 edge: no translation of t1
          int translateT1Flag = 0;//(translateVertex != 0);

          // If vertex is on t2 edge: translate t1
          if (((x < xmax && x > xmin) || xmax == xmin) &&
              ((y < ymax && y > ymin) || ymax == ymin))
            translateT1Flag = 1;//(translateVertex == 0);

          if (translateT1Flag == 1) {
            // Translate t1
            tv11 -= (B - F);
            tv21 -= (B - F);
            tv31 -= (B - F);
          } else {
            // Translate t2
            tv12 += (B - F);
            tv22 += (B - F);
            tv32 += (B - F);
          }
        } else {
          // Choose to translate triangle with biggest A
          if (fabs(A2) > fabs(A1)) {
            // Translate t2
            tv12 += (B - F);
            tv22 += (B - F);
            tv32 += (B - F);
          } else {
            // Translate t1
            tv11 -= (B - F);
            tv21 -= (B - F);
            tv31 -= (B - F);
          }
        }

        // Something is wrong is this is not true
        assert(B - F == C - E);
      }

      // Split triangle t1
      int3 tvs;
      tvs.x = tv21;
      tvs.y = tv31;
      tvs.z = tv11;
      int2 es;
      es.x = te21;
      es.y = te31;

      if (e == te21) {
        tvs.x = tv31;
        tvs.y = tv11;
        tvs.z = tv21;

        es.x = te31;
        es.y = te11;
      }

      if (e == te31) {
        tvs.x = tv11;
        tvs.y = tv21;
        tvs.z = tv31;

        es.x = te11;
        es.y = te21;
      }

      pTv[indexInTriangleArray].x = i + translateVertex;
      pTv[indexInTriangleArray].y = tvs.x;
      pTv[indexInTriangleArray].z = tvs.y;

      pTv[t1].x = i + translateVertex;
      pTv[t1].y = tvs.y;
      pTv[t1].z = tvs.z;

      if (pEt[es.x].x == t1) pEt[es.x].x = indexInTriangleArray;
      if (pEt[es.x].y == t1) pEt[es.x].y = indexInTriangleArray;

      pTe[indexInTriangleArray].x = indexInEdgeArray;
      pTe[indexInTriangleArray].y = es.x;
      pTe[indexInTriangleArray].z = indexInEdgeArray + 1;
      pTe[t1].x = indexInEdgeArray + 1;
      pTe[t1].y = es.y;
      pTe[t1].z = e;

      // Split triangle t2
      tvs.x = tv32;
      tvs.y = tv12;
      tvs.z = tv22;
      es.x = te32;
      es.y = te22;

      if (e == te22) {
        tvs.x = tv12;
        tvs.y = tv22;
        tvs.z = tv32;

        es.x = te12;
        es.y = te32;
      }

      if (e == te32) {
        tvs.x = tv22;
        tvs.y = tv32;
        tvs.z = tv12;

        es.x = te22;
        es.y = te12;
      }

      pTv[indexInTriangleArray + 1].x = i + translateVertex;
      pTv[indexInTriangleArray + 1].y = tvs.x;
      pTv[indexInTriangleArray + 1].z = tvs.y;

      pTv[t2].x = i + translateVertex;
      pTv[t2].y = tvs.z;
      pTv[t2].z = tvs.x;

      if (pEt[es.x].x == t2) pEt[es.x].x = indexInTriangleArray + 1;
      if (pEt[es.x].y == t2) pEt[es.x].y = indexInTriangleArray + 1;

      pTe[indexInTriangleArray + 1].x = indexInEdgeArray + 2;
      pTe[indexInTriangleArray + 1].y = es.x;
      pTe[indexInTriangleArray + 1].z = indexInEdgeArray;
      pTe[t2].x = e;
      pTe[t2].y = es.y;
      pTe[t2].z = indexInEdgeArray + 2;

      // Give new edges neighbouring triangles
      pEt[indexInEdgeArray].x = indexInTriangleArray;
      pEt[indexInEdgeArray].y = indexInTriangleArray + 1;
      pEt[indexInEdgeArray + 1].x = t1;
      pEt[indexInEdgeArray + 1].y = indexInTriangleArray;
      pEt[indexInEdgeArray + 2].x = indexInTriangleArray + 1;
      pEt[indexInEdgeArray + 2].y = t2;

      // Translate triangles so that indices are valid
      MakeValidIndices(pTv[indexInTriangleArray].x,
                       pTv[indexInTriangleArray].y,
                       pTv[indexInTriangleArray].z,
                       nVertex + nvAdd);
      MakeValidIndices(pTv[indexInTriangleArray + 1].x,
                       pTv[indexInTriangleArray + 1].y,
                       pTv[indexInTriangleArray + 1].z,
                       nVertex + nvAdd);
      MakeValidIndices(pTv[t1].x, pTv[t1].y, pTv[t1].z, nVertex + nvAdd);
      MakeValidIndices(pTv[t2].x, pTv[t2].y, pTv[t2].z, nVertex + nvAdd);

    } else {
      //#####################################
      // Case 2b: Insert vertex on segment e
      //#####################################

      // Single neighbouring triangle t
      int t = t1;
      if (t == -1) t = t2;

      if (pWantRefine != 0)
        pWantRefine[t] = pWantRefine[indexInTriangleArray];

      // Find vertex of neighbouring triangle not belonging to edge
      int v1    = pTv[t].x;
      int vnext = pTv[t].y;
      int vprev = pTv[t].z;

      int E1 = pTe[t].x;
      int E2 = pTe[t].y;
      int E3 = pTe[t].z;

      int e1 = E3;
      int e2 = E1;
      if (e == E1) {
        v1    = pTv[t].z;
        vprev = pTv[t].y;
        vnext = pTv[t].x;
        e1 = E2;
        e2 = E3;
      }
      if (e == E3) {
        v1    = pTv[t].y;
        vprev = pTv[t].x;
        vnext = pTv[t].z;
        e1 = E1;
        e2 = E2;
      }

      // PERIODIC
      // Create triangles
      pTv[t].x = i + translateVertex;
      pTv[t].y = vprev;
      pTv[t].z = v1;
      pTv[indexInTriangleArray].x = i + translateVertex;
      pTv[indexInTriangleArray].y = v1;
      pTv[indexInTriangleArray].z = vnext;

      // PERIODIC
      MakeValidIndices(pTv[indexInTriangleArray].x,
                       pTv[indexInTriangleArray].y,
                       pTv[indexInTriangleArray].z,
                       nVertex + nvAdd);
      MakeValidIndices(pTv[t].x, pTv[t].y, pTv[t].z, nVertex + nvAdd);

      // Give new edges triangles
      pEt[indexInEdgeArray].x = indexInTriangleArray;
      pEt[indexInEdgeArray].y = -1;
      pEt[indexInEdgeArray + 1].x = t;
      pEt[indexInEdgeArray + 1].y = indexInTriangleArray;

      if (pEt[e2].x == t) pEt[e2].x = indexInTriangleArray;
      if (pEt[e2].y == t) pEt[e2].y = indexInTriangleArray;

      // Give new triangles edges
      pTe[indexInTriangleArray].x = indexInEdgeArray + 1;
      pTe[indexInTriangleArray].y = e2;
      pTe[indexInTriangleArray].z = indexInEdgeArray;
      pTe[t].x = e;
      pTe[t].y = e1;
      pTe[t].z = indexInEdgeArray + 1;
    }
  }

  // Translate vertex if necessary
  if (periodicFlagX) {
    if (x <= minx) x += (maxx - minx);
    if (x > maxx) x -= (maxx - minx);
  }
  if (periodicFlagY) {
    if (y <= miny) y += (maxy - miny);
    if (y > maxy) y -= (maxy - miny);
  }

  // Write to vertex coordinates array
  pVc[i].x = x;
  pVc[i].y = y;

}

//######################################################################
/*! \brief Insert vertex \a n into Mesh

\param nRefine Total number of vertices to insert
\param *pElementAdd Pointer to array of triangles and edges to insert vertex into
\param nVertex Total number of vertices in Mesh
\param nEdge Total number of edges in Mesh
\param nTriangle Total number of triangles in Mesh
\param *pVcAdd Pointer to vertex coordinates to add
\param *pOnSegmentFlagScan Pointer to array of scanned flags of whether to insert on segment. We need to know this because the number of newly created triangles depends on how many points we insert on segments.
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param minx Left x boundary
\param maxx Right x boundary
\param miny Left y boundary
\param maxy Right y boundary
\param nvAdd Total number of vertices to be added
\param periodicFlagX Flag whether domain is periodic in x
\param periodicFlagY Flag whether domain is periodic in y
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param *pWantRefine Pointer to array of flags whether to refine triangles. Value of the initial triangle is set to that of the new triangles (either 1 if creating a new Mesh or 0 is refining during a simulation) */
//######################################################################

__global__ void
devInsertVertices(int nRefine, int *pElementAdd,
                  int nVertex, int nEdge, int nTriangle, real2 *pVcAdd,
                  unsigned int *pOnSegmentFlagScan,
                  real2 *pVc, int3 *pTv, int3 *pTe, int2 *pEt,
                  real minx, real maxx, real miny, real maxy, int nvAdd,
                  int periodicFlagX, int periodicFlagY,
                  const Predicates *pred, real *pParam,
                  int *pWantRefine)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    int t = pElementAdd[i];
    int e = -1;
    if (t >= nTriangle) {
      e = t - nTriangle;
      t = -1;
    }

    InsertVertex(i, t, e,
                 nVertex, nEdge, nTriangle,
                 pVcAdd[i].x, pVcAdd[i].y,
                 3*i - pOnSegmentFlagScan[i] + nEdge,
                 2*i - pOnSegmentFlagScan[i] + nTriangle,
                 pVc, pTv, pTe, pEt,
                 minx, maxx, miny, maxy,
                 nvAdd, periodicFlagX, periodicFlagY,
                 pred, pParam, pWantRefine);

    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Insert vertices with coordinates as specified in \a vertexCoordinatesAdd into triangles specified in \a triangleAdd or onto edges specified in \a edgeAdd.

  \param *vertexState Pointer to Array containing state vector. We use this to distinguish whether we are creating a new Mesh (\a vertexState=0) meaning all triangles can be refined always, or whether we are in the middle of a simulation in which case no newly created triangles can be refined in the next step before we have computed the new truncation error.*/
//######################################################################

template<class realNeq>
void Refine::InsertVertices(Connectivity * const connectivity,
                            const MeshParameter *meshParameter,
                            const Predicates *predicates,
                            Array<realNeq> * const vertexState,
                            Array<int> * const triangleWantRefine)
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  nvtxEvent *nvtxInsert = new nvtxEvent("Insert", 4);
  nvtxEvent *temp = new nvtxEvent("Segment", 0);

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nEdge = connectivity->edgeTriangles->GetSize();

  int *pElementAdd = elementAdd->GetPointer();
  int nRefine = elementAdd->GetSize();

  Array<unsigned int> *onSegmentFlagScan =
    new Array<unsigned int>(1, cudaFlag, (unsigned int) nRefine);

  int nOnSegment = FlagSegment(connectivity, onSegmentFlagScan);
  unsigned int *pOnSegmentFlagScan = onSegmentFlagScan->GetPointer();

  int nv_add = nRefine;
  int ne_add = 3*nv_add - nOnSegment;
  int nt_add = 2*nv_add - nOnSegment;

  delete temp;
  temp = new nvtxEvent("Memory", 1);

  if (verboseLevel > 1)
    std::cout << "nTriangle: " << nTriangle
              << ", nEdge: " << nEdge
              << ", nVertex: " << nVertex
              << ", vertices to be inserted in parallel: " << nRefine
              << " ";

  connectivity->vertexCoordinates->SetSize(nVertex + nv_add);
  connectivity->triangleVertices->SetSize(nTriangle + nt_add);
  connectivity->triangleEdges->SetSize(nTriangle + nt_add);
  connectivity->edgeTriangles->SetSize(nEdge + ne_add);

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();

  // All new edges are flagged to be checked
  edgeNeedsChecking->SetSize(nEdge + ne_add);
  edgeNeedsChecking->SetToSeries(nEdge, nEdge + ne_add);

  int *pWantRefine = 0;
  if (triangleWantRefine != 0) {
    triangleWantRefine->SetSize(nTriangle + nt_add);
    triangleWantRefine->SetToValue(1 - (vertexState != 0), nTriangle,
                                   triangleWantRefine->GetSize());
    pWantRefine = triangleWantRefine->GetPointer();
  }

  real *pParam = predicates->GetParamPointer(cudaFlag);

  delete temp;
  temp = new nvtxEvent("Insert", 2);

  // Insert vertices in mesh
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInsertVertices,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devInsertVertices<<<nBlocks, nThreads>>>
      (nRefine, pElementAdd,
       nVertex, nEdge, nTriangle,
       pVcAdd, pOnSegmentFlagScan,
       pVc, pTv, pTe, pEt,
       meshParameter->minx, meshParameter->maxx,
       meshParameter->miny, meshParameter->maxy, nv_add,
       meshParameter->periodicFlagX, meshParameter->periodicFlagY,
       predicates, pParam, pWantRefine);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    for (int n = 0; n < nRefine; n++) {
      int t = pElementAdd[n];
      int e = -1;
      if (t >= nTriangle) {
        e = t - nTriangle;
        t = -1;
      }

      InsertVertex(n, t, e, nVertex, nEdge, nTriangle,
                   pVcAdd[n].x, pVcAdd[n].y,
                   3*n - pOnSegmentFlagScan[n] + nEdge,
                   2*n - pOnSegmentFlagScan[n]+ nTriangle,
                   pVc, pTv, pTe, pEt,
                   meshParameter->minx, meshParameter->maxx,
                   meshParameter->miny, meshParameter->maxy, nv_add,
                   meshParameter->periodicFlagX, meshParameter->periodicFlagY,
                   predicates, pParam, pWantRefine);
    }
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("Insert.prof", nRefine, elapsedTime, cudaFlag);
#endif

  delete onSegmentFlagScan;

  delete temp;
  delete nvtxInsert;
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Refine::InsertVertices<real>(Connectivity * const connectivity,
                             const MeshParameter *meshParameter,
                             const Predicates *predicates,
                             Array<real> * const vertexState,
                             Array<int> * const triangleWantRefine);
template void
Refine::InsertVertices<real3>(Connectivity * const connectivity,
                              const MeshParameter *meshParameter,
                              const Predicates *predicates,
                              Array<real3> * const vertexState,
                              Array<int> * const triangleWantRefine);
template void
Refine::InsertVertices<real4>(Connectivity * const connectivity,
                              const MeshParameter *meshParameter,
                              const Predicates *predicates,
                              Array<real4> * const vertexState,
                              Array<int> * const triangleWantRefine);

}  // namespace astrix
