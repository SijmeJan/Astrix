// -*-c++-*-
/*! \file adjuststate.cu
\brief Functions for adjusting state when flipping edges to conserve mass and momentum

\section LICENSE
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
#include "./delaunay.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/atomic.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"

namespace astrix {

//#########################################################################
/*! \brief Adjust state around edge \a pEdgeNonDelaunay[i]

\param i Index of non-Delaunay edge to consider
\param *pEnd Pointer to list of edges that are not Delaunay
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pState Pointer to state vector*/
//#########################################################################

__host__ __device__
void AdjustStateSingle(int i, int *pEnd, int3 *pTv, int3 *pTe, int2 *pEt,
                       int nVertex, real2 *pVc, real *pVarea,
                       const Predicates *pred, real *pParam,
                       real Px, real Py, real4 *pState)
{
  const real zero  = (real) 0.0;
  const real sixth = (real) (1.0/6.0);

  // Edge to be flipped
  int edge = pEnd[i];

  // Neighbouring triangles
  int t1 = pEt[edge].x;
  int t2 = pEt[edge].y;

  int d = pTv[t1].z;
  int e = pTv[t1].x;
  int f = pTv[t1].y;

  real Dx, Ex, Fx, Dy, Ey, Fy;
  GetTriangleCoordinates(pVc, e, f, d,
                         nVertex, Px, Py,
                         Dx, Ex, Fx, Dy, Ey, Fy);

  real dx = Fx;
  real dy = Fy;

  int e2 = pTe[t1].y;
  int e3 = pTe[t1].z;

  if (edge == e2) {
    d = pTv[t1].x;
    f = pTv[t1].z;
    dx = Dx;
    dy = Dy;
  }
  if (edge == e3) {
    d = pTv[t1].y;
    f = pTv[t1].x;
    dx = Ex;
    dy = Ey;
  }

  int a = pTv[t2].z;
  int b = pTv[t2].x;
  int c = pTv[t2].y;

  real Ax, Bx, Cx, Ay, By, Cy;
  GetTriangleCoordinates(pVc, b, c, a,
                         nVertex, Px, Py,
                         Ax, Bx, Cx, Ay, By, Cy);

  real ax = Cx;
  real ay = Cy;
  real bx = Ax;
  real by = Ay;
  real cx = Bx;
  real cy = By;
  e2 = pTe[t2].y;
  e3 = pTe[t2].z;
  if (edge == e2) {
    a = pTv[t2].x;
    b = pTv[t2].y;
    c = pTv[t2].z;
    ax = Ax;
    ay = Ay;
    bx = Bx;
    by = By;
    cx = Cx;
    cy = Cy;
  }
  if (edge == e3) {
    a = pTv[t2].y;
    b = pTv[t2].z;
    c = pTv[t2].x;
    ax = Bx;
    ay = By;
    bx = Cx;
    by = Cy;
    cx = Ax;
    cy = Ay;
  }

  // pERIODIC
  TranslateVertexToVertex(b, f, Px, Py, nVertex, dx, dy);

  while (d >= nVertex) d -= nVertex;
  while (d < 0) d += nVertex;

  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Twice triangle area
  real T1 = (cx - dx)*(by - dy) - (cy - dy)*(bx - dx);
  real T2 = (ax - cx)*(by - cy) - (ay - cy)*(bx - cx);
  real T3 = (ax - dx)*(by - dy) - (ay - dy)*(bx - dx);
  real T4 = (ax - cx)*(dy - cy) - (ay - cy)*(dx - cx);

  if (T1 <= zero || T2 <= zero || T3 <= zero || T4 <= zero) {
    T1 = pred->orient2d(cx, cy, bx, by, dx, dy, pParam);
    T2 = pred->orient2d(ax, ay, bx, by, cx, cy, pParam);
    T3 = pred->orient2d(ax, ay, bx, by, dx, dy, pParam);
    T4 = pred->orient2d(ax, ay, dx, dy, cx, cy, pParam);
  }

  real dVa = T2 - T3 - T4;
  real dVb = T1 + T2 - T3;
  real dVc = T1 + T2 - T4;
  real dVd = T1 - T3 - T4;

  real densAdjust =
    pState[a].x*dVa + pState[b].x*dVb + pState[c].x*dVc + pState[d].x*dVd;
  real momxAdjust =
    pState[a].y*dVa + pState[b].y*dVb + pState[c].y*dVc + pState[d].y*dVd;
  real momyAdjust =
    pState[a].z*dVa + pState[b].z*dVb + pState[c].z*dVc + pState[d].z*dVd;

  real Va = AtomicAdd(&(pVarea[a]), -dVa*sixth);
  real Vb = AtomicAdd(&(pVarea[b]), -dVb*sixth);
  real Vc = AtomicAdd(&(pVarea[c]), -dVc*sixth);
  real Vd = AtomicAdd(&(pVarea[d]), -dVd*sixth);

  real denom =
    6.0*(Va + Vb + Vc + Vd) - dVa - dVb - dVc - dVd;
  denom = 1.0/denom;

  // Adjust state. Note that we leave the energy (= pressure) alone.
  densAdjust *= denom;
  momxAdjust *= denom;
  momyAdjust *= denom;

  AtomicAdd(&(pState[a].x), densAdjust);
  AtomicAdd(&(pState[b].x), densAdjust);
  AtomicAdd(&(pState[c].x), densAdjust);
  AtomicAdd(&(pState[d].x), densAdjust);

  AtomicAdd(&(pState[a].y), momxAdjust);
  AtomicAdd(&(pState[b].y), momxAdjust);
  AtomicAdd(&(pState[c].y), momxAdjust);
  AtomicAdd(&(pState[d].y), momxAdjust);

  AtomicAdd(&(pState[a].z), momyAdjust);
  AtomicAdd(&(pState[b].z), momyAdjust);
  AtomicAdd(&(pState[c].z), momyAdjust);
  AtomicAdd(&(pState[d].z), momyAdjust);
}

__host__ __device__
void AdjustStateSingle(int i, int *pEnd, int3 *pTv, int3 *pTe, int2 *pEt,
                       int nVertex, real2 *pVc, real *pVarea,
                       const Predicates *pred, real *pParam,
                       real Px, real Py, real3 *pState)
{
  // Dummy: not supported for three equations
}

__host__ __device__
void AdjustStateSingle(int i, int *pEnd, int3 *pTv, int3 *pTe, int2 *pEt,
                       int nVertex, real2 *pVc, real *pVarea,
                       const Predicates *pred, real *pParam,
                       real Px, real Py, real *pState)
{
  // Dummy: not supported for one equation
}

//######################################################################
/*! \brief Kernel adjusting state for all edge entries in \a pEnd

\param nNonDel Number of non-Delaunay edges
\param *pEnd Pointer to list of edges that are not Delaunay
\param *pTv Pointer to triangle vertices
\param *pTe Pointer to triangle edges
\param *pEt Pointer to edge triangles
\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to vertex coordinates
\param *pVarea Pointer to vertex areas (Voronoi cells)
\param *pred Pointer to initialised Predicates object
\param *pParam Pointer to initialised Predicates parameter vector
\param Px Periodic domain size x
\param Py Periodic domain size y
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devAdjustState(int nNonDel, int *pEnd, int3 *pTv, int3 *pTe, int2 *pEt,
               int nVertex, real2 *pVc, real *pVarea,
               const Predicates *pred, real *pParam,
               real Px, real Py, realNeq *pState)
{
  // i = edge number
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nNonDel) {
    AdjustStateSingle(i, pEnd, pTv, pTe, pEt, nVertex, pVc, pVarea,
                      pred, pParam, Px, Py, pState);

    // Next edge
    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Adjust state to conserve mass and momentum when flipping edges. All edges in \a edgeNonDelaunay are considered.

\param *connectivity Pointer to basic Mesh data
\param *vertexState Pointer to state vector
\param *predicates Pointer to Predicates object, used to check Delaunay property without roundoff error
\param *meshParameter Pointer to Mesh parameters
\param nNonDel Number of edges to be flipped*/
//######################################################################

template<class realNeq, ConservationLaw CL>
void Delaunay::AdjustState(Connectivity * const connectivity,
                           Array<realNeq> * const vertexState,
                           const Predicates *predicates,
                           const MeshParameter *meshParameter,
                           const int nNonDel)
{
  int nVertex = connectivity->vertexCoordinates->GetSize();

  int *pEnd = edgeNonDelaunay->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  connectivity->CalcVertexArea(Px, Py);
  real *pVarea = connectivity->vertexArea->GetPointer();

  realNeq *pState = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int3 *pTe = connectivity->triangleEdges->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();

  real *pParam = predicates->GetParamPointer(cudaFlag);

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devAdjustState<realNeq, CL>,
                                       (size_t) 0, 0);

    devAdjustState<realNeq, CL><<<nBlocks, nThreads>>>
      (nNonDel, pEnd, pTv, pTe, pEt, nVertex, pVc, pVarea,
       predicates, pParam, Px, Py, pState);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  } else {
    for (int i = 0; i < nNonDel; i++)
      AdjustStateSingle(i, pEnd, pTv, pTe, pEt, nVertex, pVc, pVarea,
                        predicates, pParam, Px, Py, pState);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Delaunay::AdjustState<real, CL_ADVECT>(Connectivity * const connectivity,
                                       Array<real> * const vertexState,
                                       const Predicates *predicates,
                                       const MeshParameter *meshParameter,
                                       const int nNonDel);
template void
Delaunay::AdjustState<real, CL_BURGERS>(Connectivity * const connectivity,
                                        Array<real> * const vertexState,
                                        const Predicates *predicates,
                                        const MeshParameter *meshParameter,
                                        const int nNonDel);
template void
Delaunay::AdjustState<real3, CL_CART_ISO>(Connectivity * const connectivity,
                                          Array<real3> * const vertexState,
                                          const Predicates *predicates,
                                          const MeshParameter *meshParameter,
                                          const int nNonDel);
template void
Delaunay::AdjustState<real4, CL_CART_EULER>(Connectivity * const connectivity,
                                            Array<real4> * const vertexState,
                                            const Predicates *predicates,
                                            const MeshParameter *meshParameter,
                                            const int nNonDel);
}  // namespace astrix
