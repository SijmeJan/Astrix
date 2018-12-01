// -*-c++-*-
/*! \file adjuststate.cu
\brief Functions for adjusting state when flipping edges to conserve mass and momentum

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
#include "./delaunay.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../../Common/helper_math.h"
#include "../../Common/atomic.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/state.h"

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

template<class realNeq>
__host__ __device__
void AdjustStateSingle(int i, int *pEnd, int3 *pTv, int3 *pTe, int2 *pEt,
                       int nVertex, real2 *pVc, real *pVarea,
                       const Predicates *pred, real *pParam,
                       real Px, real Py, realNeq *pState)
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

  // Six times change in vertex areas
  real dVa = T3 + T4 - T2;
  real dVb = T3 - T1 - T2;
  real dVc = T4 - T1 - T2;
  real dVd = T4 + T3 - T1;

  // Update vertex areas
  real Va = AtomicAdd(&(pVarea[a]), dVa*sixth);
  real Vb = AtomicAdd(&(pVarea[b]), dVb*sixth);
  real Vc = AtomicAdd(&(pVarea[c]), dVc*sixth);
  real Vd = AtomicAdd(&(pVarea[d]), dVd*sixth);

  /*
  // Keep mass at vertices constant: unstable!
  realNeq stateAdjustA = (-dVa/(6.0*Va + dVa))*pState[a];
  AtomicAdd(&(pState[a]), stateAdjustA);
  realNeq stateAdjustB = (-dVb/(6.0*Vb + dVb))*pState[b];
  AtomicAdd(&(pState[b]), stateAdjustB);
  realNeq stateAdjustC = (-dVc/(6.0*Vc + dVc))*pState[c];
  AtomicAdd(&(pState[c]), stateAdjustC);
  realNeq stateAdjustD = (-dVd/(6.0*Vd + dVd))*pState[d];
  AtomicAdd(&(pState[d]), stateAdjustD);
  */
  /*
  // Adjust all states by single amount
  realNeq stateAdjust =
    pState[a]*dVa + pState[b]*dVb + pState[c]*dVc + pState[d]*dVd;

  // Six times total volume after flip
  real denom =
    6.0*(Va + Vb + Vc + Vd) + dVa + dVb + dVc + dVd;
  denom = -1.0/denom;

  // Adjust state
  stateAdjust *= denom;

  AtomicAdd(&(pState[a]), stateAdjust);
  AtomicAdd(&(pState[b]), stateAdjust);
  AtomicAdd(&(pState[c]), stateAdjust);
  AtomicAdd(&(pState[d]), stateAdjust);
  */

  /*
  // Least squares approach
  real A1 = Va + sixth*dVa;
  real A2 = Vb + sixth*dVb;
  real A3 = Vc + sixth*dVc;
  real A4 = Vd + sixth*dVd;

  real nm00 = A4*A4 + A1*A1;
  real nm01 = A1*A2;
  real nm02 = A1*A3;
  real nm11 = A4*A4 + A2*A2;
  real nm12 = A2*A3;
  real nm22 = A4*A4 + A3*A3;

  real invN00 = nm11*nm22 - nm12*nm12;
  real invN01 = nm02*nm12 - nm01*nm22;
  real invN02 = nm01*nm12 - nm02*nm11;
  real invN10 = nm12*nm02 - nm01*nm22;
  real invN11 = nm00*nm22 - nm02*nm02;
  real invN12 = nm02*nm01 - nm00*nm12;
  real invN20 = nm01*nm12 - nm11*nm02;
  real invN21 = nm01*nm02 - nm00*nm12;
  real invN22 = nm00*nm11 - nm01*nm01;

  real det = nm00*invN00 + nm01*invN10 + nm02*invN20;
  det = -1.0/det;

  realNeq dM =
    pState[a]*dVa + pState[b]*dVb + pState[c]*dVc + pState[d]*dVd;

  realNeq stateAdjustA = det*(invN00*A1 + invN01*A2 + invN02*A3)*dM;
  realNeq stateAdjustB = det*(invN10*A1 + invN11*A2 + invN12*A3)*dM;
  realNeq stateAdjustC = det*(invN20*A1 + invN21*A2 + invN22*A3)*dM;

  real iA4 = -1.0/A4;
  realNeq stateAdjustD =
    (A1*iA4)*stateAdjustA +
    (A2*iA4)*stateAdjustB +
    (A3*iA4)*stateAdjustC +
    iA4*dM;

  AtomicAdd(&(pState[a]), stateAdjustA);
  AtomicAdd(&(pState[b]), stateAdjustB);
  AtomicAdd(&(pState[c]), stateAdjustC);
  AtomicAdd(&(pState[d]), stateAdjustD);
  */

  // Erase deviations
  real A1 = Va + sixth*dVa;
  real A2 = Vb + sixth*dVb;
  real A3 = Vc + sixth*dVc;
  real A4 = Vd + sixth*dVd;

  realNeq M =
    pState[a]*Va + pState[b]*Vb + pState[c]*Vc + pState[d]*Vd;
  realNeq newState = M/(A1 + A2 + A3 + A4);

  realNeq stateAdjustA = newState - pState[a];
  realNeq stateAdjustB = newState - pState[b];
  realNeq stateAdjustC = newState - pState[c];
  realNeq stateAdjustD = newState - pState[d];

  AtomicAdd(&(pState[a]), stateAdjustA);
  AtomicAdd(&(pState[b]), stateAdjustB);
  AtomicAdd(&(pState[c]), stateAdjustC);
  AtomicAdd(&(pState[d]), stateAdjustD);

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

template<class realNeq>
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

template<class realNeq>
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
                                       devAdjustState<realNeq>,
                                       (size_t) 0, 0);

    devAdjustState<realNeq><<<nBlocks, nThreads>>>
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
Delaunay::AdjustState<real>(Connectivity * const connectivity,
                            Array<real> * const vertexState,
                            const Predicates *predicates,
                            const MeshParameter *meshParameter,
                            const int nNonDel);
template void
Delaunay::AdjustState<real3>(Connectivity * const connectivity,
                             Array<real3> * const vertexState,
                             const Predicates *predicates,
                             const MeshParameter *meshParameter,
                             const int nNonDel);
template void
Delaunay::AdjustState<real4>(Connectivity * const connectivity,
                             Array<real4> * const vertexState,
                             const Predicates *predicates,
                             const MeshParameter *meshParameter,
                             const int nNonDel);
}  // namespace astrix
