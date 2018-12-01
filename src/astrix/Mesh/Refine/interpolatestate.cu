// -*-c++-*-
/*! \file interpolatestate.cu
\brief Functions for interpolating state when refining Mesh

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
#include "./refine.h"
#include "../triangleLow.h"
#include "../../Common/cudaLow.h"
#include "../Connectivity/connectivity.h"
#include "../Param/meshparameter.h"
#include "../../Common/helper_math.h"

namespace astrix {

//######################################################################
/*! \brief Interpolate state when inserting point (x, y)

Use linear interpolation to determine state at new vertex.

\param t Triangle into which to insert point (x, y)
\param e Edge onto which to insert point (x, y)
\param indexInVertexArray Index of vertex to insert
\param *state Pointer to state vector
\param *pTv Pointer to triangle vertices
\param *pEt Pointer to edge triangles
\param *pVc Pointer to vertex coordinates
\param x x-coordinate of point to insert
\param y y-coordinate of point to insert
\param *wantRefine Pointer to array of flags specifying if triangle needs to be refined. Set to zero for all inserted points.
\param nVertex Total number of vertices in Mesh
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//######################################################################

template<class realNeq>
__host__ __device__
void InterpolateSingle(int t, int e, int indexInVertexArray, realNeq *state,
                       int3 *pTv, int2 *pEt, real2 *pVc,
                       real x, real y,
                       int *wantRefine,
                       int nVertex, real Px, real Py)
{
  const real half  = (real) 0.5;

  // Unflag triangles to be refined
  if (t != -1)
    wantRefine[t] = 0;
  if (t == -1) t = pEt[e].x;
  if (t != -1)
    wantRefine[t] = 0;
  if (t == -1) t = pEt[e].y;
  if (t != -1)
    wantRefine[t] = 0;

  int v1 = pTv[t].x;
  int v2 = pTv[t].y;
  int v3 = pTv[t].z;

  real ax, bx, cx, ay, by, cy, dx = x, dy = y;
  GetTriangleCoordinates(pVc, v1, v2, v3,
                         nVertex, Px, Py,
                         ax, bx, cx, ay, by, cy);

  real T = half*((ax - cx)*(by - cy) - (ay - cy)*(bx - cx));
  real A = half*((dx - cx)*(by - cy) - (dy - cy)*(bx - cx));
  real B = half*((ax - cx)*(dy - cy) - (ay - cy)*(dx - cx));
  real C = half*((ax - dx)*(by - dy) - (ay - dy)*(bx - dx));

  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  state[indexInVertexArray] =
    (A*state[v1] + B*state[v2] + C*state[v3])/T;
}

//######################################################################
/*! \brief Kernel interpolating state for all points to be inserted

Use linear interpolation to determine state at new vertices.

\param nRefine Total number of points to add
\param *pTriangleAdd Pointer to array of triangles into which to insert points
\param *pEdgeAdd Pointer to array of edges onto which to insert points
\param *state Pointer to state vector
\param nVertex Total number of vertices in Mesh
\param nTriangle Total number of triangles in Mesh
\param *pVcAdd Pointer to vertex coordinates to add
\param *pVc Pointer to vertex coordinates
\param *pTv Pointer to triangle vertices
\param *pEt Pointer to edge triangles
\param *wantRefine Pointer to array of flags specifying if triangle needs to be refined. Set to zero for all inserted points.
\param Px Periodic domain size x
\param Py Periodic domain size y*/
//######################################################################

template<class realNeq>
__global__ void
devInterpolateState(int nRefine, int *pElementAdd, realNeq *state,
                    int nVertex, int nTriangle, real2 *pVcAdd,
                    real2 *pVc, int3 *pTv, int2 *pEt,
                    int *wantRefine, real Px, real Py)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nRefine) {
    int t = pElementAdd[i];
    int e = -1;
    if (t >= nTriangle) {
      e = t - nTriangle;
      t = -1;
    }

    InterpolateSingle<realNeq>(t, e, i + nVertex, state,
                               pTv, pEt, pVc,
                               pVcAdd[i].x, pVcAdd[i].y, wantRefine,
                               nVertex, Px, Py);

    i += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Use linear interpolation to give newly inserted vertices a state

\param *connectivity Pointer to connectivity
\param *meshParameter Pointer to mesh parameters
\param *vertexState Pointer to Array containing state vector
\param *triangleWantRefine Pointer to array indicating which triangles are to be refined*/
//######################################################################

template<class realNeq>
void Refine::InterpolateState(Connectivity * const connectivity,
                              const MeshParameter *meshParameter,
                              Array<realNeq> * const vertexState,
                              Array<int> * const triangleWantRefine)
{
  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();
  int nRefine = elementAdd->GetSize();

  // Interpolate state
  vertexState->SetSize(nVertex + nRefine);
  realNeq *state = vertexState->GetPointer();

  real2 *pVc = connectivity->vertexCoordinates->GetPointer();
  int3 *pTv = connectivity->triangleVertices->GetPointer();
  int2 *pEt = connectivity->edgeTriangles->GetPointer();
  real2 *pVcAdd = vertexCoordinatesAdd->GetPointer();

  int *pWantRefine = triangleWantRefine->GetPointer();

  int *pElementAdd = elementAdd->GetPointer();

  real Px = meshParameter->maxx - meshParameter->minx;
  real Py = meshParameter->maxy - meshParameter->miny;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devInterpolateState<realNeq>,
                                       (size_t) 0, 0);

    devInterpolateState<realNeq><<<nBlocks, nThreads>>>
      (nRefine, pElementAdd, state,
       nVertex, nTriangle, pVcAdd, pVc, pTv, pEt,
       pWantRefine, Px, Py);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nRefine; n++) {
      int t = pElementAdd[n];
      int e = -1;
      if (t >= nTriangle) {
        e = t - nTriangle;
        t = -1;
      }

      InterpolateSingle<realNeq>(t, e, n + nVertex, state,
                                 pTv, pEt, pVc, pVcAdd[n].x, pVcAdd[n].y,
                                 pWantRefine, nVertex, Px, Py);
    }
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Refine::InterpolateState<real>(Connectivity * const connectivity,
                               const MeshParameter *meshParameter,
                               Array<real> * const vertexState,
                               Array<int> * const triangleWantRefine);
template void
Refine::InterpolateState<real3>(Connectivity * const connectivity,
                                const MeshParameter *meshParameter,
                                Array<real3> * const vertexState,
                                Array<int> * const triangleWantRefine);
template void
Refine::InterpolateState<real4>(Connectivity * const connectivity,
                                const MeshParameter *meshParameter,
                                Array<real4> * const vertexState,
                                Array<int> * const triangleWantRefine);

}  // namespace astrix
