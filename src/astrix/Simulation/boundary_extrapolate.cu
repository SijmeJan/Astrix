// -*-c++-*-
/*! \file boundary_extrapolate.cu
  \brief Functions for setting boundary conditions using extrapolation

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "./simulation.h"
#include "../Mesh/mesh.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "../Common/helper_math.h"

namespace astrix {

//######################################################################
/*! \brief Extrapolate boundaries single triangle

If triangle has exactly one vertex on the boundary, we extrapolate the state to this vertex by using the state at the other two vertices.

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void ExtrapolateSingle(int n, const int3 *pTv, const real2 *pVc,
                       const int *pVbf, realNeq *pState)
{
  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  int v0 = a;
  if ((pVbf[v0] % 4) != 0) {
    // v0 lies on x boundary
    int v1 = b;
    int v2 = c;

    if ((pVbf[v1] % 4) == 0 &&
        (pVbf[v2] % 4) == 0) {
      // If other two vertices lie on opposite y
      if ((pVc[v1].y - pVc[v0].y)*(pVc[v2].y - pVc[v0].y) <= 0.0) {
        real f1 =
          fabs(pVc[v1].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);
        real f2 =
          fabs(pVc[v2].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
      }
    }
  }

  v0 = b;
  if ((pVbf[v0] % 4) != 0) {
    // v0 lies on x boundary
    int v1 = a;
    int v2 = c;

    if ((pVbf[v1] % 4) == 0 &&
        (pVbf[v2] % 4) == 0) {
      if ((pVc[v1].y - pVc[v0].y)*(pVc[v2].y - pVc[v0].y) <= 0.0) {
        real f1 =
          fabs(pVc[v1].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);
        real f2 =
          fabs(pVc[v2].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
      }
    }
  }

  v0 = c;
  if ((pVbf[v0] % 4) != 0) {
    // v0 lies on x boundary
    int v1 = a;
    int v2 = b;

    if ((pVbf[v1] % 4) == 0 &&
        (pVbf[v2] % 4) == 0) {
      if ((pVc[v1].y - pVc[v0].y)*(pVc[v2].y - pVc[v0].y) <= 0.0) {
        real f1 =
          fabs(pVc[v1].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);
        real f2 =
          fabs(pVc[v2].y - pVc[v0].y)/fabs(pVc[v2].y - pVc[v1].y);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
      }
    }
  }

  v0 = a;
  if (pVbf[v0] > 3) {
    // v0 lies on y boundary
    int v1 = b;
    int v2 = c;

    if (pVbf[v1] < 4 &&
        pVbf[v2] < 4) {
      if ((pVc[v1].x - pVc[v0].x)*(pVc[v2].x - pVc[v0].x) <= 0.0) {
        real f1 =
          fabs(pVc[v1].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);
        real f2 =
          fabs(pVc[v2].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
     }
    }
  }

  v0 = b;
  if (pVbf[v0] > 3) {
    // v0 lies on y boundary
    int v1 = a;
    int v2 = c;

    if (pVbf[v1] < 4 &&
        pVbf[v2] < 4) {
      if ((pVc[v1].x - pVc[v0].x)*(pVc[v2].x - pVc[v0].x) <= 0.0) {
        real f1 = fabs(pVc[v1].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);
        real f2 = fabs(pVc[v2].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
      }
    }
  }

  v0 = c;
  if (pVbf[v0] > 3) {
    // v0 lies on y boundary
    int v1 = a;
    int v2 = b;

    if (pVbf[v1] < 4 &&
        pVbf[v2] < 4) {
      if ((pVc[v1].x - pVc[v0].x)*(pVc[v2].x - pVc[v0].x) <= 0.0) {
        real f1 = fabs(pVc[v1].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);
        real f2 = fabs(pVc[v2].x - pVc[v0].x)/fabs(pVc[v2].x - pVc[v1].x);

        pState[v0] = f1*pState[v2] + f2*pState[v1];
      }
    }
  }
}

//######################################################################
/*! \brief Kernel for setting boundaries through extrapolation

If a triangle has exactly one vertex on the boundary, we extrapolate the state to this vertex by using the state at the other two vertices.

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVc Pointer to coordinates of vertices
\param *pVbf Pointer to array of boundary flags
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devExtrapolateBoundaries(int nTriangle, const int3 *pTv, const real2 *pVc,
                         const int *pVbf, realNeq *pState)
{
  // n=triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    ExtrapolateSingle(n, pTv, pVc, pVbf, pState);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Set state in corners to zero

When extrapolating, the corners of the mesh need special attention. In this function, we set the state to zero.

\param n Vertex to consider
\param *pVbf Pointer to array of boundary flags
\param *pState Pointer to state vector*/
//######################################################################

__host__ __device__
void SetCornersToZero(int n, const int *pVbf, real4 *pState)
{
  const real zero = (real) 0.0;

  if (pVbf[n] == 5 ||
      pVbf[n] == 6 ||
      pVbf[n] == 9 ||
      pVbf[n] == 10) {
    pState[n].x = zero;
    pState[n].y = zero;
    pState[n].z = zero;
    pState[n].w = zero;
  }
}

//! Version for 3 equation
__host__ __device__
void SetCornersToZero(int n, const int *pVbf, real3 *pState)
{
  if (pVbf[n] == 5 ||
      pVbf[n] == 6 ||
      pVbf[n] == 9 ||
      pVbf[n] == 10) {
    pState[n].x = (real) 0.0;
    pState[n].y = (real) 0.0;
    pState[n].z = (real) 0.0;
  }
}

//! Version for one equation
__host__ __device__
void SetCornersToZero(int n, const int *pVbf, real *pState)
{
  if (pVbf[n] == 5 ||
      pVbf[n] == 6 ||
      pVbf[n] == 9 ||
      pVbf[n] == 10) {
    pState[n] = (real) 0.0;
  }
}


//######################################################################
/*! \brief Kernel setting state in corners to zero

When extrapolating, the corners of the mesh need special attention. In this function, we set the state to zero.

\param nVertex Total number of vertices in Mesh
\param *pVbf Pointer to array of boundary flags
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devSetCornersToZero(int nVertex, const int *pVbf, realNeq *pState)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    SetCornersToZero(n, pVbf, pState);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Extrapolate boundaries at Mesh corners

The state has been set to zero in the corners previously. Now extrapolate the state from the two segments coming together in the corner.

\param n Triangle to consider
\param *pVbf Pointer to array of boundary flags
\param *pTv Pointer to triangle vertices
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void ExtrapolateCorners(int n, const int *pVbf,
                        const int3* __restrict__ pTv, realNeq *pState)
{
  const real half = (real) 0.5;

  int a = pTv[n].x;
  int b = pTv[n].y;
  int c = pTv[n].z;

  int v1 = a;
  if (pVbf[v1] == 5 ||
      pVbf[v1] == 6 ||
      pVbf[v1] == 9 ||
      pVbf[v1] == 10) {
    int v2 = b;
    int v3 = c;
    if (pVbf[v2] != 0) pState[v1] += half*pState[v2];
    if (pVbf[v3] != 0) pState[v1] += half*pState[v3];
  }
  v1 = b;
  if (pVbf[v1] == 5 ||
      pVbf[v1] == 6 ||
      pVbf[v1] == 9 ||
      pVbf[v1] == 10) {
    int v2 = a;
    int v3 = c;
    if (pVbf[v2] != 0) pState[v1] += half*pState[v2];
    if (pVbf[v3] != 0) pState[v1] += half*pState[v3];
  }
  v1 = c;
  if (pVbf[v1] == 5 ||
      pVbf[v1] == 6 ||
      pVbf[v1] == 9 ||
      pVbf[v1] == 10) {
    int v2 = a;
    int v3 = b;
    if (pVbf[v2] != 0) pState[v1] += half*pState[v2];
    if (pVbf[v3] != 0) pState[v1] += half*pState[v3];
  }
}

//######################################################################
/*! \brief Kernel extrapolating boundaries at Mesh corners

The state has been set to zero in the corners previously. Now extrapolate the state from the two segments coming together in the corner.

\param nTriangle Total number of triangles in Mesh
\param *pVbf Pointer to array of boundary flags
\param *pTv Pointer to triangle vertices
\param *pState Pointer to state vector*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devExtrapolateCorners(int nTriangle, const int *pVbf,
                      const int3* __restrict__ pTv, realNeq *pState)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    ExtrapolateCorners(n, pVbf, pTv, pState);
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Consider all triangles in the Mesh. If a triangle has exactly one
vertex on the boundary, we extrapolate the state to this vertex by using
the state at the other two vertices.*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::ExtrapolateBoundaries()
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexState->GetPointer();

  const real2 *pVc = mesh->VertexCoordinatesData();
  const int3 *pTv = mesh->TriangleVerticesData();
  const int *pVbf = mesh->VertexBoundaryFlagData();

  // First, extrapolate all boundaries including corners
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devExtrapolateBoundaries<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devExtrapolateBoundaries<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVc, pVbf, pState);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    // Extrapolate x
    for (int n = 0; n < nTriangle; n++)
      ExtrapolateSingle(n, pTv, pVc, pVbf, pState);
  }

  // Set the state to zero in the corners
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetCornersToZero<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devSetCornersToZero<<<nBlocks, nThreads>>>
      (nVertex, pVbf, pState);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    // Set corners to zero
    for (int n = 0; n < nVertex; n++)
      SetCornersToZero(n, pVbf, pState);
  }

  // Set the state in the corners to the average of the to joining sides
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devExtrapolateCorners<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
    devExtrapolateCorners<<<nBlocks, nThreads>>>
      (nTriangle, pVbf, pTv, pState);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++)
      ExtrapolateCorners(n, pVbf, pTv, pState);
  }
}

}  // namespace astrix
