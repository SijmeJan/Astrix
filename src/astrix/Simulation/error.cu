// -*-c++-*-
/*! \file error.cu
\brief File containing function to estimate the local truncation error, which can be used to decide where to refine the mesh.

When using an adaptive mesh, we have to decide, based on the current state of the flow, where mesh refinement is needed. We use the method outlined in Lapenta (2004), comparing cell-based and node-based operators, which gives an estimate of the local truncation error.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"
#include "./simulation.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief For a single triangle, calculate triangle-based operator and its contribution to the vertex-based operator.

For a single triangle, calculate triangle-based operator (for the internal energy equation) and its contribution to the vertex-based operator.

 \param i triangle to be dealt with.
 \param nVertex number of vertices in the current mesh.
 \param nTriangle number of triangles in the current mesh.
 \param *pTv Pointer to triangle vertices
 \param G specific heat ratio
 \param *triL pointer to triangle edge lengths.
 \param *pTn1 Pointer to triangle normals first edge
 \param *pTn2 Pointer to triangle normals second edge
 \param *pTn3 Pointer to triangle normals third edge
 \param *pVertexArea pointer to cell volume (Voronoi area)
 \param *state Pointer to state vector
 \param *pVertexOperator pointer to vertex-based operator (output).
 \param *pTriangleOperator pointer to triangle-based operator (output).*/
//######################################################################

__host__ __device__
void CalcOperatorEnergySingle(int i, int nVertex, int nTriangle,
                              const int3 *pTv, real G, const real3 *triL,
                              const real2 *pTn1, const real2 *pTn2,
                              const real2 *pTn3,
                              const real *pVertexArea, real4 *state,
                              real *pVertexOperator,
                              real *pTriangleOperator)
{
  const real sixth = (real) (1.0/6.0);
  const real tenth = (real) 0.1;
  const real third = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one   = (real) 1.0;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Triangle edge lengths
  real tl1 = triL[i].x;
  real tl2 = triL[i].y;
  real tl3 = triL[i].z;

  // Triangle inward pointing normals
  real nx1 = pTn1[i].x;
  real nx2 = pTn2[i].x;
  real nx3 = pTn3[i].x;
  real ny1 = pTn1[i].y;
  real ny2 = pTn2[i].y;
  real ny3 = pTn3[i].y;

  // State at vertex a
  real d1 = state[a].x;
  real u1 = state[a].y/d1;
  real v1 = state[a].z/d1;
  real p1 = state[a].w;
  real i1 = p1/((G - one)*d1);

  // State at vertex b
  real d2 = state[b].x;
  real u2 = state[b].y/d2;
  real v2 = state[b].z/d2;
  real p2 = state[b].w;
  real i2 = p2/((G - one)*d2);

  // State at vertex c
  real d3 = state[c].x;
  real u3 = state[c].y/d3;
  real v3 = state[c].z/d3;
  real p3 = state[c].w;
  real i3 = p3/((G - one)*d3);

  // Make sure error is finite when velocity is zero
  u1 += tenth*sqrt(G*p1/d1);
  u2 += tenth*sqrt(G*p2/d2);
  u3 += tenth*sqrt(G*p3/d3);

  // Cell centred velocity divergence
  real divuc =
    (u1*nx1 + v1*ny1)*tl1 + (u2*nx2 + v2*ny2)*tl2 + (u3*nx3 + v3*ny3)*tl3;

  // Cell centred internal energy gradient
  real gradEx = i1*nx1*tl1 + i2*nx2*tl2 + i3*nx3*tl3;
  real gradEy = i1*ny1*tl1 + i2*ny2*tl2 + i3*ny3*tl3;

  // Cell-averaged state
  real dc = third*(d1 + d2 + d3);
  real uc = third*(u1 + u2 + u3);
  real vc = third*(v1 + v2 + v3);
  real pc = third*(p1 + p2 + p3);
  real ic = third*(i1 + i2 + i3);

  real s = half*(tl1 + tl2 + tl3);
  // Triangle area
  real A = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

  // Cell-centred operator
  real operatorTriangle = pc*divuc/dc + uc*gradEx + vc*gradEy;
  pTriangleOperator[i] = operatorTriangle*sqrt(half/A)/ic;

  // Contribution to vertex operator
  real dA = (p1*divuc/d1 + u1*gradEx + v1*gradEy)*sqrt(sixth/pVertexArea[a])/ic;
  real dB = (p2*divuc/d2 + u2*gradEx + v2*gradEy)*sqrt(sixth/pVertexArea[b])/ic;
  real dC = (p3*divuc/d3 + u3*gradEx + v3*gradEy)*sqrt(sixth/pVertexArea[c])/ic;

  // Construct vertex-centred operator
  AtomicAdd(&pVertexOperator[a], dA);
  AtomicAdd(&pVertexOperator[b], dB);
  AtomicAdd(&pVertexOperator[c], dC);
}

__host__ __device__
void CalcOperatorEnergySingle(int i, int nVertex, int nTriangle,
                              const int3 *pTv, real G, const real3 *triL,
                              const real2 *pTn1, const real2 *pTn2,
                              const real2 *pTn3,
                              const real *pVertexArea, real3 *state,
                              real *pVertexOperator,
                              real *pTriangleOperator)
{
  // Dummy, not supported for 3 equations
}

__host__ __device__
void CalcOperatorEnergySingle(int i, int nVertex, int nTriangle,
                              const int3 *pTv, real G, const real3 *triL,
                              const real2 *pTn1, const real2 *pTn2,
                              const real2 *pTn3,
                              const real *pVertexArea, real *state,
                              real *pVertexOperator,
                              real *pTriangleOperator)
{
  const real sixth = (real) (1.0/6.0);
  const real half  = (real) 0.5;

  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Triangle edge lengths
  real tl1 = triL[i].x;
  real tl2 = triL[i].y;
  real tl3 = triL[i].z;

  // Triangle inward pointing normals
  real nx1 = pTn1[i].x;
  real nx2 = pTn2[i].x;
  real nx3 = pTn3[i].x;

  // State at vertex a
  real d1 = state[a];
  // State at vertex b
  real d2 = state[b];
  // State at vertex c
  real d3 = state[c];

  // Cell centred internal energy gradient
  real gradUx = d1*nx1*tl1 + d2*nx2*tl2 + d3*nx3*tl3;

  // Cell-averaged state

  real s = half*(tl1 + tl2 + tl3);
  // Triangle area
  real A = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

  // Cell-centred operator
  real operatorTriangle = gradUx;
  pTriangleOperator[i] = operatorTriangle*sqrt(half/A);

  // Contribution to vertex operator
  real dA = gradUx*sqrt(sixth/pVertexArea[a]);
  real dB = gradUx*sqrt(sixth/pVertexArea[b]);
  real dC = gradUx*sqrt(sixth/pVertexArea[c]);

  // Construct vertex-centred operator
  AtomicAdd(&pVertexOperator[a], dA);
  AtomicAdd(&pVertexOperator[b], dB);
  AtomicAdd(&pVertexOperator[c], dC);
}

//##############################################################################
/*! \brief For a single triangle, calculate the estimated local truncation error.

For a single triangle, calculate the estimated local truncation error based on the difference between the triangle-based operator and the vertex-based operator.
 \param i triangle to be dealt with.
 \param nVertex number of vertices in the current mesh.
 \param *pTv Pointer to triangle vertices
 \param *pVertexOperator pointer to vertex-based operator.
 \param *pTriangleOperator pointer to triangle-based operator.
 \param *pErrorEstimate array with estimated local truncation error (output).*/
//##############################################################################

__host__ __device__
void CalcErrorEstimateSingle(int i, int nVertex, const int3 *pTv,
                             real *pVertexOperator,
                             real *pTriangleOperator,
                             real *pErrorEstimate)
{
  int a = pTv[i].x;
  int b = pTv[i].y;
  int c = pTv[i].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  // Error estimate at vertices
  real E1 = fabs(pVertexOperator[a] - pTriangleOperator[i]);
  real E2 = fabs(pVertexOperator[b] - pTriangleOperator[i]);
  real E3 = fabs(pVertexOperator[c] - pTriangleOperator[i]);

  // Maximum norm of error over triangle
  pErrorEstimate[i] = max(pErrorEstimate[i], max(E1, max(E2, E3)));
}

// #########################################################################
/*! \brief Kernel calculating triangle-based and vertex-based operators.

Kernel calculating triangle-based and vertex-based operators for the internal energy equation.

 \param nVertex number of vertices in the current mesh.
 \param nTriangle number of triangles in the current mesh.
 \param *pTv Pointer to triangle vertices
 \param G specific heat ratio
 \param *triL pointer to triangle edge lengths.
 \param *pTn1 Pointer to triangle normals first edge
 \param *pTn2 Pointer to triangle normals second edge
 \param *pTn3 Pointer to triangle normals third edge
 \param *pVertexArea pointer to cell volume (Voronoi area)
 \param *state Pointer to state vector
 \param *pVertexOperator pointer to vertex-based operator (output).
 \param *pTriangleOperator pointer to triangle-based operator (output).*/
// #########################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcOperatorEnergy(int nVertex, int nTriangle,
                      const int3 *pTv, real G, const real3 *triL,
                      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                      const real *pVertexArea, realNeq *state,
                      real *pVertexOperator,
                      real *pTriangleOperator)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    CalcOperatorEnergySingle(i, nVertex, nTriangle, pTv, G, triL,
                             pTn1, pTn2, pTn3,
                             pVertexArea, state,
                             pVertexOperator, pTriangleOperator);

    i += gridDim.x*blockDim.x;
  }
}

// #########################################################################
/*! \brief Kernel calculating estimated local truncation error from triangle-based and vertex-based operators.

Kernel calculating estimated local truncation error from triangle-based and vertex-based operators for the internal energy equation.

 \param nTriangle number of triangles in the current mesh.
 \param nVertex number of vertices in the current mesh.
 \param *pTv Pointer to triangle vertices
 \param *pVertexOperator pointer to vertex-based operator.
 \param *pTriangleOperator pointer to triangle-based operator.
 \param *pErrorEstimate array with estimated local truncation error (output).*/
// #########################################################################

__global__ void
devCalcErrorEstimate(int nTriangle, int nVertex, const int3 *pTv,
                     real *pVertexOperator,
                     real *pTriangleOperator,
                     real *pErrorEstimate)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    CalcErrorEstimateSingle(i, nVertex, pTv,
                            pVertexOperator, pTriangleOperator,
                            pErrorEstimate);

    i += gridDim.x*blockDim.x;
  }
}

//##############################################################################
/*! Calculate estimate of discretization error.

\param *vertexState Pointer to state vector (i.e. density, momentum, etc) at vertices
\param G Ratio of specific heats*/
//##############################################################################

template<class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalcErrorEstimate()
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  // Triangle vertex indices
  const int3 *pTv = mesh->TriangleVerticesData();

  // Inward pointing edge normals
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);

  // Edge lengths
  const real3 *pTl = mesh->TriangleEdgeLengthData();

  // Voronoi cell area
  const real *pVertexArea = mesh->VertexAreaData();

  // State at vertices
  realNeq *state = vertexState->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  // Truncation error estimate
  triangleErrorEstimate->SetSize(nTriangle);
  triangleErrorEstimate->SetToValue((real) 0.0);
  real *pErrorEstimate = triangleErrorEstimate->GetPointer();

  // Vertex-based operator
  Array<real> *vertexOperator =
    new Array<real>(1, cudaFlag, (unsigned int) nVertex);
  vertexOperator->SetToValue((real) 0.0);
  real *pVertexOperator = vertexOperator->GetPointer();

  // Triangle-based operator
  Array<real> *triangleOperator =
    new Array<real>(1, cudaFlag, (unsigned int) nTriangle);
  real *pTriangleOperator = triangleOperator->GetPointer();

  // Calculate operators: vertex-based and triangle-based
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcOperatorEnergy<realNeq, CL>,
                                       (size_t) 0, 0);

    devCalcOperatorEnergy<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, nTriangle, pTv, G, pTl,
       pTn1, pTn2, pTn3,
       pVertexArea, state,
       pVertexOperator, pTriangleOperator);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcOperatorEnergySingle(i, nVertex, nTriangle, pTv, G, pTl,
                               pTn1, pTn2, pTn3,
                               pVertexArea, state,
                               pVertexOperator, pTriangleOperator);
  }

  // Estimate truncation error by comparing with triangle-based operator
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcErrorEstimate,
                                       (size_t) 0, 0);

    devCalcErrorEstimate<<<nBlocks, nThreads>>>
      (nTriangle, nVertex, pTv,
       pVertexOperator, pTriangleOperator,
       pErrorEstimate);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcErrorEstimateSingle(i, nVertex, pTv,
                              pVertexOperator, pTriangleOperator,
                              pErrorEstimate);
  }

  delete vertexOperator;
  delete triangleOperator;
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::CalcErrorEstimate();
template void Simulation<real, CL_BURGERS>::CalcErrorEstimate();
template void Simulation<real3, CL_CART_ISO>::CalcErrorEstimate();
template void Simulation<real4, CL_CART_EULER>::CalcErrorEstimate();

}  // namespace astrix
