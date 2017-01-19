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
#include "./mesh.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"
#include "./Connectivity/connectivity.h"

namespace astrix {

//######################################################################
/*! \brief For a single triangle, calculate triangle-based operator and its contribution to the vertex-based operator.

For a single triangle, calculate triangle-based operator (for the internal energy equation) and its contribution to the vertex-based operator.

 * @param i triangle to be dealt with.
 * @param nVertex number of vertices in the current mesh.
 * @param nTriangle number of triangles in the current mesh.
 * @param *tv1 pointer to array of first vertices belonging to triangles, i.e. \a tv1[i] is vertex 1 of triangle \a i.
 * @param *tv2 pointer to array of second vertices belonging to triangles, i.e. \a tv2[i] is vertex 2 of triangle \a i.
 * @param *tv3 pointer to array of third vertices belonging to triangles, i.e. \a tv3[i] is vertex 3 of triangle \a i.
 * @param G specific heat ratio
 * @param *triL pointer to triangle edge lengths.
 * @param *triNx pointer to triangle normals (x component).
 * @param *triNy pointer to triangle normals (y component).
 * @param *pVertexArea pointer to cell volume (Voronoi area)
 * @param *dens pointer to density at vertices
 * @param *momx pointer to x-momentum at vertices
 * @param *momy pointer to y-momentum at vertices
 * @param *ener pointer to total energy density at vertices
 * @param *pVertexOperator pointer to vertex-based operator (output).
 * @param *pTriangleOperator pointer to triangle-based operator (output).*/
//######################################################################

__host__ __device__
void CalcOperatorEnergySingle(int i, int nVertex, int nTriangle,
                              int3 *pTv, real G, real3 *triL,
                              real2 *pTn1, real2 *pTn2, real2 *pTn3,
                              real *pVertexArea, real4 *state,
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
                              int3 *pTv, real G, real3 *triL,
                              real2 *pTn1, real2 *pTn2, real2 *pTn3,
                              real *pVertexArea, real *state,
                              real *pVertexOperator,
                              real *pTriangleOperator)
{
  const real sixth = (real) (1.0/6.0);
  //const real tenth = (real) 0.1;
  //const real third = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  //const real one   = (real) 1.0;

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
  //real ny1 = pTn1[i].y;
  //real ny2 = pTn2[i].y;
  //real ny3 = pTn3[i].y;

  // State at vertex a
  real d1 = state[a];
  // State at vertex b
  real d2 = state[b];
  // State at vertex c
  real d3 = state[c];

  // Cell centred internal energy gradient
  real gradUx = d1*nx1*tl1 + d2*nx2*tl2 + d3*nx3*tl3;

  // Cell-averaged state
  //real dc = third*(d1 + d2 + d3);

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
 * @param i triangle to be dealt with.
 * @param nVertex number of vertices in the current mesh.
 * @param *tv1 pointer to array of first vertices belonging to triangles, i.e. \a tv1[i] is vertex 1 of triangle \a i.
 * @param *tv2 pointer to array of second vertices belonging to triangles, i.e. \a tv2[i] is vertex 2 of triangle \a i.
 * @param *tv3 pointer to array of third vertices belonging to triangles, i.e. \a tv3[i] is vertex 3 of triangle \a i.
 * @param *pVertexOperator pointer to vertex-based operator.
 * @param *pTriangleOperator pointer to triangle-based operator.
 * @param *pErrorEstimate array with estimated local truncation error (output).*/
//##############################################################################

__host__ __device__
void CalcErrorEstimateSingle(int i, int nVertex, int3 *pTv,
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

 * @param nVertex number of vertices in the current mesh.
 * @param nTriangle number of triangles in the current mesh.
 * @param *tv1 pointer to array of first vertices belonging to triangles, i.e. \a tv1[i] is vertex 1 of triangle \a i.
 * @param *tv2 pointer to array of second vertices belonging to triangles, i.e. \a tv2[i] is vertex 2 of triangle \a i.
 * @param *tv3 pointer to array of third vertices belonging to triangles, i.e. \a tv3[i] is vertex 3 of triangle \a i.
 * @param G specific heat ratio
 * @param *triL pointer to triangle edge lengths.
 * @param *triNx pointer to triangle normals (x component).
 * @param *triNy pointer to triangle normals (y component).
 * @param *pVertexArea pointer to cell volume (Voronoi area)
 * @param *dens pointer to density at vertices
 * @param *momx pointer to x-momentum at vertices
 * @param *momy pointer to y-momentum at vertices
 * @param *ener pointer to total energy density at vertices
 * @param *pVertexOperator pointer to vertex-based operator (output).
 * @param *pTriangleOperator pointer to triangle-based operator (output).*/
// #########################################################################

__global__ void
devCalcOperatorEnergy(int nVertex, int nTriangle,
                      int3 *pTv, real G, real3 *triL,
                      //real *triNx, real *triNy,
                      real2 *pTn1, real2 *pTn2, real2 *pTn3,
                      real *pVertexArea, realNeq *state,
                      //real *dens, real *momx,
                      //real *momy, real *ener,
                      real *pVertexOperator,
                      real *pTriangleOperator)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  while (i < nTriangle) {
    CalcOperatorEnergySingle(i, nVertex, nTriangle, pTv, G, triL,
                             //triNx, triNy,
                             pTn1, pTn2, pTn3,
                             pVertexArea, state,
                             //dens, momx, momy, ener,
                             pVertexOperator, pTriangleOperator);

    i += gridDim.x*blockDim.x;
  }
}

// #########################################################################
/*! \brief Kernel calculating estimated local truncation error from triangle-based and vertex-based operators.

Kernel calculating estimated local truncation error from triangle-based and vertex-based operators for the internal energy equation.

 \param nTriangle number of triangles in the current mesh.
 \param nVertex number of vertices in the current mesh.
 \param *tv1 pointer to array of first vertices belonging to triangles, i.e. \a tv1[i] is vertex 1 of triangle \a i.
 \param *tv2 pointer to array of second vertices belonging to triangles, i.e. \a tv2[i] is vertex 2 of triangle \a i.
 \param *tv3 pointer to array of third vertices belonging to triangles, i.e. \a tv3[i] is vertex 3 of triangle \a i.
 \param *pVertexOperator pointer to vertex-based operator.
 \param *pTriangleOperator pointer to triangle-based operator.
 \param *pErrorEstimate array with estimated local truncation error (output).*/
// #########################################################################

__global__ void
devCalcErrorEstimate(int nTriangle, int nVertex, int3 *pTv,
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

void Mesh::CalcErrorEstimate(Array<realNeq> *vertexState, real G)
{
  const real zero = (real) 0.0;

  int nTriangle = connectivity->triangleVertices->GetSize();
  int nVertex = connectivity->vertexCoordinates->GetSize();

  // Triangle vertex indices
  int3 *pTv = connectivity->triangleVertices->GetPointer();

  // Inward pointing edge normals
  real2 *pTn1 = triangleEdgeNormals->GetPointer(0);
  real2 *pTn2 = triangleEdgeNormals->GetPointer(1);
  real2 *pTn3 = triangleEdgeNormals->GetPointer(2);


  // Edge lengths
  real3 *triL = triangleEdgeLength->GetPointer();

  // Voronoi cell area
  real *pVertexArea = vertexArea->GetPointer();

  // State at vertices
  realNeq *state = vertexState->GetPointer();

  // Truncation error estimate
  triangleErrorEstimate->SetSize(nTriangle);
  triangleErrorEstimate->SetToValue(zero);
  real *pErrorEstimate = triangleErrorEstimate->GetPointer();

  // Vertex-based operator
  Array<real> *vertexOperator =
    new Array<real>(1, cudaFlag, (unsigned int) nVertex);
  vertexOperator->SetToValue(zero);
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
                                       devCalcOperatorEnergy,
                                       (size_t) 0, 0);

    devCalcOperatorEnergy<<<nBlocks, nThreads>>>
      (nVertex, nTriangle, pTv, G, triL,
       //triNx, triNy,
       pTn1, pTn2, pTn3,
       pVertexArea, state,
       //dens, momx, momy, ener,
       pVertexOperator, pTriangleOperator);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcOperatorEnergySingle(i, nVertex, nTriangle, pTv, G, triL,
                               //triNx, triNy,
                               pTn1, pTn2, pTn3,
                               pVertexArea, state,
                               //dens, momx, momy, ener,
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


}
