// -*-c++-*-
/*! \file source.cu
\brief File containing function to calculate source term contribution to residual.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "../Common/helper_math.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Calculating source term at vertex n.

\param n Vertex to consider
\param problemDef Problem definition
\param *pVc Pointer tovertex coordinates
\param *pVp Pointer to external potential at vertices
\param *pState Pointer to state vector
\param *pSource Pointer to source vector (output)*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void CalcVertexSourceSingle(int n, ProblemDefinition problemDef,
                            const real2 *pVc, const real *pVp,
                            const real4 *pState, real4 *pSource)
{
  pSource[n].x = 0.0;
  pSource[n].y = 0.0;
  pSource[n].z = 0.0;
  pSource[n].w = 0.0;

  if (problemDef == PROBLEM_DISC) {
    real x = pVc[n].x;
    real y = pVc[n].y;

    real r = exp(x);
    pSource[n].y =
      (Sq(pState[n].z)/Sq(Sq(r)) - Sq(pState[n].y))/pState[n].x -
      pState[n].x/Cb(r);
  }

}

//! Version for three equations
template<ConservationLaw CL>
__host__ __device__
void CalcVertexSourceSingle(int n, ProblemDefinition problemDef, const real2 *pVc,
                            const real *pVp, const real3 *pState, real3 *pSource)
{
  pSource[n].x = 0.0;
  pSource[n].y = 0.0;
  pSource[n].z = 0.0;

  if (problemDef == PROBLEM_DISC) {
    real x = pVc[n].x;
    real y = pVc[n].y;

    // Cylindrical radius
    real r = exp(x);
    // Centrifugal, gravity, geometrical
    pSource[n].y =
      (0.0*Sq(pState[n].z)/Sq(Sq(r)) - Sq(pState[n].y))/pState[n].x -
      0.0*pState[n].x/Cb(r);
  }

}

//! Version for single equation
template<ConservationLaw CL>
__host__ __device__
void CalcVertexSourceSingle(int n, ProblemDefinition problemDef, const real2 *pVc,
                            const real *pVp, const real *pState, real *pSource)
{
  pSource[n] = 0.0;

  if (problemDef == PROBLEM_SOURCE) {
    if (CL == CL_ADVECT) {
      pSource[n] = Sq(pState[n]);
      //pSource[n] = 8.0*M_PI*cos(8.0*M_PI*pVc[n].x);

      //real N = 10.0;
      //real x = pVc[n].x;
      //pSource[n] += 0.0*(N*cos(n*x)*(1.0 + x) - sin(N*x))/Sq(1.0 + x) -
      //  0.0*sin(N*x)/(1.0 + x);
    }

    if (CL == CL_BURGERS)
      pSource[n] = Sq(pState[n]);

  }
}

//######################################################################
/*! \brief Kernel calculating source term at vertices.

\param nVertex Total number of vertices in Mesh
\param problemDef Problem definition
\param *pVc Pointer to vertex coordinates
\param *pVp Pointer to external potential at vertices
\param *pState Pointer to state vector
\param *pSource Pointer to source vector (output)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcVertexSource(int nVertex, ProblemDefinition problemDef, const real2 *pVc,
                    const real *pVp, const realNeq *pState, realNeq *pSource)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    CalcVertexSourceSingle<CL>(n, problemDef, pVc, pVp, pState, pSource);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Calculating source term contribution to residual for single triangle.

\param n Triangle to consider
\param problemDef Problem definition
\param nVertex Total number of vertices in Mesh
\param *pTv Pointer to triangle vertices
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pVp Pointer to external potential at vertices
\param *pState Pointer to state vector
\param *pSource Pointer to source vector (output)
\param *pVs Pointer to vertex source term*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void CalcSourceSingle(int n, ProblemDefinition problemDef,
                      int nVertex, const int3 *pTv,
                      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                      const real3 *pTl, const real *pVp,
                      const real4 *pState, real4 *pSource, const real4 *pVs)
{
  pSource[n].x = 0.0;
  pSource[n].y = 0.0;
  pSource[n].z = 0.0;
  pSource[n].w = 0.0;

  if (problemDef == PROBLEM_SOURCE) {
    real three = (real) 3.0;
    real half = (real) 0.5;

    // Vertices belonging to triangle
    int v1 = pTv[n].x;
    int v2 = pTv[n].y;
    int v3 = pTv[n].z;
    while (v1 >= nVertex) v1 -= nVertex;
    while (v2 >= nVertex) v2 -= nVertex;
    while (v3 >= nVertex) v3 -= nVertex;
    while (v1 < 0) v1 += nVertex;
    while (v2 < 0) v2 += nVertex;
    while (v3 < 0) v3 += nVertex;

    real tl1 = pTl[n].x;
    real tl2 = pTl[n].y;
    real tl3 = pTl[n].z;

    real d1 = pState[v1].x;
    real d2 = pState[v2].x;
    real d3 = pState[v3].x;
    real dG = (d1 + d2 + d3)/three;

    /*
    real m1 = pState[v1].y;
    real m2 = pState[v2].y;
    real m3 = pState[v3].y;
    real mG = (m1 + m2 + m3)/three;

    real n1 = pState[v1].z;
    real n2 = pState[v2].z;
    real n3 = pState[v3].z;
    real nG = (n1 + n2 + n3)/three;
    */

    real dpotdx = half*
      (pVp[v1]*pTn1[n].x*tl1 +
       pVp[v2]*pTn2[n].x*tl2 +
       pVp[v3]*pTn3[n].x*tl3);
    real dpotdy = half*
      (pVp[v1]*pTn1[n].y*tl1 +
       pVp[v2]*pTn2[n].y*tl2 +
       pVp[v3]*pTn3[n].y*tl3);

    pSource[n].x = 0.0;
    pSource[n].y = dG*dpotdx;
    pSource[n].z = dG*dpotdy;
    pSource[n].w = 0.0;//mG*dpotdx + nG*dpotdy;
  }

}

//! Version for three equations
template<ConservationLaw CL>
__host__ __device__
void CalcSourceSingle(int n, ProblemDefinition problemDef,
                      int nVertex, const int3 *pTv,
                      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                      const real3 *pTl, const real *pVp,
                      const real3 *pState, real3 *pSource, const real3 *pVs)
{
  pSource[n].x = 0.0;
  pSource[n].y = 0.0;
  pSource[n].z = 0.0;

  if (problemDef == PROBLEM_SOURCE) {
    real three = (real) 3.0;
    real half = (real) 0.5;

    // Vertices belonging to triangle
    int v1 = pTv[n].x;
    int v2 = pTv[n].y;
    int v3 = pTv[n].z;
    while (v1 >= nVertex) v1 -= nVertex;
    while (v2 >= nVertex) v2 -= nVertex;
    while (v3 >= nVertex) v3 -= nVertex;
    while (v1 < 0) v1 += nVertex;
    while (v2 < 0) v2 += nVertex;
    while (v3 < 0) v3 += nVertex;

    real tl1 = pTl[n].x;
    real tl2 = pTl[n].y;
    real tl3 = pTl[n].z;

    real d1 = pState[v1].x;
    real d2 = pState[v2].x;
    real d3 = pState[v3].x;
    real dG = (d1 + d2 + d3)/three;

    real dpotdx = half*
      (pVp[v1]*pTn1[n].x*tl1 +
       pVp[v2]*pTn2[n].x*tl2 +
       pVp[v3]*pTn3[n].x*tl3);
    real dpotdy = half*
      (pVp[v1]*pTn1[n].y*tl1 +
       pVp[v2]*pTn2[n].y*tl2 +
       pVp[v3]*pTn3[n].y*tl3);

    pSource[n].x = 0.0;
    pSource[n].y = dG*dpotdx;
    pSource[n].z = dG*dpotdy;
  }

  if (problemDef == PROBLEM_DISC) {
    real three = (real) 3.0;
    real half = (real) 0.5;

    // Vertices belonging to triangle
    int v1 = pTv[n].x;
    int v2 = pTv[n].y;
    int v3 = pTv[n].z;
    while (v1 >= nVertex) v1 -= nVertex;
    while (v2 >= nVertex) v2 -= nVertex;
    while (v3 >= nVertex) v3 -= nVertex;
    while (v1 < 0) v1 += nVertex;
    while (v2 < 0) v2 += nVertex;
    while (v3 < 0) v3 += nVertex;

    real tl1 = pTl[n].x;
    real tl2 = pTl[n].y;
    real tl3 = pTl[n].z;

    // Average source term
    real3 S = (pVs[v1] + pVs[v2] + pVs[v3])/three;

    // Triangle area
    real s = half*(tl1 + tl2 + tl3);
    real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

    // Source contribution to residual (note minus sign!)
    pSource[n] = -S*area;
  }

}

//! Version for single equation
template<ConservationLaw CL>
__host__ __device__
void CalcSourceSingle(int n, ProblemDefinition problemDef,
                      int nVertex, const int3 *pTv,
                      const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                      const real3 *pTl, const real *pVp,
                      const real *pState, real *pSource, const real *pVs)
{
  pSource[n] = 0.0;

  if (problemDef == PROBLEM_SOURCE) {
    real three = (real) 3.0;

    // Vertices belonging to triangle
    int v1 = pTv[n].x;
    int v2 = pTv[n].y;
    int v3 = pTv[n].z;
    while (v1 >= nVertex) v1 -= nVertex;
    while (v2 >= nVertex) v2 -= nVertex;
    while (v3 >= nVertex) v3 -= nVertex;
    while (v1 < 0) v1 += nVertex;
    while (v2 < 0) v2 += nVertex;
    while (v3 < 0) v3 += nVertex;

    real tl1 = pTl[n].x;
    real tl2 = pTl[n].y;
    real tl3 = pTl[n].z;

    real s = (real) 0.5*(tl1 + tl2 + tl3);
    real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

    if (CL == CL_ADVECT) {

      real d1 = pVs[v1];
      real d2 = pVs[v2];
      real d3 = pVs[v3];
      real dG = (d1 + d2 + d3)/three;

      pSource[n] = -dG*area;

      /*
      real f1 = pState[v1];
      real f2 = pState[v2];
      real f3 = pState[v3];

      real dG =
        (real) 0.5*(f1*f1 + f1*f2 + f1*f3 + f2*f2 + f2*f3 + f3*f3)/three;
      pSource[n] = -dG*area;
      */
    }

    if (CL == CL_BURGERS) {
      real d1 = pVs[v1];
      real d2 = pVs[v2];
      real d3 = pVs[v3];
      real dG = (d1 + d2 + d3)/three;

      pSource[n] = -dG*area;

      /*
      real f1 = pState[v1];
      real f2 = pState[v2];
      real f3 = pState[v3];

      //real dG =
      //  (real) 0.5*(f1*f1 + f1*f2 + f1*f3 + f2*f2 + f2*f3 + f3*f3)/three;
      real dG = (f1 + f2 + f3)/three;
      pSource[n] = -dG*area;
      */
    }

  }
}

//######################################################################
/*! \brief Kernel calculating source term contribution to residual.

\param nTriangle Total number of triangles in Mesh
\param problemDef Problem definition
\param nVertex Total number of vertices in Mesh
\param *pTv Pointer to triangle vertices
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pVp Pointer to external potential at vertices
\param *pState Pointer to state vector
\param *pSource Pointer to source vector (output)
\param *pVs Pointer to vertex source term */
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcSource(int nTriangle, ProblemDefinition problemDef,
              int nVertex, const int3 *pTv,
              const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
              const real3 *pTl, const real *pVp,
              const realNeq *pState, realNeq *pSource, const realNeq *pVs)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcSourceSingle<CL>(n, problemDef, nVertex,
                         pTv, pTn1, pTn2, pTn3,
                         pTl, pVp, pState, pSource, pVs);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Calculate source contribution to residual. Result will be in \a triangleResidueSource.

\param state State vector to base source term calculation on. */
//#########################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalcSource(Array<realNeq> *state)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  ProblemDefinition problemDef = simulationParameter->problemDef;

  const int3 *pTv = mesh->TriangleVerticesData();
  const real *pVp = vertexPotential->GetPointer();
  const realNeq *pState = state->GetPointer();
  realNeq *pVs = vertexSource->GetPointer();
  realNeq *pSource = triangleResidueSource->GetPointer();

  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);

  const real3 *pTl = mesh->TriangleEdgeLengthData();
  const real2 *pVc = mesh->VertexCoordinatesData();

  // First calculate sources at vertices
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcVertexSource<realNeq, CL>,
                                       (size_t) 0, 0);

    devCalcVertexSource<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, problemDef, pVc, pVp, pState, pVs);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nVertex; i++)
      CalcVertexSourceSingle<CL>(i, problemDef, pVc, pVp, pState, pVs);
  }

  // Average over triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcSource<realNeq, CL>,
                                       (size_t) 0, 0);

    devCalcSource<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, problemDef, nVertex,
       pTv, pTn1, pTn2, pTn3, pTl, pVp, pState, pSource, pVs);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcSourceSingle<CL>(i, problemDef, nVertex,
                           pTv, pTn1, pTn2, pTn3, pTl,
                           pVp, pState, pSource, pVs);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::CalcSource(Array<real> *state);
template void Simulation<real, CL_BURGERS>::CalcSource(Array<real> *state);
template void Simulation<real3, CL_CART_ISO>::CalcSource(Array<real3> *state);
template void Simulation<real3, CL_CYL_ISO>::CalcSource(Array<real3> *state);
template void Simulation<real4, CL_CART_EULER>::CalcSource(Array<real4> *state);

}  // namespace astrix
