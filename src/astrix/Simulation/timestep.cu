// -*-c++-*-
/*! \file timestep.cu
\brief File containing function to calculate the maximum possible time step to take.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/atomic.h"
#include "../Common/cudaLow.h"
#include "../Common/profile.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Find maximum signal speed for triangle t

\param t Triangle to consider
\param a First vertex of triangle
\param b Second vertex of triangle
\param c Third vertex of triangle
\param *pState Pointer to vertex state vector
\param *pTl Pointer to triangle edge lengths
\param G Ratio of specific heats
\param G1 G - 1
\param *pVp Pointer to external potential at vertices
\param *pVc Pointer to vertex coordinates
\param *pVcs Sound speed at vertices (isothermal case)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
real FindMaxSignalSpeed(int t, int a, int b, int c,
                        real4 *pState, const real3* __restrict__ pTl,
                        const real G, const real G1, const real *pVp,
                        const real2 *pVc,
                        const real *pVcs, const real frameAngularVelocity)
{
  real zero = (real) 0.0;
  real half = (real) 0.5;
  real one = (real) 1.0;

  real vmax = zero;

  // First vertex
  real dens = pState[a].x;
  real momx = pState[a].y;
  real momy = pState[a].z;
  real ener = pState[a].w;

  real id = one/dens;
  real u = momx;
  real v = momy;
  real absv = sqrt(u*u+v*v)*id;

  // Pressure
  real p = G1*(ener - half*id*(u*u + v*v) - dens*pVp[a]);
#ifndef __CUDA_ARCH__
  if (p < zero)
    std::cout << "Negative pressure in timestep calculation!" << std::endl;
#endif

  // Sound speed
  real cs = sqrt(G*p*id);

  // Maximum signal speed
  vmax = absv + cs;

  // Second vertex
  dens = pState[b].x;
  momx = pState[b].y;
  momy = pState[b].z;
  ener = pState[b].w;

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v)*id;

  p = G1*(ener - half*id*(u*u + v*v) - dens*pVp[b]);
#ifndef __CUDA_ARCH__
  if (p < zero)
    std::cout << "Negative pressure in timestep calculation!" << std::endl;
#endif
  cs = sqrt(G*p*id);

  vmax = max(vmax, absv + cs);

  // Third vertex
  dens = pState[c].x;
  momx = pState[c].y;
  momy = pState[c].z;
  ener = pState[c].w;

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v)*id;

  p = G1*(ener - half*id*(u*u + v*v) - dens*pVp[c]);
#ifndef __CUDA_ARCH__
  if (p < zero)
    std::cout << "Negative pressure in timestep calculation!" << std::endl;
#endif
  cs = sqrt(G*p*id);

  vmax = max(vmax, absv + cs);

  // Triangle edge lengths
  real tl1 = pTl[t].x;
  real tl2 = pTl[t].y;
  real tl3 = pTl[t].z;

  // Scale with maximum edge length
  vmax = vmax*max(tl1, max(tl2, tl3));

  return vmax;
}

template<ConservationLaw CL>
__host__ __device__
real FindMaxSignalSpeed(int t, int a, int b, int c,
                        real3 *pState, const real3* __restrict__ pTl,
                        const real G, const real G1, const real *pVp,
                        const real2 *pVc,
                        const real *pVcs, const real frameAngularVelocity)
{
  real zero = (real) 0.0;
  real one = (real) 1.0;

  real vmax = zero;

  // First vertex
  real dens = pState[a].x;
  real momx = pState[a].y;
  real momy = pState[a].z;

  real ir4 = one;
  if (CL == CL_CYL_ISO)
    ir4 = exp(-(real)4.0*pVc[a].x);

  real id = one/dens;
  real u = momx;
  real v = momy;
  real absv = sqrt(u*u+v*v*ir4)*id;

  // Sound speed
  real cs = pVcs[a];

  // Maximum signal speed
  vmax = absv + cs;

  // Second vertex
  dens = pState[b].x;
  momx = pState[b].y;
  momy = pState[b].z;

  if (CL == CL_CYL_ISO)
    ir4 = exp(-(real)4.0*pVc[b].x);

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v*ir4)*id;

  // Sound speed
  cs = pVcs[b];

  vmax = max(vmax, absv + cs);

  // Third vertex
  dens = pState[c].x;
  momx = pState[c].y;
  momy = pState[c].z;

  if (CL == CL_CYL_ISO)
    ir4 = exp(-(real)4.0*pVc[c].x);

  id = one/dens;
  u = momx;
  v = momy;
  absv = sqrt(u*u+v*v*ir4)*id;

  // Sound speed
  cs = pVcs[c];

  vmax = max(vmax, absv + cs);

  // Triangle edge lengths
  real tl1 = pTl[t].x;
  real tl2 = pTl[t].y;
  real tl3 = pTl[t].z;

  // Scale with maximum edge length
  vmax = vmax*max(tl1, max(tl2, tl3));

  return vmax;
}

template<ConservationLaw CL>
__host__ __device__
real FindMaxSignalSpeed(int t, int a, int b, int c,
                        real *pState, const real3* __restrict__ pTl,
                        const real G, const real G1, const real *pVp,
                        const real2 *pVc,
                        const real *pVcs, const real frameAngularVelocity)
{
  // Triangle edge lengths
  real tl1 = pTl[t].x;
  real tl2 = pTl[t].y;
  real tl3 = pTl[t].z;

  if (CL == CL_BURGERS) {
    real vmax = max(fabs(pState[a]), max(fabs(pState[b]), fabs(pState[c])));
    return vmax*max(tl1, max(tl2, tl3));
  } else {
    // Scalar advection with velocity unity
    return 1.0*max(tl1, max(tl2, tl3));
  }
}

//######################################################################
/*! \brief Find maximum signal speed for triangle t and add it atomically to all of its vertices

\param t Triangle to consider
\param *pTv Pointer to vertices of triangle
\param *pState Pointer to vertex state vector
\param *pTl Pointer to triangle edge lengths
\param *pVts Pointer to maximum time step at vertex (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param *pVp Pointer to external potential at vertices
\param *pVc Pointer to vertex coordinates
\param *pVcs Sound speed at vertices (isothermal case)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__host__ __device__
void CalcVmaxSingle(int t, const int3* __restrict__ pTv, realNeq *pState,
                    const real3* __restrict__ pTl, real *pVts,
                    int nVertex, const real G, const real G1, const real *pVp,
                    const real2 *pVc, const real *pVcs,
                    const real frameAngularVelocity)
{
  int a = pTv[t].x;
  int b = pTv[t].y;
  int c = pTv[t].z;
  while (a >= nVertex) a -= nVertex;
  while (b >= nVertex) b -= nVertex;
  while (c >= nVertex) c -= nVertex;
  while (a < 0) a += nVertex;
  while (b < 0) b += nVertex;
  while (c < 0) c += nVertex;

  real vMax = FindMaxSignalSpeed<CL>(t, a, b, c, pState, pTl, G, G1, pVp,
                                     pVc, pVcs, frameAngularVelocity);

  AtomicAdd(&pVts[a], vMax);
  AtomicAdd(&pVts[b], vMax);
  AtomicAdd(&pVts[c], vMax);
}

//######################################################################
/*! \brief Kernel finding maximum signal speed for triangles and add it atomically to all of the vertices

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle
\param *tv2 Pointer to second vertex of triangle
\param *tv3 Pointer to third vertex of triangle
\param *dens Pointer to density
\param *momx Pointer to x momentum
\param *momy Pointer to y momentum
\param *ener Pointer to energy
\param *pTriangleEdgeLengths Pointer to triangle edge lengths
\param *pVts Pointer to maximum time step at vertex (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param *pVp Pointer to external potential at vertices
\param *pVc Pointer to vertex coordinates
\param *pVcs Sound speed at vertices (isothermal case)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcVmax(const int nTriangle, const int3* __restrict__ pTv, realNeq *pState,
            const real3* __restrict__ pTl, real *pVts,
            const int nVertex, const real G, const real G1, const real *pVp,
            const real2 *pVc,
            const real *pVcs, const real frameAngularVelocity)
{
  // n = triangle number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcVmaxSingle<realNeq, CL>(n, pTv, pState, pTl, pVts, nVertex, G, G1, pVp,
                                pVc, pVcs, frameAngularVelocity);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! \brief Calculate maximum allowed time step for vertex \a n

\param n Vertex to consider
\param *pVts Pointer to array containing sum of maximum signal speeds for all triangles sharing \a n
\param *pVarea Pointer to array of areas assosiated with vertices (Voronoi cells)*/
//######################################################################

__host__ __device__
void CalcVertexTimeStepSingle(int n, real *pVts, const real *pVarea)
{
  real two = (real) 2.0;
  pVts[n] = two*pVarea[n]/pVts[n];
}

//######################################################################
/*! \brief Kernel calculating maximum allowed time step for vertices

\param nVertex Total number of vertices in Mesh
\param *pVts Pointer to array containing sum of maximum signal speeds for all triangles sharing vertex
\param *pVarea Pointer to array of areas assosiated with vertices (Voronoi cells)*/
//######################################################################

__global__ void
devCalcVertexTimeStep(const int nVertex, real *pVts, const real *pVarea)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    CalcVertexTimeStepSingle(n, pVts, pVarea);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate maximum possible time step.*/
//######################################################################

template <class realNeq, ConservationLaw CL>
real Simulation<realNeq, CL>::CalcVertexTimeStep()
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  const unsigned int nVertex = mesh->GetNVertex();
  const int nTriangle = mesh->GetNTriangle();

  realNeq *pState = vertexState->GetPointer();
  const real *pVp = vertexPotential->GetPointer();
  const real *pVcs = vertexSoundSpeed->GetPointer();
  const real G = simulationParameter->specificHeatRatio;

  const int3 *pTv = mesh->TriangleVerticesData();
  const real3 *pTl = mesh->TriangleEdgeLengthData();
  const real *pVarea = mesh->VertexAreaData();
  const real2 *pVc = mesh->VertexCoordinatesData();

  const real frameAngularVelocity = 0.0;

  Array<real> *vertexTimestep = new Array<real>(1, cudaFlag, nVertex);
  vertexTimestep->SetToValue(0.0);
  real *pVts = vertexTimestep->GetPointer();

  // First calculate maximum signal speed
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcVmax<realNeq, CL>,
                                       (size_t) 0, 0);

    // Execute kernel...
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcVmax<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, pTv, pState, pTl, pVts, nVertex, G, G - 1.0, pVp, pVc, pVcs,
       frameAngularVelocity);
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
    for (int n = 0; n < nTriangle; n++)
      CalcVmaxSingle<realNeq, CL>(n, pTv, pState, pTl, pVts,
                                  nVertex, G, G - 1.0, pVp, pVc, pVcs,
                                  frameAngularVelocity);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("SignalSpeed.prof2", nTriangle, elapsedTime, cudaFlag);
#endif

  // Convert maximum signal speed into vertex time step
  if (cudaFlag == 1) {
    int nThreads = 128;
    int nBlocks  = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcVertexTimeStep,
                                       (size_t) 0, 0);

    // Execute kernel...
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcVertexTimeStep<<<nBlocks, nThreads>>>
      (nVertex, pVts, pVarea);
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
    for (unsigned int n = 0; n < nVertex; n++)
      CalcVertexTimeStepSingle(n, pVts, pVarea);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("CalcTimeStep.prof2", nVertex, elapsedTime, cudaFlag);
#endif

  // Find the minimum
  real dt = simulationParameter->CFLnumber*vertexTimestep->Minimum();

  // End exactly on maxSimulationTime
  if (simulationTime + dt > simulationParameter->maxSimulationTime)
    dt = simulationParameter->maxSimulationTime - simulationTime;

  delete vertexTimestep;

  return dt;
}

//##############################################################################
// Instantiate
//##############################################################################

template real Simulation<real, CL_ADVECT>::CalcVertexTimeStep();
template real Simulation<real, CL_BURGERS>::CalcVertexTimeStep();
template real Simulation<real3, CL_CART_ISO>::CalcVertexTimeStep();
template real Simulation<real3, CL_CYL_ISO>::CalcVertexTimeStep();
template real Simulation<real4, CL_CART_EULER>::CalcVertexTimeStep();

}  // namespace astrix
