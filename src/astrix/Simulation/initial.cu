// -*-c++-*-
/*! \file initial.cu
\brief Functions to set initial conditions

\section LICENSE
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
#include "./Param/simulationparameter.h"

namespace astrix {

__host__ __device__
real funcF(real t)
{
  if (t <= 0.0f) return (real) 0.0;
  return exp(-(real) 1.0/(t + 1.0e-10));
}

__host__ __device__
real funcBump(real t)
{
  return funcF(t)/(funcF(t) + funcF((real) 1.0 - t));
}

//##############################################################################
/*! \brief Set initial conditions at vertex \a n

\param n Vertex to consider
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVpot Pointer to gravitational potential at vertices
\param *pState Pointer to state vector (output)
\param G Ratio of specific heats*/
//##############################################################################

__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
                      real *pVpot, real4 *state, real G, real time, real Px)
{
  // const real onethird = (real) (1.0/3.0);
  const real zero = (real) 0.0;
  const real half = (real) 0.5;
  const real one = (real) 1.0;
  const real two = (real) 2.0;
  const real five = (real) 5.0;

  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = zero;
  real momx = zero;
  real momy = zero;
  real ener = zero;

  if (problemDef == PROBLEM_LINEAR) {
    real amp = (real) 1.0e-4;
    real k = two*M_PI;
    real c0 = one;
    real p0 = c0*c0/G;

    dens = one;
    momx = two;
    momy = (real) 1.0e-10;

    momx += amp*cos(k*vertX);
    // momy += amp*cos(k*vertY);

    ener = half*(Sq(momx) + Sq(momy))/dens + p0/(G - one);
  }

  if (problemDef == PROBLEM_VORTEX) {
    real x = vertX;
    real y = vertY;
    real vx = 0.0;
    real vy = 0.0;

    // Temperature
    real temp = 1.0;

    for (int i = -1; i < 2; i++) {
      real xc = five + time + i*Px;
      real yc = five;

      real beta = five;

      real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

      temp -= (G - one)*Sq(beta)*exp(one - r*r)/(8.0f*G*M_PI*M_PI);

      vx -= half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
      vy += half*(x - xc)*beta*exp(half - half*r*r)/M_PI;
    }

    dens = std::pow(temp, (real)(one/(G - one)));
    real pres = dens*temp;

    momx = (real)1.0e-10 + dens*vx;
    momy = (real)2.0e-10 + dens*vy;
    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
  }

  if (problemDef == PROBLEM_CYL) {
    dens = 1.0;
    momx = 2.0;
    momy = 1.0e-10;
    ener = half*(Sq(momx) + Sq(momy))/dens + 2.5f/(G - one);
  }

  if (problemDef == PROBLEM_KH) {
    real a = 4.0f;
    real smoothHat =
      funcBump(a*(vertY - 0.25f) + 0.5f)*funcBump(-a*(vertY - 0.75f) + 0.5f);

    dens = 1.0f + smoothHat;
    momx = dens*(-0.5f + smoothHat);
    momy = 0.0f;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 2.5f/(G - 1.0f);
  }

  if (problemDef == PROBLEM_SOD) {
    if (vertX < 0.5f) {
      dens = 1.0f;
      momx = 1.0e-10f;
      momy = 1.0e-10f;
      ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 1.0f/(G - 1.0f);
    } else {
      dens = 0.125f;
      momx = 1.2e-10f;
      momy = 2.0e-10f;
      ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 0.1f/(G - 1.0f);
    }
  }

  if (problemDef == PROBLEM_NOH) {
    real x = vertX;
    real y = vertY;
    real r = sqrt(x*x + y*y);

    real pres = 1.0e-6;

    if (r < time/3.0) {
      dens = 16.0;
      momx = zero;
      momy = zero;
      pres = 16.0/3.0;
    } else {
      dens = one + time/(r + 1.0e-10);
      momx = dens*(-x/(r + 1.0e-10) + 1.0e-10);
      momy = dens*(-y/(r + 1.0e-10) - 2.0e-10);
    }

    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
  }

  if (problemDef == PROBLEM_BLAST) {
    // Interacting blast waves
    real p = 0.01f;
    if (vertX < 0.1f) p = 1000.0f;
    if (vertX > 0.9f) p = 100.0f;

    dens = 1.0f;
    momx = 1.0e-10f;
    momy = 1.0e-10f;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + p/(G - 1.0f);
  }

  if (problemDef == PROBLEM_RIEMANN) {
    real f = 1.0f - (vertX > 0.8f);
    real g = 1.0f - (vertY > 0.8f);

    // CASE 3
    real densRB = 1.5f;
    real momxRB = 1.0e-10f;
    real momyRB = 1.0e-10f;
    real enerRB = 0.5f*(Sq(momxRB) + Sq(momyRB))/densRB + 1.5f/(G - 1.0f);

    real densLB = 0.5322581f;
    real momxLB = 1.2060454f*densLB;
    real momyLB = 0.0f;
    real enerLB = 0.5f*(Sq(momxLB) + Sq(momyLB))/densLB + 0.3f/(G - 1.0f);

    real densRO = 0.5322581f;
    real momxRO = 0.0f;
    real momyRO = 1.2060454f*densRO;
    real enerRO = 0.5f*(Sq(momxRO) + Sq(momyRO))/densRO + 0.3f/(G - 1.0f);

    real densLO = 0.1379928f;
    real momxLO = 1.2060454f*densLO;
    real momyLO = 1.2060454f*densLO;
    real enerLO = 0.5f*(Sq(momxLO) + Sq(momyLO))/densLO + 0.0290323f/(G - 1.0f);

    /*
    // CASE 6
    real densRB = 1.0f;
    real momxRB = 0.75f*densRB;
    real momyRB = -0.5f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 2.0f;
    real momxLB = 0.75f*densLB;
    real momyLB = 0.5f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 1.0f/(G-1.0f);

    real densRO = 3.0f;
    real momxRO = -0.75f*densRO;
    real momyRO = -0.5f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 1.0f/(G-1.0f);

    real densLO = 1.0f;
    real momxLO = -0.75f*densLO;
    real momyLO = 0.5f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 1.0f/(G-1.0f);
    */
    /*
    // CASE 15
    real densRB = 1.0f;
    real momxRB = 0.1f*densRB;
    real momyRB = -0.3f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 0.5197f;
    real momxLB = -0.6259f*densLB;
    real momyLB = -0.3f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 0.4f/(G-1.0f);

    real densRO = 0.5313f;
    real momxRO = 0.1f*densRO;
    real momyRO = 0.4276f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 0.4f/(G-1.0f);

    real densLO = 0.8f;
    real momxLO = 0.1f*densLO;
    real momyLO = -0.3f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 0.4f/(G-1.0f);
    */
    /*
    // Case 17
    real densRB = 1.0f;
    real momxRB = 1.0e-10f*densRB;
    real momyRB = -0.4f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 2.0f;
    real momxLB = 1.0e-10f*densLB;
    real momyLB = -0.3f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 1.0f/(G-1.0f);

    real densRO = 0.5197f;
    real momxRO = 1.0e-10f*densRO;
    real momyRO = -1.1259f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 0.4f/(G-1.0f);

    real densLO = 1.0625f;
    real momxLO = 1.0e-10f*densLO;
    real momyLO = 0.2145f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 0.4f/(G-1.0f);
    */

    dens =
      densRB*(1.0f - f)*(1.0f - g) + densLB*f*(1.0f - g) +
      densRO*(1.0f - f)*g + densLO*f*g;
    momx =
      momxRB*(1.0f - f)*(1.0f - g) + momxLB*f*(1.0f - g) +
      momxRO*(1.0f - f)*g + momxLO*f*g;
    momy =
      momyRB*(1.0f - f)*(1.0f - g) + momyLB*f*(1.0f - g) +
      momyRO*(1.0f - f)*g + momyLO*f*g;
    ener =
      enerRB*(1.0f - f)*(1.0f - g) + enerLB*f*(1.0f - g) +
      enerRO*(1.0f - f)*g + enerLO*f*g;
  }

  state[n].x = dens;
  state[n].y = momx;
  state[n].z = momy;
  state[n].w = ener;
}

__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
                      real *pVpot, real *state, real G, real time, real Px)
{
  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = (real) 1.0;

  if (problemDef == PROBLEM_LINEAR) {
    for (int i = -1; i < 2; i++) {
      real x = vertX;
      real xc = time + i*Px;

      if (fabs(x - xc) <= (real) 0.25) dens += Sq(cos(2.0*M_PI*(x - xc)));
    }
  }

  if (problemDef == PROBLEM_VORTEX) {
#if BURGERS == 1
    real x = vertX;
    real y = vertY;

    real u0 = 1.0;
    real u1 = 0.0;
    real u2 = 0.0;
    real amp = 0.001;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        // Start center location
        real xStart = -1.0 + (real)i;
        real yStart = -1.0 + (real)j;

        // Current centre location
        real cx = xStart + time;
        real cy = yStart + time;

        real r = sqrt(Sq(x - cx) + Sq(y - cy));

        if (r < 0.25) {
          u1 += amp*cos(2.0*M_PI*r)*cos(2.0*M_PI*r);
          u2 += amp*amp*4.0*M_PI*time*(x + y)*cos(2.0*M_PI*r)*
            cos(2.0*M_PI*r)*cos(2.0*M_PI*r)*sin(2.0*M_PI*r)/(r + 1.0e-10);
        }
      }
    }

    dens = u0 + u1 + u2;
#else
    real half = (real) 0.5;

    for (int i = -1; i < 2; i++) {
      real x = vertX;
      real y = vertY;

      real xc = half + i*Px + time;
      real yc = half;

      real r = sqrt(Sq(x - xc) + Sq(y - yc));

      if (r <= (real) 0.25) dens += Sq(cos(2.0*M_PI*r));
    }
#endif
  }

  if (problemDef == PROBLEM_RIEMANN) {
    real x = vertX;
    real y = vertY;

    dens = (real) 1.0e-10;
    if (x  > (real) -0.6 && x < (real) - 0.1 &&
        y > (real) -0.35 && y < (real) 0.15) dens += 1.0;
  }

  if (problemDef == PROBLEM_SOURCE) {
    //real x = vertX;

    dens = 1.0;// + exp(-1000.0*x*x);
  }

  state[n] = dens;
}

//######################################################################
/*! \brief Kernel setting initial conditions

\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVpot Pointer to gravitational potential at vertices
\param *pState Pointer to state vector (output)
\param G Ratio of specific heats*/
//######################################################################

__global__ void
devSetInitial(int nVertex, const real2 *pVc, ProblemDefinition problemDef,
              real *pVertexPotential, realNeq *state,
              real G, real time, real Px)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    SetInitialSingle(n, pVc, problemDef, pVertexPotential, state,
                     G, time, Px);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Set initial conditions for all vertices based on problemSpecification*/
//######################################################################

void Simulation::SetInitial(real time)
{
  int nVertex = mesh->GetNVertex();

  realNeq *state = vertexState->GetPointer();
  real *pVertexPotential = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();

  real Px = mesh->GetPx();
  real G = simulationParameter->specificHeatRatio;
  ProblemDefinition p = simulationParameter->problemDef;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetInitial,
                                       (size_t) 0, 0);

    devSetInitial<<<nBlocks, nThreads>>>
      (nVertex, pVc, p, pVertexPotential, state, G, time, Px);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nVertex; n++)
      SetInitialSingle(n, pVc, p, pVertexPotential, state, G, time, Px);
  }

  try {
    // Add KH eigenvector
    if (p == PROBLEM_KH)
      KHAddEigenVector();
  }
  catch (...) {
    std::cout << "Warning: reading KH eigenvector file failed!" << std::endl;
    std::cout << "Running simulation without adding KH eigenvector!"
              << std::endl;
  }
}

}  // namespace astrix
