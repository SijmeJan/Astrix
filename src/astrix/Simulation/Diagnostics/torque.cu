// -*-c++-*-
/*! \file torque.cu
\brief File containing function to calculate the torque on an embedded planet.

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
#include "../../Common/cudaLow.h"
#include "../../Mesh/mesh.h"
#include "../../Mesh/triangleLow.h"
#include "./diagnostics.h"

namespace astrix {

//##############################################################################
//! General form of torque calculation
//##############################################################################

template<class T, ConservationLaw CL>
__host__ __device__
void CalcTriangleTorqueSingle(int n,
                              int nVertex,
                              const int3 *pTv,
                              const real2 *pVc,
                              const real2 meshPeriod,
                              T *pState,
                              real *pTorque)
{
  pTorque[n] = 0.0;
}

//! Cylindrical isothermal
template<>
__host__ __device__
void CalcTriangleTorqueSingle<real3, CL_CYL_ISO>(int n,
                                                 int nVertex,
                                                 const int3 *pTv,
                                                 const real2 *pVc,
                                                 const real2 meshPeriod,
                                                 real3 *pState,
                                                 real *pTorque)
{
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;

  real Px = meshPeriod.x;
  real Py = meshPeriod.y;

  real x1, x2, x3, y1, y2, y3;
  GetTriangleCoordinates(pVc,
                         v1, v2, v3,
                         nVertex, Px, Py,
                         x1, x2, x3,
                         y1, y2, y3);

  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Gaussian quadrature on triangle
  const int N = 37;
  real s[N] = {0.333333333333333333333333333333,
               0.950275662924105565450352089520,
               0.024862168537947217274823955239,
               0.024862168537947217274823955239,
               0.171614914923835347556304795551,
               0.414192542538082326221847602214,
               0.414192542538082326221847602214,
               0.539412243677190440263092985511,
               0.230293878161404779868453507244,
               0.230293878161404779868453507244,
               0.772160036676532561750285570113,
               0.113919981661733719124857214943,
               0.113919981661733719124857214943,
               0.009085399949835353883572964740,
               0.495457300025082323058213517632,
               0.495457300025082323058213517632,
               0.062277290305886993497083640527,
               0.468861354847056503251458179727,
               0.468861354847056503251458179727,
               0.022076289653624405142446876931,
               0.022076289653624405142446876931,
               0.851306504174348550389457672223,
               0.851306504174348550389457672223,
               0.126617206172027096933163647918,
               0.126617206172027096933163647918,
               0.018620522802520968955913511549,
               0.018620522802520968955913511549,
               0.689441970728591295496647976487,
               0.689441970728591295496647976487,
               0.291937506468887771754472382212,
               0.291937506468887771754472382212,
               0.096506481292159228736516560903,
               0.096506481292159228736516560903,
               0.635867859433872768286976979827,
               0.635867859433872768286976979827,
               0.267625659273967961282458816185,
               0.267625659273967961282458816185};
  real t[N] = {0.333333333333333333333333333333,
               0.024862168537947217274823955239,
               0.950275662924105565450352089520,
               0.024862168537947217274823955239,
               0.414192542538082326221847602214,
               0.171614914923835347556304795551,
               0.414192542538082326221847602214,
               0.230293878161404779868453507244,
               0.539412243677190440263092985511,
               0.230293878161404779868453507244,
               0.113919981661733719124857214943,
               0.772160036676532561750285570113,
               0.113919981661733719124857214943,
               0.495457300025082323058213517632,
               0.009085399949835353883572964740,
               0.495457300025082323058213517632,
               0.468861354847056503251458179727,
               0.062277290305886993497083640527,
               0.468861354847056503251458179727,
               0.851306504174348550389457672223,
               0.126617206172027096933163647918,
               0.022076289653624405142446876931,
               0.126617206172027096933163647918,
               0.022076289653624405142446876931,
               0.851306504174348550389457672223,
               0.689441970728591295496647976487,
               0.291937506468887771754472382212,
               0.018620522802520968955913511549,
               0.291937506468887771754472382212,
               0.018620522802520968955913511549,
               0.689441970728591295496647976487,
               0.635867859433872768286976979827,
               0.267625659273967961282458816185,
               0.096506481292159228736516560903,
               0.267625659273967961282458816185,
               0.096506481292159228736516560903,
               0.635867859433872768286976979827};
  real w[N] = {0.051739766065744133555179145422,
               0.008007799555564801597804123460,
               0.008007799555564801597804123460,
               0.008007799555564801597804123460,
               0.046868898981821644823226732071,
               0.046868898981821644823226732071,
               0.046868898981821644823226732071,
               0.046590940183976487960361770070,
               0.046590940183976487960361770070,
               0.046590940183976487960361770070,
               0.031016943313796381407646220131,
               0.031016943313796381407646220131,
               0.031016943313796381407646220131,
               0.010791612736631273623178240136,
               0.010791612736631273623178240136,
               0.010791612736631273623178240136,
               0.032195534242431618819414482205,
               0.032195534242431618819414482205,
               0.032195534242431618819414482205,
               0.015445834210701583817692900053,
               0.015445834210701583817692900053,
               0.015445834210701583817692900053,
               0.015445834210701583817692900053,
               0.015445834210701583817692900053,
               0.015445834210701583817692900053,
               0.017822989923178661888748319485,
               0.017822989923178661888748319485,
               0.017822989923178661888748319485,
               0.017822989923178661888748319485,
               0.017822989923178661888748319485,
               0.017822989923178661888748319485,
               0.037038683681384627918546472190,
               0.037038683681384627918546472190,
               0.037038683681384627918546472190,
               0.037038683681384627918546472190,
               0.037038683681384627918546472190,
               0.037038683681384627918546472190};


  real f = 0.0;

  for (int i = 0; i < N; i++) {
    // x_i, y_i
    real x = x1 + (x2 - x1)*s[i] + (x3 - x1)*t[i];
    real y = y1 + (y2 - y1)*s[i] + (y3 - y1)*t[i];

    real dens = pState[v1].x +
      (pState[v2].x - pState[v1].x)*s[i] +
      (pState[v3].x - pState[v1].x)*t[i];

    // Cylindrical radius
    real r = exp(x);

    real q = 1.0e-5;
    real eps = 0.4*0.05;

    real dpotdy = q*r*sin(y - M_PI)*
      pow(r*r + 1.0 - 2.0*r*cos(y - M_PI) + eps*eps, -1.5);

    f = f + w[i]*dens*dpotdy;
  }

  real area = 0.5*abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1));

  pTorque[n] = f*area;
}

//######################################################################
//! Kernel: Calculate torque at triangles
//######################################################################

template<class T, ConservationLaw CL>
__global__ void
devCalcTriangleTorque(int nTriangle,
                      int nVertex,
                      const int3 *pTv,
                      const real2 *pVc,
                      const real2 meshPeriod,
                      T *pState,
                      real *pTorque)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcTriangleTorqueSingle<T, CL>(n, nVertex, pTv, pVc,
                                    meshPeriod, pState, pTorque);

    n += blockDim.x*gridDim.x;
  }
}

//#########################################################################
/*! Compute total torque on planet.

\param *state Pointer to state vector
\param *mesh Pointer to Mesh object*/
//#########################################################################

template <class T, ConservationLaw CL>
real Diagnostics<T, CL>::Torque(Array<T> *state, Mesh *mesh)
{
  int cudaFlag = state->GetCudaFlag();
  int nVertex = mesh->GetNVertex();

  T *pState = state->GetPointer();

  const real2 *pVc = mesh->VertexCoordinatesData();
  int nTriangle = mesh->GetNTriangle();
  const int3 *pTv = mesh->TriangleVerticesData();

  real2 meshPeriod;
  meshPeriod.x = mesh->GetPx();
  meshPeriod.y = mesh->GetPy();

  Array<real> *torque = new Array<real>(1, cudaFlag, nTriangle);
  real *pTorque = torque->GetPointer();

  // Calculate torque at triangles
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcTriangleTorque<T, CL>,
                                       (size_t) 0, 0);

    devCalcTriangleTorque<T, CL><<<nBlocks, nThreads>>>
      (nTriangle, nVertex, pTv, pVc, meshPeriod, pState, pTorque);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 0; i < nTriangle; i++)
      CalcTriangleTorqueSingle<T, CL>(i, nVertex, pTv, pVc,
                                      meshPeriod, pState, pTorque);
  }

  real total_torque = torque->Sum();

  delete torque;

  return total_torque;
}


//###################################################
// Instantiate
//###################################################

template
real Diagnostics<real, CL_ADVECT>::Torque(Array<real> *state, Mesh *mesh);
template
real Diagnostics<real, CL_BURGERS>::Torque(Array<real> *state, Mesh *mesh);
template
real Diagnostics<real3, CL_CART_ISO>::Torque(Array<real3> *state, Mesh *mesh);
template
real Diagnostics<real3, CL_CYL_ISO>::Torque(Array<real3> *state, Mesh *mesh);
template
real Diagnostics<real4, CL_CART_EULER>::Torque(Array<real4> *state, Mesh *mesh);

}  // namespace astrix
