// -*-c++-*-
/*! \file totalresLDA.cu
\brief File containing functions for calculating space-time LDA residue

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <fstream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./upwind.h"
#include "../Common/profile.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! \brief Calculate space-time LDA residue at triangle n

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresTot Triangle total residue
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices
\param *pTrad Triangle average x (needed for cylindrical geometry)
\param *pVcs Sound speed at vertices (isothermal case)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void CalcTotalResLDASingle(int n,
                           const int3* __restrict__ pTv,
                           const real4* __restrict__ pVz,
                           real4 *pTresLDA0, real4 *pTresLDA1,
                           real4 *pTresLDA2, real4 *pTresTot,
                           const real2 *pTn1,
                           const real2 *pTn2,
                           const real2 *pTn3,
                           const real3* __restrict__ pTl,
                           int nVertex, real G, real G1, real G2,
                           const real *pVp,
                           const real *pTrad, const real *pVcs,
                           const real frameAngularVelocity)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  int vs1 = pTv[n].x;
  int vs2 = pTv[n].y;
  int vs3 = pTv[n].z;
  while (vs1 >= nVertex) vs1 -= nVertex;
  while (vs2 >= nVertex) vs2 -= nVertex;
  while (vs3 >= nVertex) vs3 -= nVertex;
  while (vs1 < 0) vs1 += nVertex;
  while (vs2 < 0) vs2 += nVertex;
  while (vs3 < 0) vs3 += nVertex;

  // External potential at vertices
  real pot0 = pVp[vs1];
  real pot1 = pVp[vs2];
  real pot2 = pVp[vs3];
  real pot = (pot0 + pot1 + pot2)*onethird;

  // Parameter vector at vertices
  real Zv00 = pVz[vs1].x;
  real Zv01 = pVz[vs1].y;
  real Zv02 = pVz[vs1].z;
  real Zv03 = pVz[vs1].w;
  real Zv10 = pVz[vs2].x;
  real Zv11 = pVz[vs2].y;
  real Zv12 = pVz[vs2].z;
  real Zv13 = pVz[vs2].w;
  real Zv20 = pVz[vs3].x;
  real Zv21 = pVz[vs3].y;
  real Zv22 = pVz[vs3].z;
  real Zv23 = pVz[vs3].w;

  // Average parameter vector
  real Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  real Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  real Z2 = (Zv02 + Zv12 + Zv22)*onethird;
  real Z3 = (Zv03 + Zv13 + Zv23)*onethird;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real alpha = G1*half*(utilde*utilde + vtilde*vtilde) - G1*pot;

  real htilde = Z3/Z0;
  real ctilde = sqrt(G1*(htilde - 2.0*pot) - alpha);
  real ic = one/ctilde;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real hc = htilde*ic;
  real uc = utilde*ic;
  real vc = vtilde*ic;
  real ac = alpha*ic;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;

  // First direction
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;
  real wtilde = (uc*nx + vc*ny)*ctilde;

  real l1 = min(zero, wtilde + ctilde);
  real l2 = min(zero, wtilde - ctilde);
  real l3 = min(zero, wtilde);

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);

  real nm00 = tl*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  real nm01 = tl*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  real nm02 = tl*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  real nm03 = tl*eulerKMP03(G1, ic, l1l2l3);
  real nm10 = tl*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  real nm11 = tl*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  real nm12 = tl*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  real nm13 = tl*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  real nm20 = tl*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  real nm21 = tl*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  real nm22 = tl*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  real nm23 = tl*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  real nm30 = tl*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  real nm31 = tl*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  real nm32 = tl*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  real nm33 = tl*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);

  // Second direction
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;

  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  nm00 += tl*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  nm01 += tl*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  nm02 += tl*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  nm03 += tl*eulerKMP03(G1, ic, l1l2l3);
  nm10 += tl*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  nm13 += tl*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  nm20 += tl*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  nm22 += tl*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  nm23 += tl*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  nm30 += tl*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  nm31 += tl*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  nm32 += tl*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  nm33 += tl*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);

  // Third direction
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  nm00 += tl*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  nm01 += tl*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  nm02 += tl*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  nm03 += tl*eulerKMP03(G1, ic, l1l2l3);
  nm10 += tl*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  nm13 += tl*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  nm20 += tl*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  nm22 += tl*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  nm23 += tl*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  nm30 += tl*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  nm31 += tl*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  nm32 += tl*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  nm33 += tl*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate inverse of NM = Sum(K-)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real d1 = nm03*nm12 - nm02*nm13;
  real d2 = nm03*nm11 - nm01*nm13;
  real d3 = nm02*nm11 - nm01*nm12;
  real d4 = nm03*nm10 - nm00*nm13;
  real d5 = nm02*nm10 - nm00*nm12;
  real d6 = nm01*nm10 - nm00*nm11;

  real f1 = (nm21*nm30 - nm20*nm31);
  real f2 = (nm20*nm32 - nm22*nm30);
  real f3 = (nm23*nm30 - nm20*nm33);
  real f4 = (nm22*nm31 - nm21*nm32);
  real f5 = (nm21*nm33 - nm23*nm31);
  real f6 = (nm23*nm32 - nm22*nm33);

  real det = d1*f1 + d2*f2 + d3*f3 + d4*f4 + d5*f5 + d6*f6;

  real invN02 = -d1*nm31 + d2*nm32 - d3*nm33;
  real invN12 = d1*nm30 - d4*nm32 + d5*nm33;
  real invN22 = -d2*nm30 + d4*nm31 - d6*nm33;
  real invN32 = d3*nm30 - d5*nm31 + d6*nm32;
  real invN03 = d1*nm21 - d2*nm22 + d3*nm23;
  real invN13 = -d1*nm20 + d4*nm22 - d5*nm23;
  real invN23 = d2*nm20 - d4*nm21 + d6*nm23;
  real invN33 = -d3*nm20 + d5*nm21 - d6*nm22;

  real invN00 = -nm11*f6 - nm12*f5 - nm13*f4;
  real invN01 = nm01*f6 + nm02*f5 + nm03*f4;
  real invN10 = nm10*f6 - nm12*f3 - nm13*f2;
  real invN11 = -nm00*f6 + nm02*f3 + nm03*f2;
  real invN20 = nm10*f5 + nm11*f3 - nm13*f1;
  real invN21 = -nm00*f5 - nm01*f3 + nm03*f1;
  real invN30 = nm10*f4 + nm11*f2 + nm12*f1;
  real invN31 = -nm00*f4 - nm01*f2 - nm02*f1;

  real Wtilde0 =
    invN00*pTresTot[n].x +
    invN01*pTresTot[n].y +
    invN02*pTresTot[n].z +
    invN03*pTresTot[n].w;
  real Wtilde1 =
    invN10*pTresTot[n].x +
    invN11*pTresTot[n].y +
    invN12*pTresTot[n].z +
    invN13*pTresTot[n].w;
  real Wtilde2 =
    invN20*pTresTot[n].x +
    invN21*pTresTot[n].y +
    invN22*pTresTot[n].z +
    invN23*pTresTot[n].w;
  real Wtilde3 =
    invN30*pTresTot[n].x +
    invN31*pTresTot[n].y +
    invN32*pTresTot[n].z +
    invN33*pTresTot[n].w;

  if (det != zero) det = one/det;

  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;
  Wtilde3 *= det;

  real ResLDA;

  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3)
    -Wtilde1*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP03(G1, ic, l1l2l3);
  pTresLDA0[n].x = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA0[n].y = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA0[n].z = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA0[n].w = ResLDA;

  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  // Second direction
  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3)
    -Wtilde1*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP03(G1, ic, l1l2l3);
  pTresLDA1[n].x = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA1[n].y = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA1[n].z = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA1[n].w = ResLDA;

  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  // Third direction
  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3)
    -Wtilde1*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP03(G1, ic, l1l2l3);
  pTresLDA2[n].x = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA2[n].y = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA2[n].z = ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA2[n].w = ResLDA;
}




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//-----------------------------------------------------------------------------
// Systems of 3 equations: Cartesian and cylindrical isothermal hydrodynamics
//-----------------------------------------------------------------------------
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



template<ConservationLaw CL>
__host__ __device__
void CalcTotalResLDASingle(int n,
                           const int3* __restrict__ pTv,
                           const real3* __restrict__ pVz,
                           real3 *pTresLDA0, real3 *pTresLDA1,
                           real3 *pTresLDA2, real3 *pTresTot,
                           const real2 *pTn1,
                           const real2 *pTn2,
                           const real2 *pTn3,
                           const real3* __restrict__ pTl,
                           int nVertex, real G, real G1, real G2,
                           const real *pVp,
                           const real *pTrad, const real *pVcs,
                           const real frameAngularVelocity)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  int vs1 = pTv[n].x;
  int vs2 = pTv[n].y;
  int vs3 = pTv[n].z;
  while (vs1 >= nVertex) vs1 -= nVertex;
  while (vs2 >= nVertex) vs2 -= nVertex;
  while (vs3 >= nVertex) vs3 -= nVertex;
  while (vs1 < 0) vs1 += nVertex;
  while (vs2 < 0) vs2 += nVertex;
  while (vs3 < 0) vs3 += nVertex;

  // Average sound speed
  real ctilde = onethird*(pVcs[vs1] + pVcs[vs2] + pVcs[vs3]);

  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv00 = pVz[vs1].x;
  real Zv01 = pVz[vs1].y;
  real Zv02 = pVz[vs1].z;
  real Zv10 = pVz[vs2].x;
  real Zv11 = pVz[vs2].y;
  real Zv12 = pVz[vs2].z;
  real Zv20 = pVz[vs3].x;
  real Zv21 = pVz[vs3].y;
  real Zv22 = pVz[vs3].z;

  // Average parameter vector
  real Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  real Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  real Z2 = (Zv02 + Zv12 + Zv22)*onethird;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;

  real ic = one/ctilde;

  // Average radius triangle
  real r2 = one;
  real ir2 = one;
  real Omega = zero;

  if (CL == CL_CYL_ISO) {
    r2 = pTrad[n];
    r2 = r2*r2;
    ir2 = one/r2;
    Omega = frameAngularVelocity;
  }

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real uc = utilde*ic;
  real vc = vtilde*ic;

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;

  // First direction
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;
  real wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  real l1 = min(zero, wtilde + ctilde - Omega*ny);
  real l2 = min(zero, wtilde - ctilde - Omega*ny);
  real l3 = min(zero, wtilde - Omega*ny);

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);

  real nm00 = tl*isoKMP00(ic, wtilde, l1l2, l2);
  real nm01 = tl*isoKMP01(nx, ic, l1l2);
  real nm02 = tl*isoKMP02(ny*ir2, ic, l1l2);
  real nm10 = tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  real nm11 = tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  real nm12 = tl*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  real nm20 = tl*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  real nm21 = tl*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  real nm22 = tl*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);

  // Second direction
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;

  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = min(wtilde + ctilde - Omega*ny, zero);
  l2 = min(wtilde - ctilde - Omega*ny, zero);
  l3 = min(wtilde - Omega*ny, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  nm00 += tl*isoKMP00(ic, wtilde, l1l2, l2);
  nm01 += tl*isoKMP01(nx, ic, l1l2);
  nm02 += tl*isoKMP02(ny*ir2, ic, l1l2);
  nm10 += tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  nm20 += tl*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  nm22 += tl*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);

  // Third direction
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = min(wtilde + ctilde - Omega*ny, zero);
  l2 = min(wtilde - ctilde - Omega*ny, zero);
  l3 = min(wtilde - Omega*ny, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  nm00 += tl*isoKMP00(ic, wtilde, l1l2, l2);
  nm01 += tl*isoKMP01(nx, ic, l1l2);
  nm02 += tl*isoKMP02(ny*ir2, ic, l1l2);
  nm10 += tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  nm20 += tl*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  nm22 += tl*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate inverse of NM = Sum(K-)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real invN00 = nm11*nm22 - nm12*nm21;
  real invN01 = nm02*nm21 - nm01*nm22;
  real invN02 = nm01*nm12 - nm02*nm11;
  real invN10 = nm12*nm20 - nm10*nm22;
  real invN11 = nm00*nm22 - nm02*nm20;
  real invN12 = nm02*nm10 - nm00*nm12;
  real invN20 = nm10*nm21 - nm11*nm20;
  real invN21 = nm01*nm20 - nm00*nm21;
  real invN22 = nm00*nm11 - nm01*nm10;

  real det = nm00*invN00 + nm01*invN10 + nm02*invN20;

  real Wtilde0 =
    invN00*pTresTot[n].x +
    invN01*pTresTot[n].y +
    invN02*pTresTot[n].z;
  real Wtilde1 =
    invN10*pTresTot[n].x +
    invN11*pTresTot[n].y +
    invN12*pTresTot[n].z;
  real Wtilde2 =
    invN20*pTresTot[n].x +
    invN21*pTresTot[n].y +
    invN22*pTresTot[n].z;

  if (det != zero) det = one/det;

  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;

  real ResLDA;

  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = half*(wtilde + ctilde - Omega*ny +
             fabs(wtilde + ctilde - Omega*ny));
  l2 = half*(wtilde - ctilde - Omega*ny +
             fabs(wtilde - ctilde - Omega*ny));
  l3 = half*(wtilde - Omega*ny +
             fabs(wtilde - Omega*ny));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny*ir2, ic, l1l2);
  pTresLDA0[n].x = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  pTresLDA0[n].y = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  pTresLDA0[n].z = ResLDA;

  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = half*(wtilde + ctilde - Omega*ny +
             fabs(wtilde + ctilde - Omega*ny));
  l2 = half*(wtilde - ctilde - Omega*ny +
             fabs(wtilde - ctilde - Omega*ny));
  l3 = half*(wtilde - Omega*ny +
             fabs(wtilde - Omega*ny));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny*ir2, ic, l1l2);
  pTresLDA1[n].x = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  pTresLDA1[n].y = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  pTresLDA1[n].z = ResLDA;

  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = half*(wtilde + ctilde - Omega*ny +
             fabs(wtilde + ctilde - Omega*ny));
  l2 = half*(wtilde - ctilde - Omega*ny +
             fabs(wtilde - ctilde - Omega*ny));
  l3 = half*(wtilde - Omega*ny +
             fabs(wtilde - Omega*ny));

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResLDA =
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny*ir2, ic, l1l2);
  pTresLDA2[n].x = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  pTresLDA2[n].y = ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  pTresLDA2[n].z = ResLDA;
}




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//-----------------------------------------------------------------------------
// Systems of 1 equation: Linear advection and Burgers equation
//-----------------------------------------------------------------------------
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



template<ConservationLaw CL>
__host__ __device__
void CalcTotalResLDASingle(int n,
                           const int3* __restrict__ pTv,
                           const real* __restrict__ pVz,
                           real *pTresLDA0, real *pTresLDA1,
                           real *pTresLDA2, real *pTresTot,
                           const real2 *pTn1,
                           const real2 *pTn2,
                           const real2 *pTn3,
                           const real3* __restrict__  pTl,
                           int nVertex, real G, real G1, real G2,
                           const real *pVp,
                           const real *pTrad, const real *pVcs,
                           const real frameAngularVelocity)
{
  const real zero  = (real) 0.0;
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Average parameter vector
  real vx = one;
  real vy = zero;
  if (CL == CL_BURGERS) {
    const real onethird = (real) (1.0/3.0);

    real Zv0 = pVz[v1];
    real Zv1 = pVz[v2];
    real Zv2 = pVz[v3];

    real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
    vx = Z0;
    vy = Z0;
  }

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // First direction
  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;
  real nx = half*tl1*tnx1;
  real ny = half*tl1*tny1;

  real l1 = min(zero, vx*nx + vy*ny);
  real nm = l1;

  // Second direction
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;
  nx = half*tl2*tnx2;
  ny = half*tl2*tny2;

  l1 = min(zero, vx*nx + vy*ny);
  nm += l1;

  // Third direction
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;
  nx = half*tl3*tnx3;
  ny = half*tl3*tny3;

  l1 = min(zero, vx*nx + vy*ny);
  nm += l1;

  real invN = one;
  if (nm != zero) invN /= nm;

  real Wtilde = invN*pTresTot[n];

  // First direction
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA0[n] = -l1*Wtilde;

  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA1[n] = -l1*Wtilde;

  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA2[n] = -l1*Wtilde;
}

//######################################################################
/*! \brief Kernel calculating space-time LDA residue for all triangles

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresTot Triangle total residue
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices
\param *pTrad Triangle average x (needed for cylindrical geometry)
\param *pVcs Sound speed at vertices (isothermal case)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcTotalResLDA(int nTriangle, const int3* __restrict__ pTv,
                   const realNeq* __restrict__ pVz,
                   realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
                   realNeq *pTresTot, const real2 *pTn1, const real2 *pTn2,
                   const real2 *pTn3,
                   const real3* __restrict__ pTl,
                   int nVertex, real G, real G1, real G2,  const real *pVp,
                   const real *pTrad, const real *pVcs,
                   const real frameAngularVelocity)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcTotalResLDASingle<CL>(n, pTv, pVz, pTresLDA0, pTresLDA1, pTresLDA2,
                              pTresTot, pTn1, pTn2, pTn3, pTl,
                              nVertex, G, G1, G2, pVp,
                              pTrad, pVcs, frameAngularVelocity);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate space-time LDA residue for all triangles; result in \a
triangleResidueLDA.*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalcTotalResLDA()
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif

  int transformFlag = 0;
  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexParameterVector->TransformToDevice();

      triangleResidueTotal->TransformToDevice();
      triangleResidueLDA->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexParameterVector->TransformToHost();

      triangleResidueTotal->TransformToHost();
      triangleResidueLDA->TransformToHost();

      cudaFlag = 0;
    }
  }

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pVz = vertexParameterVector->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real *pVcs = vertexSoundSpeed->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);

  realNeq *pTresTot = triangleResidueTotal->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);
  const real3 *pTl  = mesh->TriangleEdgeLengthData();

  const real *pTrad = mesh->TriangleAverageXData();
  real frameAngularVelocity = 0.0;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcTotalResLDA<realNeq, CL>,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcTotalResLDA<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVz,
       pTresLDA0, pTresLDA1, pTresLDA2, pTresTot,
       pTn1, pTn2, pTn3, pTl, nVertex, G, G - 1.0, G - 2.0, pVp,
       pTrad, pVcs, frameAngularVelocity);
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
      CalcTotalResLDASingle<CL>(n, pTv, pVz,
                                pTresLDA0, pTresLDA1, pTresLDA2, pTresTot,
                                pTn1, pTn2, pTn3, pTl, nVertex,
                                G, G - 1.0, G - 2.0, pVp,
                                pTrad, pVcs, frameAngularVelocity);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("CalcTotalResLDA.prof2", nTriangle, elapsedTime, cudaFlag);
#endif

  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexParameterVector->TransformToDevice();

      triangleResidueTotal->TransformToDevice();
      triangleResidueLDA->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexParameterVector->TransformToHost();

      triangleResidueTotal->TransformToHost();
      triangleResidueLDA->TransformToHost();

      cudaFlag = 0;
    }
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::CalcTotalResLDA();
template void Simulation<real, CL_BURGERS>::CalcTotalResLDA();
template void Simulation<real3, CL_CART_ISO>::CalcTotalResLDA();
template void Simulation<real3, CL_CYL_ISO>::CalcTotalResLDA();
template void Simulation<real4, CL_CART_EULER>::CalcTotalResLDA();

}  // namespace astrix
