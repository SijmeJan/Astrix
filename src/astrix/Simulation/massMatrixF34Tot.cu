// -*-c++-*-
/*! \file massMatrixF34Tot.cu
\brief File containing functions for F3 and F4 mass matrix

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
#include "./Param/simulationparameter.h"
#include "./upwind.h"

namespace astrix {

//######################################################################
/*! \brief Calculate F3/F4 mass matrix contribution to residual

\param n Triangle to consider
\param dt Time step
\param massMatrix Mass matrix to use (should be 3 or 4)
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pDstate Pointer to state difference at vertices
\param *pTresTot Triangle total residue
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void MassMatrixF34TotSingle(int n, real dt, int massMatrix,
                            const int3* __restrict__ pTv,
                            const real4* __restrict__ pVz,
                            const real4* __restrict__ pDstate,
                            real4 *pTresTot, const real2 *pTn1,
                            const real2 *pTn2, const real2 *pTn3,
                            const real3 *pTl, int nVertex,
                            real G, real G1, real G2, real *pVp)
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

  // State differences
  real dW00 = pDstate[vs1].x;
  real dW01 = pDstate[vs1].y;
  real dW02 = pDstate[vs1].z;
  real dW03 = pDstate[vs1].w;
  real dW10 = pDstate[vs2].x;
  real dW11 = pDstate[vs2].y;
  real dW12 = pDstate[vs2].z;
  real dW13 = pDstate[vs2].w;
  real dW20 = pDstate[vs3].x;
  real dW21 = pDstate[vs3].y;
  real dW22 = pDstate[vs3].z;
  real dW23 = pDstate[vs3].w;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(tl1 + tl2 + tl3);
  real Adt = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3))*onethird/dt;
  if (massMatrix == 3) Adt = -Adt;

  dW00 = dW00*Adt;
  dW01 = dW01*Adt;
  dW02 = dW02*Adt;
  dW03 = dW03*Adt;
  dW10 = dW10*Adt;
  dW11 = dW11*Adt;
  dW12 = dW12*Adt;
  dW13 = dW13*Adt;
  dW20 = dW20*Adt;
  dW21 = dW21*Adt;
  dW22 = dW22*Adt;
  dW23 = dW23*Adt;

  // Parameter vector at vertices: 12 uncoalesced loads
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

  // Wtilde = N*dW0
  real Wtilde0 = invN00*dW00 + invN01*dW01 + invN02*dW02 + invN03*dW03;
  real Wtilde1 = invN10*dW00 + invN11*dW01 + invN12*dW02 + invN13*dW03;
  real Wtilde2 = invN20*dW00 + invN21*dW01 + invN22*dW02 + invN23*dW03;
  real Wtilde3 = invN30*dW00 + invN31*dW01 + invN32*dW02 + invN33*dW03;

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
  pTresTot[n].x += tl1*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl1*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresTot[n].z += tl1*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresTot[n].w += tl1*ResLDA;

  // Wtilde = N*dW1
  Wtilde0 = invN00*dW10 + invN01*dW11 + invN02*dW12 + invN03*dW13;
  Wtilde1 = invN10*dW10 + invN11*dW11 + invN12*dW12 + invN13*dW13;
  Wtilde2 = invN20*dW10 + invN21*dW11 + invN22*dW12 + invN23*dW13;
  Wtilde3 = invN30*dW10 + invN31*dW11 + invN32*dW12 + invN33*dW13;
  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;
  Wtilde3 *= det;

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
  pTresTot[n].x += tl2*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl2*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresTot[n].z += tl2*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresTot[n].w += tl2*ResLDA;

  Wtilde0 = invN00*dW20 + invN01*dW21 + invN02*dW22 + invN03*dW23;
  Wtilde1 = invN10*dW20 + invN11*dW21 + invN12*dW22 + invN13*dW23;
  Wtilde2 = invN20*dW20 + invN21*dW21 + invN22*dW22 + invN23*dW23;
  Wtilde3 = invN30*dW20 + invN31*dW21 + invN32*dW22 + invN33*dW23;
  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;
  Wtilde3 *= det;

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
  pTresTot[n].x += tl3*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl3*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresTot[n].z += tl3*ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresTot[n].w += tl3*ResLDA;
}

//! Version for three equations
template<ConservationLaw CL>
__host__ __device__
void MassMatrixF34TotSingle(int n, real dt, int massMatrix,
                            const int3* __restrict__ pTv,
                            const real3* __restrict__ pVz,
                            const real3* __restrict__ pDstate,
                            real3 *pTresTot, const real2 *pTn1,
                            const real2 *pTn2, const real2 *pTn3,
                            const real3 *pTl, int nVertex,
                            real G, real G1, real G2, real *pVp)
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

  // State differences
  real dW00 = pDstate[vs1].x;
  real dW01 = pDstate[vs1].y;
  real dW02 = pDstate[vs1].z;
  real dW10 = pDstate[vs2].x;
  real dW11 = pDstate[vs2].y;
  real dW12 = pDstate[vs2].z;
  real dW20 = pDstate[vs3].x;
  real dW21 = pDstate[vs3].y;
  real dW22 = pDstate[vs3].z;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(tl1 + tl2 + tl3);
  real Adt = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3))*onethird/dt;
  if (massMatrix == 3) Adt = -Adt;

  dW00 = dW00*Adt;
  dW01 = dW01*Adt;
  dW02 = dW02*Adt;
  dW10 = dW10*Adt;
  dW11 = dW11*Adt;
  dW12 = dW12*Adt;
  dW20 = dW20*Adt;
  dW21 = dW21*Adt;
  dW22 = dW22*Adt;

  //real resTot0 = (dW00 + dW10 + dW20)*Adt;
  //real resTot1 = (dW01 + dW11 + dW21)*Adt;
  //real resTot2 = (dW02 + dW12 + dW22)*Adt;

  // Parameter vector at vertices
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

  real ctilde = one;
  real ic = one/ctilde;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real uc = utilde*ic;
  real vc = vtilde*ic;

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

  real nm00 = tl*isoKMP00(ic, wtilde, l1l2, l2);
  real nm01 = tl*isoKMP01(nx, ic, l1l2);
  real nm02 = tl*isoKMP02(ny, ic, l1l2);
  real nm10 = tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  real nm11 = tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  real nm12 = tl*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  real nm20 = tl*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  real nm21 = tl*isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  real nm22 = tl*isoKMP22(ny, vc, l1l2l3, l1l2, l3);

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

  nm00 += tl*isoKMP00(ic, wtilde, l1l2, l2);
  nm01 += tl*isoKMP01(nx, ic, l1l2);
  nm02 += tl*isoKMP02(ny, ic, l1l2);
  nm10 += tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  nm20 += tl*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  nm22 += tl*isoKMP22(ny, vc, l1l2l3, l1l2, l3);

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

  nm00 += tl*isoKMP00(ic, wtilde, l1l2, l2);
  nm01 += tl*isoKMP01(nx, ic, l1l2);
  nm02 += tl*isoKMP02(ny, ic, l1l2);
  nm10 += tl*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  nm11 += tl*isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  nm12 += tl*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  nm20 += tl*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  nm21 += tl*isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  nm22 += tl*isoKMP22(ny, vc, l1l2l3, l1l2, l3);

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
  if (det != zero) det = one/det;

  // Wtilde = N*phi
  //real Wtilde0 = invN00*resTot0 + invN01*resTot1 + invN02*resTot2;
  //real Wtilde1 = invN10*resTot0 + invN11*resTot1 + invN12*resTot2;
  //real Wtilde2 = invN20*resTot0 + invN21*resTot1 + invN22*resTot2;
  real Wtilde0 = invN00*dW00 + invN01*dW01 + invN02*dW02;
  real Wtilde1 = invN10*dW00 + invN11*dW01 + invN12*dW02;
  real Wtilde2 = invN20*dW00 + invN21*dW01 + invN22*dW02;
  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;

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
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny, ic, l1l2);
  pTresTot[n].x += tl1*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl1*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresTot[n].z += tl1*ResLDA;

  Wtilde0 = invN00*dW10 + invN01*dW11 + invN02*dW12;
  Wtilde1 = invN10*dW10 + invN11*dW11 + invN12*dW12;
  Wtilde2 = invN20*dW10 + invN21*dW11 + invN22*dW12;
  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;

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
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny, ic, l1l2);
  pTresTot[n].x += tl2*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl2*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresTot[n].z += tl2*ResLDA;

  Wtilde0 = invN00*dW20 + invN01*dW21 + invN02*dW22;
  Wtilde1 = invN10*dW20 + invN11*dW21 + invN12*dW22;
  Wtilde2 = invN20*dW20 + invN21*dW21 + invN22*dW22;
  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;

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
    -Wtilde0*isoKMP00(ic, wtilde, l1l2, l2)
    -Wtilde1*isoKMP01(nx, ic, l1l2)
    -Wtilde2*isoKMP02(ny, ic, l1l2);
  pTresTot[n].x += tl3*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP11(nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresTot[n].y += tl3*ResLDA;

  ResLDA =
    -Wtilde0*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*isoKMP21(nx, ny, vc, l1l2l3, l1l2)
    -Wtilde2*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresTot[n].z += tl3*ResLDA;
}

//! Version for single equation
template<ConservationLaw CL>
__host__ __device__
void MassMatrixF34TotSingle(int n, real dt, int massMatrix,
                            const int3* __restrict__ pTv,
                            const real* __restrict__ pVz,
                            const real* __restrict__ pDstate,
                            real *pTresTot, const real2 *pTn1,
                            const real2 *pTn2, const real2 *pTn3,
                            const real3 *pTl, int nVertex,
                            real G, real G1, real G2, real *pVp)
{
  const real zero = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;

  int vs1 = pTv[n].x;
  int vs2 = pTv[n].y;
  int vs3 = pTv[n].z;
  while (vs1 >= nVertex) vs1 -= nVertex;
  while (vs2 >= nVertex) vs2 -= nVertex;
  while (vs3 >= nVertex) vs3 -= nVertex;
  while (vs1 < 0) vs1 += nVertex;
  while (vs2 < 0) vs2 += nVertex;
  while (vs3 < 0) vs3 += nVertex;

  // State differences
  real dW0 = pDstate[vs1];
  real dW1 = pDstate[vs2];
  real dW2 = pDstate[vs3];

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(tl1 + tl2 + tl3);
  real Adt = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3))*onethird/dt;
  if (massMatrix == 3) Adt = -Adt;

  dW0 *= Adt;
  dW1 *= Adt;
  dW2 *= Adt;

  // Average parameter vector
  real vx = (real) 1.0;
  real vy = zero;
  if (CL == CL_BURGERS) {
    real Zv0 = pVz[vs1];
    real Zv1 = pVz[vs2];
    real Zv2 = pVz[vs3];

    real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
    vx = Z0;
    vy = Z0;
  }

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate N
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;

  // First direction
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;

  real l1 = min(zero, vx*nx + vy*ny);
  real nm = l1*tl;

  // Second direction
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;

  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;

  l1 = min(zero, vx*nx + vy*ny);
  nm += l1*tl;

  // Third direction
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;

  l1 = min(zero, vx*nx + vy*ny);
  nm += l1*tl;

  real invN = (real) 1.0;
  if (nm != zero) invN /= nm;

  // Wtilde = N*phi
  real Wtilde = invN*dW0;
  real ResLDA;

  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*tl1*Tnx1;
  ny = half*tl1*Tny1;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  ResLDA = -l1*Wtilde;

  pTresTot[n] += ResLDA;

  // Wtilde = N*dW1
  Wtilde = invN*dW1;

  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  // Second direction
  nx = half*tl2*Tnx2;
  ny = half*tl2*Tny2;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  ResLDA = -l1*Wtilde;

  pTresTot[n] += ResLDA;

  // Wtilde = N*dW1
  Wtilde = invN*dW2;

  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  // Third direction
  nx = half*tl3*Tnx3;
  ny = half*tl3*Tny3;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  ResLDA = -l1*Wtilde;

  pTresTot[n] += ResLDA;
}

//######################################################################
/*! \brief Kernel calculating space-time LDA residue for all triangles

\param nTriangle Total number of triangles in Mesh
\param dt Time step
\param massMatrix Mass matrix to use (should be 3 or 4)
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pDstate Pointer to state difference at vertices
\param *pTresTot Triangle total residue
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devMassMatrixF34Tot(int nTriangle, real dt, int massMatrix,
                    const int3* __restrict__ pTv,
                    const realNeq* __restrict__ pVz,
                    const realNeq* __restrict__ pDstate,
                    realNeq *pTresTot, const real2 *pTn1, const real2 *pTn2,
                    const real2 *pTn3, const real3 *pTl,
                    int nVertex, real G, real G1, real G2, real *pVp)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    MassMatrixF34TotSingle<CL>(n, dt, massMatrix, pTv, pVz, pDstate,
                               pTresTot, pTn1, pTn2, pTn3, pTl,
                               nVertex, G, G1, G2, pVp);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate mass matrix contribution F3/F4 to total residual.*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::MassMatrixF34Tot(real dt, int massMatrix)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pVz = vertexParameterVector->GetPointer();
  realNeq *pDstate = vertexStateDiff->GetPointer();
  real G = simulationParameter->specificHeatRatio;
  real *pVp = vertexPotential->GetPointer();

  realNeq *pTresTot = triangleResidueTotal->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();
  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);
  const real3 *pTl  = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devMassMatrixF34Tot<realNeq, CL>,
                                       (size_t) 0, 0);

    devMassMatrixF34Tot<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, dt, massMatrix, pTv, pVz, pDstate,
       pTresTot, pTn1, pTn2, pTn3, pTl, nVertex,
       G, G - 1.0, G - 2.0, pVp);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  } else {
    for (int n = 0; n < nTriangle; n++)
      MassMatrixF34TotSingle<CL>(n, dt, massMatrix, pTv, pVz, pDstate,
                                 pTresTot, pTn1, pTn2, pTn3, pTl, nVertex,
                                 G, G - 1.0, G - 2.0, pVp);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void
Simulation<real, CL_ADVECT>::MassMatrixF34Tot(real dt, int massMatrix);
template void
Simulation<real, CL_BURGERS>::MassMatrixF34Tot(real dt, int massMatrix);
template void
Simulation<real3, CL_CART_ISO>::MassMatrixF34Tot(real dt, int massMatrix);
template void
Simulation<real3, CL_CYL_ISO>::MassMatrixF34Tot(real dt, int massMatrix);
template void
Simulation<real4, CL_CART_EULER>::MassMatrixF34Tot(real dt, int massMatrix);

}  // namespace astrix
