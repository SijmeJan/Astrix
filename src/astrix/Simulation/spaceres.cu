// -*-c++-*-
/*! \file spaceres.cu
\brief File containing functions for calculating spatial residue

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>
#include <iomanip>

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
/*! \brief Calculate spacial residue at triangle n

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pResSource Pointer to source term contribution to residual
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresTot Triangle total residue
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2*/
//######################################################################

__host__ __device__
void CalcSpaceResSingle(int n, const int3 *pTv, real4 *pVz,
                        const real2 *pTn1, const real2 *pTn2,
                        const real2 *pTn3, const real3 *pTl,
                        real4 *pResSource,
                        real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                        real4 *pTresLDA0, real4 *pTresLDA1, real4 *pTresLDA2,
                        real4 *pTresTot, int nVertex, real G, real G1, real G2)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one = (real) 1.0;
  const real two = (real) 2.0;

  // Vertices belonging to triangle: 3 coalesced reads
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv00 = pVz[v1].x;
  real Zv01 = pVz[v1].y;
  real Zv02 = pVz[v1].z;
  real Zv03 = pVz[v1].w;
  real Zv10 = pVz[v2].x;
  real Zv11 = pVz[v2].y;
  real Zv12 = pVz[v2].z;
  real Zv13 = pVz[v2].w;
  real Zv20 = pVz[v3].x;
  real Zv21 = pVz[v3].y;
  real Zv22 = pVz[v3].z;
  real Zv23 = pVz[v3].w;

  // Average parameter vector
  real Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  real Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  real Z2 = (Zv02 + Zv12 + Zv22)*onethird;
  real Z3 = (Zv03 + Zv13 + Zv23)*onethird;

  // Average state at vertices
  real What00 = two*Z0*Zv00;
  real What01 = Z1*Zv00 + Z0*Zv01;
  real What02 = Z2*Zv00 + Z0*Zv02;
  real What03 = (Z3*Zv00 + G1*Z1*Zv01 + G1*Z2*Zv02 + Z0*Zv03)/G;
  real What10 = two*Z0*Zv10;
  real What11 = Z1*Zv10 + Z0*Zv11;
  real What12 = Z2*Zv10 + Z0*Zv12;
  real What13 = (Z3*Zv10 + G1*Z1*Zv11 + G1*Z2*Zv12 + Z0*Zv13)/G;
  real What20 = two*Z0*Zv20;
  real What21 = Z1*Zv20 + Z0*Zv21;
  real What22 = Z2*Zv20 + Z0*Zv22;
  real What23 = (Z3*Zv20 + G1*Z1*Zv21 + G1*Z2*Zv22 + Z0*Zv23)/G;

  real ResTot0 = pResSource[n].x;
  real ResTot1 = pResSource[n].y;
  real ResTot2 = pResSource[n].z;
  real ResTot3 = pResSource[n].w;

  real Wtemp0 = pResSource[n].x;
  real Wtemp1 = pResSource[n].y;
  real Wtemp2 = pResSource[n].z;
  real Wtemp3 = pResSource[n].w;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real htilde = Z3/Z0;
  real alpha  = G1*half*(Sq(utilde) + Sq(vtilde));

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tnx2 = pTn2[n].x;
  real tnx3 = pTn3[n].x;
  real tny1 = pTn1[n].y;
  real tny2 = pTn2[n].y;
  real tny3 = pTn3[n].y;

  // First direction
  real tl = tl1;
  real nx = half*tl*tnx1;
  real ny = half*tl*tny1;
  real wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What01*eulerK01(nx) +
    What02*eulerK02(ny);
  ResTot1 +=
    What00*eulerK10(nx, alpha, wtilde, utilde) +
    What01*eulerK11(G2, nx, wtilde, utilde) +
    What02*eulerK12(G1, nx, ny, utilde, vtilde) +
    What03*eulerK13(G1, nx);
  ResTot2 +=
    What00*eulerK20(ny, alpha, wtilde, vtilde) +
    What01*eulerK21(G1, nx, ny, utilde, vtilde) +
    What02*eulerK22(G2, ny, wtilde, vtilde) +
    What03*eulerK23(G1, ny);
  ResTot3 +=
    What00*eulerK30(alpha, htilde, wtilde) +
    What01*eulerK31(G1, nx, htilde, wtilde, utilde) +
    What02*eulerK32(G1, ny, htilde, wtilde, vtilde) +
    What03*eulerK33(G, wtilde);

  // Second direction
  tl = tl2;
  nx = half*tl*tnx2;
  ny = half*tl*tny2;
  wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What11*eulerK01(nx) +
    What12*eulerK02(ny);
  ResTot1 +=
    What10*eulerK10(nx, alpha, wtilde, utilde) +
    What11*eulerK11(G2, nx, wtilde, utilde) +
    What12*eulerK12(G1, nx, ny, utilde, vtilde) +
    What13*eulerK13(G1, nx);
  ResTot2 +=
    What10*eulerK20(ny, alpha, wtilde, vtilde) +
    What11*eulerK21(G1, nx, ny, utilde, vtilde) +
    What12*eulerK22(G2, ny, wtilde, vtilde) +
    What13*eulerK23(G1, ny);
  ResTot3 +=
    What10*eulerK30(alpha, htilde, wtilde) +
    What11*eulerK31(G1, nx, htilde, wtilde, utilde) +
    What12*eulerK32(G1, ny, htilde, wtilde, vtilde) +
    What13*eulerK33(G, wtilde);

  // Third direction
  tl = tl3;
  nx = half*tl*tnx3;
  ny = half*tl*tny3;
  wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What21*eulerK01(nx) +
    What22*eulerK02(ny);
  pTresTot[n].x = ResTot0;
  ResTot1 +=
    What20*eulerK10(nx, alpha, wtilde, utilde) +
    What21*eulerK11(G2, nx, wtilde, utilde) +
    What22*eulerK12(G1, nx, ny, utilde, vtilde) +
    What23*eulerK13(G1, nx);
  pTresTot[n].y = ResTot1;
  ResTot2 +=
    What20*eulerK20(ny, alpha, wtilde, vtilde) +
    What21*eulerK21(G1, nx, ny, utilde, vtilde) +
    What22*eulerK22(G2, ny, wtilde, vtilde) +
    What23*eulerK23(G1, ny);
  pTresTot[n].z = ResTot2;
  ResTot3 +=
    What20*eulerK30(alpha, htilde, wtilde) +
    What21*eulerK31(G1, nx, htilde, wtilde, utilde) +
    What22*eulerK32(G1, ny, htilde, wtilde, vtilde) +
    What23*eulerK33(G, wtilde);
  pTresTot[n].w = ResTot3;

  /*
#ifndef __CUDA_ARCH__
  std::cout << ResTot0 << " "
            << ResTot1 << " "
            << ResTot2 << " "
            << ResTot3 << std::endl;
  int qq; std::cin >> qq;
#endif
  */

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real km;
  real ic = one/sqrt(G1*htilde - alpha);
  real ctilde = one/ic;

  real hc = htilde*ic;
  real uc = utilde*ic;
  real vc = vtilde*ic;
  real ac = alpha*ic;

  // First direction
  nx = tnx1;
  ny = tny1;
  tl = half*tl1;
  wtilde = utilde*nx + vtilde*ny;

  real l1 = min(zero, wtilde + ctilde);
  real l2 = min(zero, wtilde - ctilde);
  real l3 = min(zero, wtilde);

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);

  km = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  Wtemp0 += tl*km*What00;
  real nm00 = tl*km;

  km = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What01;
  real nm01 = tl*km;

  km = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What02;
  real nm02 = tl*km;

  km = eulerKMP03(G1, ic, l1l2l3);
  Wtemp0 += tl*km*What03;
  real nm03 = tl*km;

  km = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What00;
  real nm10 = tl*km;

  km = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What01;
  real nm11 = tl*km;

  km = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What02;
  real nm12 = tl*km;

  km = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What03;
  real nm13 = tl*km;

  km = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What00;
  real nm20 = tl*km;

  km = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What01;
  real nm21 = tl*km;

  km = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What02;
  real nm22 = tl*km;

  km = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What03;
  real nm23 = tl*km;

  km = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What00;
  real nm30 = tl*km;

  km = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What01;
  real nm31 = tl*km;

  km = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What02;
  real nm32 = tl*km;

  km = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  Wtemp3 += tl*km*What03;
  real nm33 = tl*km;

  // Second direction
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  km = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  Wtemp0 += tl*km*What10;
  nm00 += tl*km;

  km = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What11;
  nm01 += tl*km;

  km = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What12;
  nm02 += tl*km;

  km = eulerKMP03(G1, ic, l1l2l3);
  Wtemp0 += tl*km*What13;
  nm03 += tl*km;

  km = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What10;
  nm10 += tl*km;

  km = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What11;
  nm11 += tl*km;

  km = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What12;
  nm12 += tl*km;

  km = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What13;
  nm13 += tl*km;

  km = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What10;
  nm20 += tl*km;

  km = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What11;
  nm21 += tl*km;

  km = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What12;
  nm22 += tl*km;

  km = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What13;
  nm23 += tl*km;

  km = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What10;
  nm30 += tl*km;

  km = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What11;
  nm31 += tl*km;

  km = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What12;
  nm32 += tl*km;

  km = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  Wtemp3 += tl*km*What13;
  nm33 += tl*km;

  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  km = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  Wtemp0 += tl*km*What20;
  nm00 += tl*km;

  km = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What21;
  nm01 += tl*km;

  km = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp0 += tl*km*What22;
  nm02 += tl*km;

  km = eulerKMP03(G1, ic, l1l2l3);
  Wtemp0 += tl*km*What23;
  nm03 += tl*km;

  km = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What20;
  nm10 += tl*km;

  km = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What21;
  nm11 += tl*km;

  km = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What22;
  nm12 += tl*km;

  km = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What23;
  nm13 += tl*km;

  km = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What20;
  nm20 += tl*km;

  km = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What21;
  nm21 += tl*km;

  km = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What22;
  nm22 += tl*km;

  km = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What23;
  nm23 += tl*km;

  km = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What20;
  nm30 += tl*km;

  km = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What21;
  nm31 += tl*km;

  km = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  Wtemp3 += tl*km*What22;
  nm32 += tl*km;

  km = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  Wtemp3 += tl*km*What23;
  nm33 += tl*km;

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

  // Wtilde = Nm*Wtemp
  real Wtilde0 = invN00*Wtemp0 + invN01*Wtemp1 + invN02*Wtemp2 + invN03*Wtemp3;
  real Wtilde1 = invN10*Wtemp0 + invN11*Wtemp1 + invN12*Wtemp2 + invN13*Wtemp3;
  real Wtilde2 = invN20*Wtemp0 + invN21*Wtemp1 + invN22*Wtemp2 + invN23*Wtemp3;
  real Wtilde3 = invN30*Wtemp0 + invN31*Wtemp1 + invN32*Wtemp2 + invN33*Wtemp3;

  // Wtemp = Nm*ResTot
  Wtemp0 = invN00*ResTot0 + invN01*ResTot1 + invN02*ResTot2 + invN03*ResTot3;
  Wtemp1 = invN10*ResTot0 + invN11*ResTot1 + invN12*ResTot2 + invN13*ResTot3;
  Wtemp2 = invN20*ResTot0 + invN21*ResTot1 + invN22*ResTot2 + invN23*ResTot3;
  Wtemp3 = invN30*ResTot0 + invN31*ResTot1 + invN32*ResTot2 + invN33*ResTot3;

  if (det != zero) det = one/det;

  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;
  Wtilde3 *= det;

  // What = What - Wtilde
  What00 -= Wtilde0;
  What01 -= Wtilde1;
  What02 -= Wtilde2;
  What03 -= Wtilde3;
  What10 -= Wtilde0;
  What11 -= Wtilde1;
  What12 -= Wtilde2;
  What13 -= Wtilde3;
  What20 -= Wtilde0;
  What21 -= Wtilde1;
  What22 -= Wtilde2;
  What23 -= Wtilde3;

  Wtemp0 *= det;
  Wtemp1 *= det;
  Wtemp2 *= det;
  Wtemp3 *= det;

  // PhiN = Kp*(What - Ninv*Sum(Km*What))
  real ResN, ResLDA, kp;

  real Tnx1 = pTn1[n].x;
  real Tnx2 = pTn2[n].x;
  real Tnx3 = pTn3[n].x;
  real Tny1 = pTn1[n].y;
  real Tny2 = pTn2[n].y;
  real Tny3 = pTn3[n].y;

  // First direction
  nx = Tnx1;
  ny = Tny1;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  kp = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;

  pTresN0[n].x   = half*ResN;
  pTresLDA0[n].x = half*ResLDA;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;

  pTresN0[n].y   = half*ResN;
  pTresLDA0[n].y = half*ResLDA;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;

  pTresN0[n].z   = half*ResN;
  pTresLDA0[n].z = half*ResLDA;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;

  pTresN0[n].w   = half*ResN;
  pTresLDA0[n].w = half*ResLDA;

  // Second direction
  nx = Tnx2;
  ny = Tny2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  kp = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  ResN   =  kp*What10;
  ResLDA = -kp*Wtemp0;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;

  pTresN1[n].x   = half*ResN;
  pTresLDA1[n].x = half*ResLDA;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;

  pTresN1[n].y   = half*ResN;
  pTresLDA1[n].y = half*ResLDA;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN   =  kp*What10;
  ResLDA = -kp*Wtemp0;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;

  pTresN1[n].z   = half*ResN;
  pTresLDA1[n].z = half*ResLDA;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;

  pTresN1[n].w   = half*ResN;
  pTresLDA1[n].w = half*ResLDA;

  // Third direction
  nx = Tnx3;
  ny = Tny3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  kp = eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3);
  ResN   =  kp*What20;
  ResLDA = -kp*Wtemp0;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;

  pTresN2[n].x   = half*ResN;
  pTresLDA2[n].x = half*ResLDA;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;

  pTresN2[n].y   = half*ResN;
  pTresLDA2[n].y = half*ResLDA;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;

  pTresN2[n].z   = half*ResN;
  pTresLDA2[n].z = half*ResLDA;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN   =  kp*What20;
  ResLDA = -kp*Wtemp0;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;

  pTresN2[n].w   = half*ResN;
  pTresLDA2[n].w = half*ResLDA;
}

__host__ __device__
void CalcSpaceResSingle(int n, const int3 *pTv, real3 *pVz,
                        const real2 *pTn1, const real2 *pTn2,
                        const real2 *pTn3, const real3 *pTl,
                        real3 *pResSource,
                        real3 *pTresN0, real3 *pTresN1, real3 *pTresN2,
                        real3 *pTresLDA0, real3 *pTresLDA1, real3 *pTresLDA2,
                        real3 *pTresTot, int nVertex, real G, real G1, real G2)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one = (real) 1.0;
  const real two = (real) 2.0;

  // Vertices belonging to triangle: 3 coalesced reads
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv00 = pVz[v1].x;
  real Zv01 = pVz[v1].y;
  real Zv02 = pVz[v1].z;
  real Zv10 = pVz[v2].x;
  real Zv11 = pVz[v2].y;
  real Zv12 = pVz[v2].z;
  real Zv20 = pVz[v3].x;
  real Zv21 = pVz[v3].y;
  real Zv22 = pVz[v3].z;

  // Average parameter vector
  real Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  real Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  real Z2 = (Zv02 + Zv12 + Zv22)*onethird;

  // Average state at vertices
  real What00 = two*Z0*Zv00;
  real What01 = Z1*Zv00 + Z0*Zv01;
  real What02 = Z2*Zv00 + Z0*Zv02;
  real What10 = two*Z0*Zv10;
  real What11 = Z1*Zv10 + Z0*Zv11;
  real What12 = Z2*Zv10 + Z0*Zv12;
  real What20 = two*Z0*Zv20;
  real What21 = Z1*Zv20 + Z0*Zv21;
  real What22 = Z2*Zv20 + Z0*Zv22;

  // Source term residual
  // real rhoAve  = Z0*Z0;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tnx2 = pTn2[n].x;
  real tnx3 = pTn3[n].x;
  real tny1 = pTn1[n].y;
  real tny2 = pTn2[n].y;
  real tny3 = pTn3[n].y;

  real ResTot0 = pResSource[n].x;
  real ResTot1 = pResSource[n].y;
  real ResTot2 = pResSource[n].z;

  real Wtemp0 = pResSource[n].x;
  real Wtemp1 = pResSource[n].y;
  real Wtemp2 = pResSource[n].z;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real ctilde = one;     // Sound speed is unity by assumption!

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // First direction
  real tl = tl1;
  real nx = half*tl*tnx1;
  real ny = half*tl*tny1;
  real wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What01*isoK01(nx) +
    What02*isoK02(ny);
  ResTot1 +=
    What00*isoK10(nx, ctilde, wtilde, utilde) +
    What01*isoK11(nx, wtilde, utilde) +
    What02*isoK12(ny, utilde);
  ResTot2 +=
    What00*isoK20(ny, ctilde, wtilde, vtilde) +
    What01*isoK21(nx, vtilde) +
    What02*isoK22(ny, wtilde, vtilde);

  // Second direction
  tl = tl2;
  nx = half*tl*tnx2;
  ny = half*tl*tny2;
  wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What11*isoK01(nx) +
    What12*isoK02(ny);
  ResTot1 +=
    What10*isoK10(nx, ctilde, wtilde, utilde) +
    What11*isoK11(nx, wtilde, utilde) +
    What12*isoK12(ny, utilde);
  ResTot2 +=
    What10*isoK20(ny, ctilde, wtilde, vtilde) +
    What11*isoK21(nx, vtilde) +
    What12*isoK22(ny, wtilde, vtilde);

  // Third direction
  tl = tl3;
  nx = half*tl*tnx3;
  ny = half*tl*tny3;
  wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What21*isoK01(nx) +
    What22*isoK02(ny);
  ResTot1 +=
    What20*isoK10(nx, ctilde, wtilde, utilde) +
    What21*isoK11(nx, wtilde, utilde) +
    What22*isoK12(ny, utilde);
  ResTot2 +=
    What20*isoK20(ny, ctilde, wtilde, vtilde) +
    What21*isoK21(nx, vtilde) +
    What22*isoK22(ny, wtilde, vtilde);

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real km;
  real ic = one/ctilde;

  real uc = utilde*ic;
  real vc = vtilde*ic;

  // First direction
  nx = tnx1;
  ny = tny1;
  tl = half*tl1;
  wtilde = utilde*nx + vtilde*ny;

  real l1 = min(zero, wtilde + ctilde);
  real l2 = min(zero, wtilde - ctilde);
  real l3 = min(zero, wtilde);

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);

  // km[0][0][0]
  km = isoKMP00(ic, wtilde, l1l2, l2);
  Wtemp0 += tl*km*What00;
  real nm00 = tl*km;

  // km[0][0][1]
  km = isoKMP01(nx, ic, l1l2);
  Wtemp0 += tl*km*What01;
  real nm01 = tl*km;

  // km[0][0][2]
  km = isoKMP02(ny, ic, l1l2);
  Wtemp0 += tl*km*What02;
  real nm02 = tl*km;

  // km[0][1][0]
  km = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What00;
  real nm10 = tl*km;

  // km[0][1][1]
  km = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What01;
  real nm11 = tl*km;

  // km[0][1][2]
  km = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What02;
  real nm12 = tl*km;

  // km[0][2][0]
  km = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What00;
  real nm20 = tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What01;
  real nm21 = tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What02;
  real nm22 = tl*km;

  // Second direction
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // km[0][0][0]
  km = isoKMP00(ic, wtilde, l1l2, l2);
  Wtemp0 += tl*km*What10;
  nm00 += tl*km;

  // km[0][0][1]
  km = isoKMP01(nx, ic, l1l2);
  Wtemp0 += tl*km*What11;
  nm01 += tl*km;

  // km[0][0][2]
  km = isoKMP02(ny, ic, l1l2);
  Wtemp0 += tl*km*What12;
  nm02 += tl*km;

  // km[0][1][0]
  km = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What10;
  nm10 += tl*km;

  // km[0][1][1]
  km = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What11;
  nm11 += tl*km;

  // km[0][1][2]
  km = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What12;
  nm12 += tl*km;

  // km[0][2][0]
  km = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What10;
  nm20 += tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What11;
  nm21 += tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What12;
  nm22 += tl*km;

  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // km[0][0][0]
  km = isoKMP00(ic, wtilde, l1l2, l2);
  Wtemp0 += tl*km*What20;
  nm00 += tl*km;

  // km[0][0][1]
  km = isoKMP01(nx, ic, l1l2);
  Wtemp0 += tl*km*What21;
  nm01 += tl*km;

  // km[0][0][2]
  km = isoKMP02(ny, ic, l1l2);
  Wtemp0 += tl*km*What22;
  nm02 += tl*km;

  // km[0][1][0]
  km = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  Wtemp1 += tl*km*What20;
  nm10 += tl*km;

  // km[0][1][1]
  km = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  Wtemp1 += tl*km*What21;
  nm11 += tl*km;

  // km[0][1][2]
  km = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What22;
  nm12 += tl*km;

  // km[0][2][0]
  km = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What20;
  nm20 += tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What21;
  nm21 += tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What22;
  nm22 += tl*km;

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

  // Wtilde = Nm*Wtemp
  real Wtilde0 = invN00*Wtemp0 + invN01*Wtemp1 + invN02*Wtemp2;
  real Wtilde1 = invN10*Wtemp0 + invN11*Wtemp1 + invN12*Wtemp2;
  real Wtilde2 = invN20*Wtemp0 + invN21*Wtemp1 + invN22*Wtemp2;

  // Wtemp = Nm*ResTot
  Wtemp0 = invN00*ResTot0 + invN01*ResTot1 + invN02*ResTot2;
  Wtemp1 = invN10*ResTot0 + invN11*ResTot1 + invN12*ResTot2;
  Wtemp2 = invN20*ResTot0 + invN21*ResTot1 + invN22*ResTot2;

  if (det != zero) det = one/det;

  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;

  // What = What - Wtilde
  What00 -= Wtilde0;
  What01 -= Wtilde1;
  What02 -= Wtilde2;
  What10 -= Wtilde0;
  What11 -= Wtilde1;
  What12 -= Wtilde2;
  What20 -= Wtilde0;
  What21 -= Wtilde1;
  What22 -= Wtilde2;

  Wtemp0 *= det;
  Wtemp1 *= det;
  Wtemp2 *= det;

  // PhiN = Kp*(What - Ninv*Sum(Km*What))
  real ResN, ResLDA, kp;

  real Tnx1 = pTn1[n].x;
  real Tnx2 = pTn2[n].x;
  real Tnx3 = pTn3[n].x;
  real Tny1 = pTn1[n].y;
  real Tny2 = pTn2[n].y;
  real Tny3 = pTn3[n].y;

  // First direction
  nx = Tnx1;
  ny = Tny1;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  // kp[0][0][2]
  kp = isoKMP02(ny, ic, l1l2);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  pTresN0[n].x   = half*ResN;
  pTresLDA0[n].x = half*ResLDA;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  pTresN0[n].y   = half*ResN;
  pTresLDA0[n].y = half*ResLDA;

  // kp[0][2][0]
  kp = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;

  pTresN0[n].z   = half*ResN;
  pTresLDA0[n].z = half*ResLDA;

  // Second direction
  nx = Tnx2;
  ny = Tny2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  // kp[0][0][2]
  kp = isoKMP02(ny, ic, l1l2);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  pTresN1[n].x   = half*ResN;
  pTresLDA1[n].x = half*ResLDA;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  pTresN1[n].y   = half*ResN;
  pTresLDA1[n].y = half*ResLDA;

  // kp[0][2][0]
  kp = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;

  pTresN1[n].z   = half*ResN;
  pTresLDA1[n].z = half*ResLDA;

  // Third direction
  nx = Tnx3;
  ny = Tny3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

    // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  // kp[0][0][2]
  kp = isoKMP02(ny, ic, l1l2);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  pTresN2[n].x   = half*ResN;
  pTresLDA2[n].x = half*ResLDA;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  pTresN2[n].y   = half*ResN;
  pTresLDA2[n].y = half*ResLDA;

  // kp[0][2][0]
  kp = isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny, vc, l1l2l3, l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;

  pTresN2[n].z   = half*ResN;
  pTresLDA2[n].z = half*ResLDA;
}

__host__ __device__
void CalcSpaceResSingle(int n, const int3 *pTv, real *pVz,
                        const real2 *pTn1, const real2 *pTn2,
                        const real2 *pTn3, const real3 *pTl,
                        real *pResSource,
                        real *pTresN0, real *pTresN1, real *pTresN2,
                        real *pTresLDA0, real *pTresLDA1, real *pTresLDA2,
                        real *pTresTot, int nVertex, real G, real G1, real G2)
{
  const real zero  = (real) 0.0;
  const real half  = (real) 0.5;
  const real one = (real) 1.0;

  // Vertices belonging to triangle: 3 coalesced reads
  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv0 = pVz[v1];
  real Zv1 = pVz[v2];
  real Zv2 = pVz[v3];

  // Average parameter vector
#if BURGERS == 1
  const real onethird = (real) (1.0/3.0);
  real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
  real vx = Z0;
  real vy = Z0;
#else
  real vx = one;
  real vy = one;
#endif

  // Average state at vertices
  real What0 = Zv0;
  real What1 = Zv1;
  real What2 = Zv2;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tnx2 = pTn2[n].x;
  real tnx3 = pTn3[n].x;
  real tny1 = pTn1[n].y;
  real tny2 = pTn2[n].y;
  real tny3 = pTn3[n].y;

  // Total residue
  real ResTot = pResSource[n];
  real Wtemp = pResSource[n];

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // First direction
  real nx = half*tl1*tnx1;
  real ny = half*tl1*tny1;

  ResTot += (vx*nx + vy*ny)*What0;

  // Second direction
  nx = half*tl2*tnx2;
  ny = half*tl2*tny2;

  ResTot += (vx*nx + vy*ny)*What1;

  // Third direction
  nx = half*tl3*tnx3;
  ny = half*tl3*tny3;

  ResTot += (vx*nx + vy*ny)*What2;

  pTresTot[n] = ResTot;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // First direction
  nx = tnx1;
  ny = tny1;
  real tl = half*tl1;

  real l1 = min(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What0;
  real nm = tl*l1;

  // Second direction
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;

  l1 = min(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What1;
  nm += tl*l1;

  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;

  l1 = min(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What2;
  nm += tl*l1;

  real invN = one;
  if (nm != zero) invN /= nm;

  real Wtilde = invN*Wtemp;

  // Wtemp = Nm*ResTot
  Wtemp = invN*ResTot;

  What0 -= Wtilde;
  What1 -= Wtilde;
  What2 -= Wtilde;

  // PhiN = Kp*(What - Ninv*Sum(Km*What))
  real ResN, ResLDA;

  real Tnx1 = pTn1[n].x;
  real Tnx2 = pTn2[n].x;
  real Tnx3 = pTn3[n].x;
  real Tny1 = pTn1[n].y;
  real Tny2 = pTn2[n].y;
  real Tny3 = pTn3[n].y;

  nx = Tnx1;
  ny = Tny1;
  l1 = max(zero, vx*nx + vy*ny);

  ResN   = l1*What0;
  ResLDA =-l1*Wtemp;

  pTresN0[n]   = half*ResN;
  pTresLDA0[n] = half*ResLDA;

  // Second direction
  nx = Tnx2;
  ny = Tny2;
  l1 = max(zero, vx*nx + vy*ny);

  ResN   = l1*What1;
  ResLDA =-l1*Wtemp;

  pTresN1[n]   = half*ResN;
  pTresLDA1[n] = half*ResLDA;

  // Third direction
  nx = Tnx3;
  ny = Tny3;

  l1 = max(zero, vx*nx + vy*ny);

  ResN   = l1*What2;
  ResLDA =-l1*Wtemp;

  pTresN2[n]   = half*ResN;
  pTresLDA2[n] = half*ResLDA;
}

//######################################################################
/*! \brief Kernel calculating spacial residue for all triangles

\param nTriangle Total number of triangles in Mesh
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pResSource Pointer to residual due to source terms
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param *pTresLDA0 Triangle residue LDA direction 0
\param *pTresLDA1 Triangle residue LDA direction 1
\param *pTresLDA2 Triangle residue LDA direction 2
\param *pTresTot Triangle total residue
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2*/
//######################################################################

__global__ void
devCalcSpaceRes(int nTriangle, const int3 *pTv, realNeq *pVz,
                const real2 *pTn1, const real2 *pTn2,
                const real2 *pTn3, const real3 *pTl, realNeq *pResSource,
                realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
                realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
                realNeq *pTresTot, int nVertex, real G, real G1, real G2)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;


  while (n < nTriangle) {
    CalcSpaceResSingle(n, pTv, pVz, pTn1, pTn2, pTn3, pTl, pResSource,
                       pTresN0, pTresN1, pTresN2,
                       pTresLDA0, pTresLDA1, pTresLDA2,
                       pTresTot, nVertex, G, G1, G2);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate spatial residue for all triangles; result in \a triangleResidueN,
\a triangleResidueLDA and \a triangleResidueTotal.*/
//######################################################################

void Simulation::CalcResidual()
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 2.0f;
  gpuErrchk( cudaEventCreate(&start) );
  gpuErrchk( cudaEventCreate(&stop) );
#endif
  int transformFlag = 0;
  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexParameterVector->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
      triangleResidueSource->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexPotential->TransformToHost();
      vertexParameterVector->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleResidueTotal->TransformToHost();
      triangleResidueSource->TransformToHost();

      cudaFlag = 0;
    }
  }

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pResSource = triangleResidueSource->GetPointer();
  realNeq *pVz = vertexParameterVector->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);

  realNeq *pTresTot = triangleResidueTotal->GetPointer();

  const int3 *pTv = mesh->TriangleVerticesData();

  const real2 *pTn1 = mesh->TriangleEdgeNormalsData(0);
  const real2 *pTn2 = mesh->TriangleEdgeNormalsData(1);
  const real2 *pTn3 = mesh->TriangleEdgeNormalsData(2);

  const real3 *pTl = mesh->TriangleEdgeLengthData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devCalcSpaceRes,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcSpaceRes<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVz,
       pTn1, pTn2, pTn3, pTl, pResSource,
       pTresN0, pTresN1, pTresN2,
       pTresLDA0, pTresLDA1, pTresLDA2,
       pTresTot, nVertex, G, G - 1.0, G - 2.0);
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
      CalcSpaceResSingle(n, pTv, pVz,
                         pTn1, pTn2, pTn3, pTl, pResSource,
                         pTresN0, pTresN1, pTresN2,
                         pTresLDA0, pTresLDA1, pTresLDA2,
                         pTresTot, nVertex, G, G - 1.0, G - 2.0);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("CalcResidual.prof2", nTriangle, elapsedTime, cudaFlag);
#endif

  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 1) {
      vertexParameterVector->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleResidueTotal->TransformToHost();
      triangleResidueSource->TransformToHost();

      cudaFlag = 0;
    } else {
      vertexParameterVector->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
      triangleResidueSource->TransformToDevice();

      cudaFlag = 1;
    }
  }
}

}  // namespace astrix
