// -*-c++-*-
/*! \file totalresNtot.cu
\brief File containing functions for calculating space-time N + total residue

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
#include "./upwind.h"
#include "../Common/profile.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//######################################################################
/*! Calculate space-time N + total residue at triangle n

\param n Triangle to consider
\param dt Time step
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pDstate Pointer to state differences (new - old)
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pResSource Source term contribution to residual
\param *pTresN0 Triangle residue N direction 0 (output)
\param *pTresN1 Triangle residue N direction 1 (output)
\param *pTresN2 Triangle residue N direction 2 (output)
\param *pTresTot Triangle total residue (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param iG 1/G
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<ConservationLaw CL>
__host__ __device__
void CalcTotalResNtotSingle(const int n, const real dt,
                            const int3* __restrict__ pTv,
                            const real4* __restrict__ pVz,
                            real4 *pDstate,
                            const real2 *pTn1,
                            const real2 *pTn2,
                            const real2 *pTn3,
                            const real3* __restrict__ pTl,
                            real4 *pResSource,
                            real4 *pTresN0,
                            real4 *pTresN1,
                            real4 *pTresN2,
                            real4 *pTresTot,
                            int nVertex, const real G, const real G1,
                            const real G2, const real iG, const real *pVp)
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

  // State difference between old and RK1
  real dW00 = pDstate[v1].x;
  real dW01 = pDstate[v1].y;
  real dW02 = pDstate[v1].z;
  real dW03 = pDstate[v1].w;
  real dW10 = pDstate[v2].x;
  real dW11 = pDstate[v2].y;
  real dW12 = pDstate[v2].z;
  real dW13 = pDstate[v2].w;
  real dW20 = pDstate[v3].x;
  real dW21 = pDstate[v3].y;
  real dW22 = pDstate[v3].z;
  real dW23 = pDstate[v3].w;

  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  // Adt = |T|/(3*dt)
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*onethird/dt;

  // For N residuals divide by edge length
  real Ail1 = Adt/Tl1;
  real Ail2 = Adt/Tl2;
  real Ail3 = Adt/Tl3;

  // Replace ResN
  pTresN0[n].x = half*pTresN0[n].x + dW00*Ail1;
  pTresN0[n].y = half*pTresN0[n].y + dW01*Ail1;
  pTresN0[n].z = half*pTresN0[n].z + dW02*Ail1;
  pTresN0[n].w = half*pTresN0[n].w + dW03*Ail1;
  pTresN1[n].x = half*pTresN1[n].x + dW10*Ail2;
  pTresN1[n].y = half*pTresN1[n].y + dW11*Ail2;
  pTresN1[n].z = half*pTresN1[n].z + dW12*Ail2;
  pTresN1[n].w = half*pTresN1[n].w + dW13*Ail2;
  pTresN2[n].x = half*pTresN2[n].x + dW20*Ail3;
  pTresN2[n].y = half*pTresN2[n].y + dW21*Ail3;
  pTresN2[n].z = half*pTresN2[n].z + dW22*Ail3;
  pTresN2[n].w = half*pTresN2[n].w + dW23*Ail3;

  // Replace ResTot
  real ResTot0 = pTresTot[n].x + two*Adt*(dW00 + dW10 + dW20);
  real ResTot1 = pTresTot[n].y + two*Adt*(dW01 + dW11 + dW21);
  real ResTot2 = pTresTot[n].z + two*Adt*(dW02 + dW12 + dW22);
  real ResTot3 = pTresTot[n].w + two*Adt*(dW03 + dW13 + dW23);

  // External potential at vertices
  real pot0 = pVp[v1];
  real pot1 = pVp[v2];
  real pot2 = pVp[v3];
  real pot = (pot0 + pot1 + pot2)*onethird;

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
  real What03 = (Z3*Zv00 + G1*(Z1*Zv01 + Z2*Zv02) +
                 Z0*Zv03 + two*G1*pot*Z0*Zv00)*iG;
  real What10 = two*Z0*Zv10;
  real What11 = Z1*Zv10 + Z0*Zv11;
  real What12 = Z2*Zv10 + Z0*Zv12;
  real What13 = (Z3*Zv10 + G1*(Z1*Zv11 + Z2*Zv12) +
                 Z0*Zv13 + two*G1*pot*Z0*Zv10)*iG;
  real What20 = two*Z0*Zv20;
  real What21 = Z1*Zv20 + Z0*Zv21;
  real What22 = Z2*Zv20 + Z0*Zv22;
  real What23 = (Z3*Zv20 + G1*(Z1*Zv21 + Z2*Zv22) +
                 Z0*Zv23 + two*G1*pot*Z0*Zv20)*iG;

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real ResSource0 = pResSource[n].x;
  real ResSource1 = pResSource[n].y;
  real ResSource2 = pResSource[n].z;
  real ResSource3 = pResSource[n].w;

  ResTot0 += ResSource0;
  ResTot1 += ResSource1;
  ResTot2 += ResSource2;
  ResTot3 += ResSource3;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real htilde = Z3/Z0;
  real alpha = G1*half*(Sq(utilde) + Sq(vtilde)) - G1*pot;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real Wtemp0 = ResSource0;
  real Wtemp1 = ResSource1;
  real Wtemp2 = ResSource2;
  real Wtemp3 = ResSource3;


#ifndef CONTOUR
  // First direction
  real nx = half*tnx1*tl1;
  real ny = half*tny1*tl1;
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
  nx = half*tnx2*tl2;
  ny = half*tny2*tl2;
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
  nx = half*tnx3*tl3;
  ny = half*tny3*tl3;
  wtilde = utilde*nx + vtilde*ny;

  ResTot0 +=
    What21*eulerK01(nx) +
    What22*eulerK02(ny);
  pTresTot[n].x = half*ResTot0;
  ResTot1 +=
    What20*eulerK10(nx, alpha, wtilde, utilde) +
    What21*eulerK11(G2, nx, wtilde, utilde) +
    What22*eulerK12(G1, nx, ny, utilde, vtilde) +
    What23*eulerK13(G1, nx);
  pTresTot[n].y = half*ResTot1;
  ResTot2 +=
    What20*eulerK20(ny, alpha, wtilde, vtilde) +
    What21*eulerK21(G1, nx, ny, utilde, vtilde) +
    What22*eulerK22(G2, ny, wtilde, vtilde) +
    What23*eulerK23(G1, ny);
  pTresTot[n].z = half*ResTot2;
  ResTot3 +=
    What20*eulerK30(alpha, htilde, wtilde) +
    What21*eulerK31(G1, nx, htilde, wtilde, utilde) +
    What22*eulerK32(G1, ny, htilde, wtilde, vtilde) +
    What23*eulerK33(G, wtilde);
  pTresTot[n].w = half*ResTot3;

#else

  real GG = G1/G;
  real res0 =
    tl3*(Zv00*Zv01 + (Zv00 + Zv10)*(Zv01 + Zv11) + Zv10*Zv11)*tnx3/6.0 +
    tl3*(Zv00*Zv02 + (Zv00 + Zv10)*(Zv02 + Zv12) + Zv10*Zv12)*tny3/6.0 +
    tl1*(Zv10*Zv11 + (Zv10 + Zv20)*(Zv11 + Zv21) + Zv20*Zv21)*tnx1/6.0 +
    tl1*(Zv10*Zv12 + (Zv10 + Zv20)*(Zv12 + Zv22) + Zv20*Zv22)*tny1/6.0 +
    tl2*(Zv20*Zv21 + (Zv20 + Zv00)*(Zv21 + Zv01) + Zv00*Zv01)*tnx2/6.0 +
    tl2*(Zv20*Zv22 + (Zv20 + Zv00)*(Zv22 + Zv02) + Zv00*Zv02)*tny2/6.0;
  ResTot0 -= res0;
  pTresTot[n].x = half*ResTot0;
  Wtemp0 -= res0;
  real res1 =
    tl3*(Sq(Zv01) +
         GG*(Zv00*Zv03 -
             half*(Sq(Zv01) + Sq(Zv02)) -
             Sq(Zv00)*pot0) +
         Sq(Zv01 + Zv11) +
         GG*((Zv00 + Zv10)*(Zv03 + Zv13) -
             half*(Sq(Zv01 + Zv11) + Sq(Zv02 + Zv12)) -
             half*Sq(Zv00 + Zv10)*(pot0 + pot1)) +
         Sq(Zv11) +
         GG*(Zv10*Zv13 -
             half*(Sq(Zv11) + Sq(Zv12)) -
             Sq(Zv10)*pot1))*tnx3/6.0 +
    tl3*(Zv01*Zv02 + (Zv01 + Zv11)*(Zv02 + Zv12) + Zv11*Zv12)*tny3/6.0 +
    tl1*(Sq(Zv11) +
         GG*(Zv10*Zv13 -
             half*(Sq(Zv11) + Sq(Zv12)) -
             Sq(Zv10)*pot1) +
         Sq(Zv11 + Zv21) +
         GG*((Zv10 + Zv20)*(Zv13 + Zv23) -
             half*(Sq(Zv11 + Zv21) + Sq(Zv12 + Zv22)) -
             half*Sq(Zv10 + Zv20)*(pot1 + pot2)) +
         Sq(Zv21) +
         GG*(Zv20*Zv23 -
             half*(Sq(Zv21) + Sq(Zv22)) -
             Sq(Zv20)*pot2))*tnx1/6.0 +
    tl1*(Zv11*Zv12 + (Zv11 + Zv21)*(Zv12 + Zv22) + Zv21*Zv22)*tny1/6.0 +
    tl2*(Sq(Zv01) +
         GG*(Zv00*Zv03 -
             half*(Sq(Zv01) + Sq(Zv02)) -
             Sq(Zv00)*pot0) +
         Sq(Zv01 + Zv21) +
         GG*((Zv00 + Zv20)*(Zv03 + Zv23) -
             half*(Sq(Zv01 + Zv21) + Sq(Zv02 + Zv22)) -
             half*Sq(Zv00 + Zv20)*(pot0 + pot2)) +
         Sq(Zv21) +
         GG*(Zv20*Zv23 -
             half*(Sq(Zv21) + Sq(Zv22)) -
             Sq(Zv20)*pot2))*tnx2/6.0 +
    tl2*(Zv01*Zv02 + (Zv01 + Zv21)*(Zv02 + Zv22) + Zv21*Zv22)*tny2/6.0;
  ResTot1 -= res1;
  pTresTot[n].y = half*ResTot1;
  Wtemp1 -= res1;
  real res2 =
    tl3*(Sq(Zv02) +
         GG*(Zv00*Zv03 -
             half*(Sq(Zv01) + Sq(Zv02)) -
             Sq(Zv00)*pot0) +
         Sq(Zv02 + Zv12) +
         GG*((Zv00 + Zv10)*(Zv03 + Zv13) -
             half*(Sq(Zv01 + Zv11) + Sq(Zv02 + Zv12)) -
             half*Sq(Zv00 + Zv10)*(pot0 + pot1)) +
         Sq(Zv12) +
         GG*(Zv10*Zv13 -
             half*(Sq(Zv11) + Sq(Zv12)) -
             Sq(Zv10)*pot1))*tny3/6.0 +
    tl3*(Zv01*Zv02 + (Zv01 + Zv11)*(Zv02 + Zv12) + Zv11*Zv12)*tnx3/6.0 +
    tl1*(Sq(Zv12) +
         GG*(Zv10*Zv13 -
             half*(Sq(Zv11) + Sq(Zv12)) -
             Sq(Zv10)*pot1) +
         Sq(Zv12 + Zv22) +
         GG*((Zv10 + Zv20)*(Zv13 + Zv23) -
             half*(Sq(Zv11 + Zv21) + Sq(Zv12 + Zv22)) -
             half*Sq(Zv10 + Zv20)*(pot1 + pot2)) +
         Sq(Zv22) +
         GG*(Zv20*Zv23 -
             half*(Sq(Zv21) + Sq(Zv22)) -
             Sq(Zv20)*pot2))*tny1/6.0 +
    tl1*(Zv11*Zv12 + (Zv11 + Zv21)*(Zv12 + Zv22) + Zv21*Zv22)*tnx1/6.0 +
    tl2*(Sq(Zv02) +
         GG*(Zv00*Zv03 -
             half*(Sq(Zv01) + Sq(Zv02)) -
             Sq(Zv00)*pot0) +
         Sq(Zv02 + Zv22) +
         GG*((Zv00 + Zv20)*(Zv03 + Zv23) -
             half*(Sq(Zv01 + Zv21) + Sq(Zv02 + Zv22)) -
             half*Sq(Zv00 + Zv20)*(pot0 + pot2)) +
         Sq(Zv22) +
         GG*(Zv20*Zv23 -
             half*(Sq(Zv21) + Sq(Zv22)) -
             Sq(Zv20)*pot2))*tny2/6.0 +
    tl2*(Zv01*Zv02 + (Zv01 + Zv21)*(Zv02 + Zv22) + Zv21*Zv22)*tnx2/6.0;
  ResTot2 -= res2;
  pTresTot[n].z = half*ResTot2;
  Wtemp2 -= res2;
  real res3 =
    tl3*(Zv03*Zv01 + (Zv03 + Zv13)*(Zv01 + Zv11) + Zv13*Zv11)*tnx3/6.0 +
    tl3*(Zv03*Zv02 + (Zv03 + Zv13)*(Zv02 + Zv12) + Zv13*Zv12)*tny3/6.0 +
    tl1*(Zv13*Zv11 + (Zv13 + Zv23)*(Zv11 + Zv21) + Zv23*Zv21)*tnx1/6.0 +
    tl1*(Zv13*Zv12 + (Zv13 + Zv23)*(Zv12 + Zv22) + Zv23*Zv22)*tny1/6.0 +
    tl2*(Zv23*Zv21 + (Zv23 + Zv03)*(Zv21 + Zv01) + Zv03*Zv01)*tnx2/6.0 +
    tl2*(Zv23*Zv22 + (Zv23 + Zv03)*(Zv22 + Zv02) + Zv03*Zv02)*tny2/6.0;
  ResTot3 -= res3;
  pTresTot[n].w = half*ResTot3;
  Wtemp3 -= res3;

#endif

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real km;
  real ic = one/sqrt(G1*(htilde - two*pot) - alpha);
  real ctilde = one/ic;

  real hc = htilde*ic;
  real uc = utilde*ic;
  real vc = vtilde*ic;
  real ac = alpha*ic;

  // First direction

#ifndef CONTOUR
  nx = half*tnx1;
  ny = half*tny1;
  real tl = tl1;
  wtilde = utilde*nx + vtilde*ny;

  real l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  real l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  real l3 = half*(wtilde - fabs(wtilde));
#else
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;
  real wtilde = utilde*nx + vtilde*ny;

  real l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  real l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  real l3 = -half*(wtilde + fabs(wtilde));
#endif

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
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uc*nx + vc*ny)*ctilde;

#ifndef CONTOUR
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
#else
  l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = -half*(wtilde + fabs(wtilde));
#endif

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
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uc*nx + vc*ny)*ctilde;

#ifndef CONTOUR
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
#else
  l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = -half*(wtilde + fabs(wtilde));
#endif

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

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Wtilde = N*(ResSource + K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // Wtilde = Nm*Wtemp
  real Wtilde0 = invN00*Wtemp0 + invN01*Wtemp1 + invN02*Wtemp2 + invN03*Wtemp3;
  real Wtilde1 = invN10*Wtemp0 + invN11*Wtemp1 + invN12*Wtemp2 + invN13*Wtemp3;
  real Wtilde2 = invN20*Wtemp0 + invN21*Wtemp1 + invN22*Wtemp2 + invN23*Wtemp3;
  real Wtilde3 = invN30*Wtemp0 + invN31*Wtemp1 + invN32*Wtemp2 + invN33*Wtemp3;

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

  // ResN += 0.5*K+*(What - N*(ResSource + K-*What)
  real ResN;

  // First direction
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What00*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3) +
    What01*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2) +
    What02*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2) +
    What03*eulerKMP03(G1, ic, l1l2l3);
  pTresN0[n].x += half*ResN;

  ResN =
    What00*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2) +
    What01*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3) +
    What02*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What03*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresN0[n].y += half*ResN;

  ResN =
    What00*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2) +
    What01*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What02*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3) +
    What03*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresN0[n].z += half*ResN;

  ResN =
    What00*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2) +
    What01*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2) +
    What02*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2) +
    What03*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresN0[n].w += half*ResN;

  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What10*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3) +
    What11*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2) +
    What12*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2) +
    What13*eulerKMP03(G1, ic, l1l2l3);
  pTresN1[n].x += half*ResN;

  ResN =
    What10*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2) +
    What11*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3) +
    What12*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What13*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresN1[n].y += half*ResN;

  ResN =
    What10*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2) +
    What11*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What12*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3) +
    What13*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresN1[n].z += half*ResN;

  ResN =
    What10*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2) +
    What11*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2) +
    What12*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2) +
    What13*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresN1[n].w += half*ResN;

  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What20*eulerKMP00(ac, ic, wtilde, l1l2l3, l1l2, l3) +
    What21*eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2) +
    What22*eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2) +
    What23*eulerKMP03(G1, ic, l1l2l3);
  pTresN2[n].x += half*ResN;

  ResN =
    What20*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2) +
    What21*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3) +
    What22*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What23*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresN2[n].y += half*ResN;

  ResN =
    What20*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2) +
    What21*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2) +
    What22*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3) +
    What23*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresN2[n].z += half*ResN;

  ResN =
    What20*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2) +
    What21*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2) +
    What22*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2) +
    What23*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresN2[n].w += half*ResN;
}

template<ConservationLaw CL>
__host__ __device__
void CalcTotalResNtotSingle(const int n, const real dt,
                            const int3* __restrict__ pTv,
                            const real3* __restrict__ pVz, real3 *pDstate,
                            const real2 *pTn1, const real2 *pTn2,
                            const real2 *pTn3, const real3* __restrict__ pTl,
                            real3 *pResSource, real3 *pTresN0, real3 *pTresN1,
                            real3 *pTresN2, real3 *pTresTot, int nVertex,
                            const real G, const real G1,
                            const real G2, const real iG, const real *pVp)
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

  // State difference between old and RK1
  real dW00 = pDstate[v1].x;
  real dW01 = pDstate[v1].y;
  real dW02 = pDstate[v1].z;
  real dW10 = pDstate[v2].x;
  real dW11 = pDstate[v2].y;
  real dW12 = pDstate[v2].z;
  real dW20 = pDstate[v3].x;
  real dW21 = pDstate[v3].y;
  real dW22 = pDstate[v3].z;

  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  // Adt = |T|/(3*dt)
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*onethird/dt;

  // For N residuals divide by edge length
  real Ail1 = Adt/Tl1;
  real Ail2 = Adt/Tl2;
  real Ail3 = Adt/Tl3;

  // Replace ResN
  pTresN0[n].x = half*pTresN0[n].x + dW00*Ail1;
  pTresN0[n].y = half*pTresN0[n].y + dW01*Ail1;
  pTresN0[n].z = half*pTresN0[n].z + dW02*Ail1;
  pTresN1[n].x = half*pTresN1[n].x + dW10*Ail2;
  pTresN1[n].y = half*pTresN1[n].y + dW11*Ail2;
  pTresN1[n].z = half*pTresN1[n].z + dW12*Ail2;
  pTresN2[n].x = half*pTresN2[n].x + dW20*Ail3;
  pTresN2[n].y = half*pTresN2[n].y + dW21*Ail3;
  pTresN2[n].z = half*pTresN2[n].z + dW22*Ail3;

  // Replace ResTot
  real ResTot0 = pTresTot[n].x + two*Adt*(dW00 + dW10 + dW20);
  real ResTot1 = pTresTot[n].y + two*Adt*(dW01 + dW11 + dW21);
  real ResTot2 = pTresTot[n].z + two*Adt*(dW02 + dW12 + dW22);

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

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real ResSource0 = pResSource[n].x;
  real ResSource1 = pResSource[n].y;
  real ResSource2 = pResSource[n].z;

  ResTot0 += ResSource0;
  ResTot1 += ResSource1;
  ResTot2 += ResSource2;

  real Wtemp0 = ResSource0;
  real Wtemp1 = ResSource1;
  real Wtemp2 = ResSource2;

  // Matrix element K- + K+
  // real kk;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real ctilde = one;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifndef CONTOUR
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
  pTresTot[n].x = half*ResTot0;
  ResTot1 +=
    What20*isoK10(nx, ctilde, wtilde, utilde) +
    What21*isoK11(nx, wtilde, utilde) +
    What22*isoK12(ny, utilde);
  pTresTot[n].y = half*ResTot1;
  ResTot2 +=
    What20*isoK20(ny, ctilde, wtilde, vtilde) +
    What21*isoK21(nx, vtilde) +
    What22*isoK22(ny, wtilde, vtilde);
  pTresTot[n].z = half*ResTot2;

#else

  real res0 =
    tl3*(Zv00*Zv01 + (Zv00 + Zv10)*(Zv01 + Zv11) + Zv10*Zv11)*tnx3/6.0 +
    tl3*(Zv00*Zv02 + (Zv00 + Zv10)*(Zv02 + Zv12) + Zv10*Zv12)*tny3/6.0 +
    tl1*(Zv10*Zv11 + (Zv10 + Zv20)*(Zv11 + Zv21) + Zv20*Zv21)*tnx1/6.0 +
    tl1*(Zv10*Zv12 + (Zv10 + Zv20)*(Zv12 + Zv22) + Zv20*Zv22)*tny1/6.0 +
    tl2*(Zv20*Zv21 + (Zv20 + Zv00)*(Zv21 + Zv01) + Zv00*Zv01)*tnx2/6.0 +
    tl2*(Zv20*Zv22 + (Zv20 + Zv00)*(Zv22 + Zv02) + Zv00*Zv02)*tny2/6.0;
  ResTot0 -= res0;
  pTresTot[n].x = 0.5*ResTot0;
  Wtemp0 -= res0;
  real res1 =
    tl3*(Sq(Zv01) + Sq(Zv00) + Sq(Zv01 + Zv11) + Sq(Zv00 + Zv10) +
         Sq(Zv11) + Sq(Zv10))*tnx3/6.0 +
    tl3*(Zv01*Zv02 + (Zv01 + Zv11)*(Zv02 + Zv12) + Zv11*Zv12)*tny3/6.0 +
    tl1*(Sq(Zv11) + Sq(Zv10) + Sq(Zv11 + Zv21) + Sq(Zv10 + Zv20) +
         Sq(Zv21) + Sq(Zv20))*tnx1/6.0 +
    tl1*(Zv11*Zv12 + (Zv11 + Zv21)*(Zv12 + Zv22) + Zv21*Zv22)*tny1/6.0 +
    tl2*(Sq(Zv01) + Sq(Zv00) + Sq(Zv01 + Zv21) + Sq(Zv00 + Zv20) +
         Sq(Zv21) + Sq(Zv20))*tnx2/6.0 +
    tl2*(Zv01*Zv02 + (Zv01 + Zv21)*(Zv02 + Zv22) + Zv21*Zv22)*tny2/6.0;
  ResTot1 -= res1;
  pTresTot[n].y = 0.5*ResTot1;
  Wtemp1 -= res1;
  real res2 =
    tl3*(Sq(Zv02) + Sq(Zv00) + Sq(Zv02 + Zv12) + Sq(Zv00 + Zv10) +
         Sq(Zv12) + Sq(Zv10))*tny3/6.0 +
    tl3*(Zv01*Zv02 + (Zv01 + Zv11)*(Zv02 + Zv12) + Zv11*Zv12)*tnx3/6.0 +
    tl1*(Sq(Zv12) + Sq(Zv10) + Sq(Zv12 + Zv22) + Sq(Zv10 + Zv20) +
         Sq(Zv22) + Sq(Zv20))*tny1/6.0 +
    tl1*(Zv11*Zv12 + (Zv11 + Zv21)*(Zv12 + Zv22) + Zv21*Zv22)*tnx1/6.0 +
    tl2*(Sq(Zv02) + Sq(Zv00) + Sq(Zv02 + Zv22) + Sq(Zv00 + Zv20) +
         Sq(Zv22) + Sq(Zv20))*tny2/6.0 +
    tl2*(Zv01*Zv02 + (Zv01 + Zv21)*(Zv02 + Zv22) + Zv21*Zv22)*tnx2/6.0;
  ResTot2 -= res2;
  pTresTot[n].z = 0.5*ResTot2;
  Wtemp2 -= res2;

#endif

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real km;
  real ic = one/ctilde;
  real uc = utilde*ic;
  real vc = vtilde*ic;

  // First direction
#ifndef CONTOUR
  nx = half*tnx1;
  ny = half*tny1;
  tl = tl1;
  wtilde = utilde*nx + vtilde*ny;

  // real l1 = min(zero, wtilde + ctilde);
  // real l2 = min(zero, wtilde - ctilde);
  // real l3 = min(zero, wtilde);
  real l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  real l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  real l3 = half*(wtilde - fabs(wtilde));
#else
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;
  real wtilde = utilde*nx + vtilde*ny;

  real l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  real l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  real l3 = -half*(wtilde + fabs(wtilde));
#endif

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
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  // l1 = min(wtilde + ctilde, zero);
  // l2 = min(wtilde - ctilde, zero);
  // l3 = min(wtilde, zero);
#ifndef CONTOUR
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
#else
  l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = -half*(wtilde + fabs(wtilde));
#endif

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
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  // l1 = min(wtilde + ctilde, zero);
  // l2 = min(wtilde - ctilde, zero);
  // l3 = min(wtilde, zero);
#ifndef CONTOUR
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
#else
  l1 = -half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = -half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = -half*(wtilde + fabs(wtilde));
#endif

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

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Wtilde = N*(ResSource + K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // Wtilde = Nm*Wtemp
  real Wtilde0 = invN00*Wtemp0 + invN01*Wtemp1 + invN02*Wtemp2;
  real Wtilde1 = invN10*Wtemp0 + invN11*Wtemp1 + invN12*Wtemp2;
  real Wtilde2 = invN20*Wtemp0 + invN21*Wtemp1 + invN22*Wtemp2;

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

  // ResN += 0.5*K+*(What - N*(ResSource + K-*What)
  real ResN;

  // First direction
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What00*isoKMP00(ic, wtilde, l1l2, l2) +
    What01*isoKMP01(nx, ic, l1l2) +
    What02*isoKMP02(ny, ic, l1l2);
  pTresN0[n].x += half*ResN;

  ResN =
    What00*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2) +
    What01*isoKMP11(nx, uc, l1l2l3, l1l2, l3) +
    What02*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresN0[n].y += half*ResN;

  ResN =
    What00*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2) +
    What01*isoKMP21(nx, ny, vc, l1l2l3, l1l2) +
    What02*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresN0[n].z += half*ResN;

  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What10*isoKMP00(ic, wtilde, l1l2, l2) +
    What11*isoKMP01(nx, ic, l1l2) +
    What12*isoKMP02(ny, ic, l1l2);
  pTresN1[n].x += half*ResN;

  ResN =
    What10*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2) +
    What11*isoKMP11(nx, uc, l1l2l3, l1l2, l3) +
    What12*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresN1[n].y += half*ResN;

  ResN =
    What10*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2) +
    What11*isoKMP21(nx, ny, vc, l1l2l3, l1l2) +
    What12*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresN1[n].z += half*ResN;

  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uc*nx + vc*ny)*ctilde;

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  ResN =
    What20*isoKMP00(ic, wtilde, l1l2, l2) +
    What21*isoKMP01(nx, ic, l1l2) +
    What22*isoKMP02(ny, ic, l1l2);
  pTresN2[n].x += half*ResN;

  ResN =
    What20*isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2) +
    What21*isoKMP11(nx, uc, l1l2l3, l1l2, l3) +
    What22*isoKMP12(nx, ny, uc, l1l2l3, l1l2);
  pTresN2[n].y += half*ResN;

  ResN =
    What20*isoKMP20(ny, ctilde, vc, wtilde, l1l2l3, l1l2) +
    What21*isoKMP21(nx, ny, vc, l1l2l3, l1l2) +
    What22*isoKMP22(ny, vc, l1l2l3, l1l2, l3);
  pTresN2[n].z += half*ResN;
}

template<ConservationLaw CL>
__host__ __device__
void CalcTotalResNtotSingle(const int n, const real dt,
                            const int3* __restrict__ pTv,
                            const real* __restrict__ pVz, real *pDstate,
                            const real2 *pTn1, const real2 *pTn2,
                            const real2 *pTn3, const real3* __restrict__ pTl,
                            real *pResSource, real *pTresN0, real *pTresN1,
                            real *pTresN2, real *pTresTot, int nVertex,
                            const real G, const real G1,
                            const real G2, const real iG, const real *pVp)
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

  real dW0 = pDstate[v1];
  real dW1 = pDstate[v2];
  real dW2 = pDstate[v3];

  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*onethird/dt;

  real Ail1 = Adt/Tl1;
  real Ail2 = Adt/Tl2;
  real Ail3 = Adt/Tl3;

  // Replace ResN
  pTresN0[n] = half*pTresN0[n] + dW0*Ail1;
  pTresN1[n] = half*pTresN1[n] + dW1*Ail2;
  pTresN2[n] = half*pTresN2[n] + dW2*Ail3;

  // Replace ResTot
  real ResTot = pTresTot[n] + two*Adt*(dW0 + dW1 + dW2);

  ResTot += pResSource[n];

  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv0 = pVz[v1];
  real Zv1 = pVz[v2];
  real Zv2 = pVz[v3];

  // Average parameter vector
  real vx = one;
  real vy = zero;

  if (CL == CL_BURGERS) {
    real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
    vx = Z0;
    vy = Z0;
  }

  // Average state at vertices
  real What0 = Zv0;
  real What1 = Zv1;
  real What2 = Zv2;

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real Wtemp = pResSource[n];

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#ifndef CONTOUR
  // First direction
  real nx = half*tnx1*tl1;
  real ny = half*tny1*tl1;

  ResTot += (vx*nx + vy*ny)*What0;

  // Second direction
  nx = half*tnx2*tl2;
  ny = half*tny2*tl2;

  ResTot += (vx*nx + vy*ny)*What1;

  // Third direction
  nx = half*tnx3*tl3;
  ny = half*tny3*tl3;

  ResTot += (vx*nx + vy*ny)*What2;

  pTresTot[n] = half*ResTot;
#else
  real res =
    tl3*half*vx*(What0 + What1)*tnx3 +
    tl3*half*vy*(What0 + What1)*tny3 +
    tl1*half*vx*(What1 + What2)*tnx1 +
    tl1*half*vy*(What1 + What2)*tny1 +
    tl2*half*vx*(What2 + What0)*tnx2 +
    tl2*half*vy*(What2 + What0)*tny2;
  ResTot -= res;
  pTresTot[n] = half*ResTot;
  Wtemp -= res;
#endif

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // First direction
#ifndef CONTOUR
  nx = half*tnx1;
  ny = half*tny1;
  real tl = tl1;

  real l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
#else
  real nx = half*tnx1;
  real ny = half*tny1;
  real tl = tl1;

  real l1 = -half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
#endif

  Wtemp += tl*l1*What0;
  real nm = tl*l1;

  // Second direction
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;

#ifndef CONTOUR
  l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
#else
  l1 = -half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
#endif

  Wtemp += tl*l1*What1;
  nm += tl*l1;

  // Third direction
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;

#ifndef CONTOUR
  l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
#else
  l1 = -half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
#endif

  Wtemp += tl*l1*What2;
  nm += tl*l1;

  real invN = one;
  if (nm != zero) invN /= nm;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Wtilde = N*(ResSource + K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // Wtilde = Nm*Wtemp
  real Wtilde = invN*Wtemp;

  // What = What - Wtilde
  What0 -= Wtilde;
  What1 -= Wtilde;
  What2 -= Wtilde;

  // ResN += 0.5*K+*(What - N*(ResSource + K-*What)
  real ResN;

  // First direction
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  l1 = max(zero, vx*nx + vy*ny);
  ResN = l1*What0;
  pTresN0[n] += half*ResN;

  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;

  l1 = max(zero, vx*nx + vy*ny);
  ResN = l1*What1;
  pTresN1[n] += half*ResN;

  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;

  l1 = max(zero, vx*nx + vy*ny);
  ResN = l1*What2;
  pTresN2[n] += half*ResN;
}

//######################################################################
/*! Kernel calculating space-time N + total residue for all triangles

\param nTriangle Total number of triangles in Mesh
\param dt Time step
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param *pDstate Pointer to state differences (new - old)
\param *pTn1 Pointer first triangle edge normal
\param *pTn2 Pointer second triangle edge normal
\param *pTn3 Pointer third triangle edge normal
\param *pTl Pointer to triangle edge lengths
\param *pVpot Pointer to gravitational potential at vertices
\param *pTresN0 Triangle residue N direction 0 (output)
\param *pTresN1 Triangle residue N direction 1 (output)
\param *pTresN2 Triangle residue N direction 2 (output)
\param *pTresTot Triangle total residue (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param iG 1/G
\param *pVp Pointer to external potential at vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devCalcTotalResNtot(int nTriangle, real dt,
                    const int3* __restrict__ pTv,
                    const realNeq* __restrict__ pVz, realNeq *pDstate,
                    const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
                    const real3* __restrict__  pTl, realNeq *pResSource,
                    realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
                    realNeq *pTresTot, int nVertex,
                    real G, real G1, real G2, real iG, const real *pVp)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcTotalResNtotSingle<CL>(n, dt, pTv, pVz, pDstate,
                               pTn1, pTn2, pTn3, pTl, pResSource,
                               pTresN0, pTresN1, pTresN2,
                               pTresTot, nVertex, G, G1, G2, iG, pVp);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Calculate space-time residue (N + total) for all triangles; result in  \a triangleResidueN and \a triangleResidueTotal.

\param dt Time step*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::CalcTotalResNtot(real dt)
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
      vertexStateDiff->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
      triangleResidueSource->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexParameterVector->TransformToHost();
      vertexStateDiff->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueTotal->TransformToHost();
      triangleResidueSource->TransformToHost();

      cudaFlag = 0;
    }
  }

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pDstate = vertexStateDiff->GetPointer();
  real *pVp = vertexPotential->GetPointer();

  real G = simulationParameter->specificHeatRatio;

  realNeq *pVz = vertexParameterVector->GetPointer();

  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  realNeq *pTresTot = triangleResidueTotal->GetPointer();
  realNeq *pResSource = triangleResidueSource->GetPointer();

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
                                       devCalcTotalResNtot<realNeq, CL>,
                                       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(start, 0) );
#endif
    devCalcTotalResNtot<realNeq, CL><<<nBlocks, nThreads>>>
      (nTriangle, dt, pTv, pVz, pDstate,
       pTn1, pTn2, pTn3, pTl, pResSource,
       pTresN0, pTresN1, pTresN2,
       pTresTot, nVertex, G, G - 1.0, G - 2.0, 1.0/G, pVp);
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
      CalcTotalResNtotSingle<CL>(n, dt, pTv, pVz, pDstate,
                                 pTn1, pTn2, pTn3, pTl, pResSource,
                                 pTresN0, pTresN1, pTresN2,
                                 pTresTot, nVertex, G, G - 1.0, G - 2.0,
                                 1.0/G, pVp);
#ifdef TIME_ASTRIX
    gpuErrchk( cudaEventRecord(stop, 0) );
    gpuErrchk( cudaEventSynchronize(stop) );
#endif
  }

#ifdef TIME_ASTRIX
  gpuErrchk( cudaEventElapsedTime(&elapsedTime, start, stop) );
  WriteProfileFile("CalcTotalResNtot.prof2", nTriangle, elapsedTime, cudaFlag);
#endif

  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexStateDiff->TransformToDevice();
      vertexParameterVector->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
      triangleResidueSource->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexStateDiff->TransformToHost();
      vertexParameterVector->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueTotal->TransformToHost();
      triangleResidueSource->TransformToHost();

      cudaFlag = 0;
    }
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::CalcTotalResNtot(real dt);
template void Simulation<real, CL_BURGERS>::CalcTotalResNtot(real dt);
template void Simulation<real3, CL_CART_ISO>::CalcTotalResNtot(real dt);
template void Simulation<real4, CL_CART_EULER>::CalcTotalResNtot(real dt);

}  // namespace astrix
