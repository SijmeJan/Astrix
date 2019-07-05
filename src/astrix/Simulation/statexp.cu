// -*-c++-*-
/*! \file statexp.cu
\brief File containing functions for stationary extrapolation

*/ /* \section LICENSE
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
#include "./contour.h"
#include "../Common/profile.h"
#include "./Param/simulationparameter.h"
#include "./triangle_normals.h"

namespace astrix {

//######################################################################
/*! \brief Calculate spatial residue at triangle n

\param n Triangle to consider
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param Tn Class containing triangle normals
\param *pResSource Pointer to source term contribution to residual
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices
\param cs0 Soundspeed at x=0 (cylindrical isothermal only)
\param cspow Power law index of soundspeed (cylindrical isothermal only)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)
\param *pVc Pointer to vertex coordinates*/
//######################################################################


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//-----------------------------------------------------------------------------
// Systems of 4 equations: Cartesian hydrodynamics
//-----------------------------------------------------------------------------
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template<ConservationLaw CL>
__host__ __device__
void StatExpSingle(int n, const int3 *pTv, real4 *pVz,
                   TriangleNormals Tn,
                   real4 *pResSource,
                   real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
                   int nVertex, real G, real G1, real G2,
                   const real *pVp, const real cs0, const real cspow,
                   const real frameAngularVelocity, const real2 *pVc)
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

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real htilde = Z3/Z0;
  real alpha  = G1*half*(Sq(utilde) + Sq(vtilde)) - G1*pot;

  //######################################################################
  // Assume hydrostatic stationary state, pressure balancing source term
  //######################################################################

  real tl1 = Tn.pTl[n].x;
  real tl2 = Tn.pTl[n].y;
  real tl3 = Tn.pTl[n].z;

  // Triangle area
  real s = half*(tl1 + tl2 + tl3);
  real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

  // Average source term = -ResSource/Area
  real S = -pResSource[n].y/area;

  // p = (g-1)*(h - 0.5*(u^2 + v^2) - pot)*rho/g
  real p0 = G1*(Zv03*Zv00 - half*(Zv01*Zv01 + Zv02*Zv02) - pot0*Sq(Zv00))/G;
  // Extrapolate pressure for hydrostatic balance
  real p1 = p0 + (pVc[v2].x - pVc[v1].x)*S;
  real p2 = p0 + (pVc[v3].x - pVc[v1].x)*S;

  real dp1 = p1 -
    G1*(Zv13*Zv10 - half*(Zv11*Zv11 + Zv12*Zv12) - pot1*Sq(Zv10))/G;
  real dp2 = p2 -
    G1*(Zv23*Zv20 - half*(Zv21*Zv21 + Zv22*Zv22) - pot2*Sq(Zv20))/G;

  // Adjust parameter vector:
  // Z3  = sqrt(rho)*(u*u/2 + v*v/2 + pot + G*p/(G-1)/rho)
  // dZ3 = sqrt(rho)*G*(p_new - p_old)/(G-1)/rho
  //Zv13 += G*dp1/(Zv10*G1);
  //Zv23 += G*dp2/(Zv20*G1);

  //Z3 = (Zv03 + Zv13 + Zv23)*onethird;

  // Average state at vertices
  real What00 = two*Z0*Zv00;
  real What01 = Z1*Zv00 + Z0*Zv01;
  real What02 = Z2*Zv00 + Z0*Zv02;
  real What03 = (Z3*Zv00 + G1*Z1*Zv01 + G1*Z2*Zv02 +
                 Z0*Zv03 + two*G1*pot*Z0*Zv00)/G;
  real What10 = two*Z0*Zv10;
  real What11 = Z1*Zv10 + Z0*Zv11;
  real What12 = Z2*Zv10 + Z0*Zv12;
  real What13 = (Z3*Zv10 + G1*Z1*Zv11 + G1*Z2*Zv12 +
                 Z0*Zv13 + two*G1*pot*Z0*Zv10)/G;
  real What20 = two*Z0*Zv20;
  real What21 = Z1*Zv20 + Z0*Zv21;
  real What22 = Z2*Zv20 + Z0*Zv22;
  real What23 = (Z3*Zv20 + G1*Z1*Zv21 + G1*Z2*Zv22 +
                 Z0*Zv23 + two*G1*pot*Z0*Zv20)/G;

  real Wtemp0 = zero;
  real Wtemp1 = zero;
  real Wtemp2 = zero;
  real Wtemp3 = zero;

  real tnx1 = Tn.pTn1[n].x;
  real tnx2 = Tn.pTn2[n].x;
  real tnx3 = Tn.pTn3[n].x;
  real tny1 = Tn.pTn1[n].y;
  real tny2 = Tn.pTn2[n].y;
  real tny3 = Tn.pTn3[n].y;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K+*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real km;
  real ic = one/sqrt(G1*(htilde - two*pot) - alpha);
  real ctilde = one/ic;

  real hc = htilde*ic;
  real uc = utilde*ic;
  real vc = vtilde*ic;
  real ac = alpha*ic;

  // First direction

  real nx = tnx1;
  real ny = tny1;
  real tl = half*tl1;
  real wtilde = utilde*nx + vtilde*ny;

  real l1 = max(zero, wtilde + ctilde);
  real l2 = max(zero, wtilde - ctilde);
  real l3 = max(zero, wtilde);

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

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

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

  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);

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

  // PhiN = Kp*(What - Ninv*Sum(Km*What))
  real ResN, kp;

  real Tnx1 = Tn.pTn1[n].x;
  real Tnx2 = Tn.pTn2[n].x;
  real Tnx3 = Tn.pTn3[n].x;
  real Tny1 = Tn.pTn1[n].y;
  real Tny2 = Tn.pTn2[n].y;
  real Tny3 = Tn.pTn3[n].y;

  real4 ResBefore0 = pTresN0[n];
  real4 ResBefore1 = pTresN1[n];
  real4 ResBefore2 = pTresN2[n];

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
  ResN = kp*What00;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What01;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What02;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN += kp*What03;

  pTresN0[n].x -= half*ResN;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What00;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What01;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What02;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What03;

  pTresN0[n].y -= half*ResN;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What00;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What01;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN += kp*What02;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What03;

  pTresN0[n].z -= half*ResN;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN = kp*What00;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN += kp*What01;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN += kp*What02;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN += kp*What03;

  pTresN0[n].w -= half*ResN;

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
  ResN = kp*What10;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What11;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What12;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN += kp*What13;

  pTresN1[n].x -= half*ResN;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What10;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What11;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What12;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What13;

  pTresN1[n].y -= half*ResN;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What10;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What11;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN += kp*What12;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What13;

  pTresN1[n].z -= half*ResN;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN = kp*What10;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN += kp*What11;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN += kp*What12;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN += kp*What13;

  pTresN1[n].w -= half*ResN;

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
  ResN = kp*What20;

  kp = eulerKMP01(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What21;

  kp = eulerKMP02(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What22;

  kp = eulerKMP03(G1, ic, l1l2l3);
  ResN += kp*What23;

  pTresN2[n].x -= half*ResN;

  kp = eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What20;

  kp = eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What21;

  kp = eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What22;

  kp = eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  ResN += kp*What23;

  pTresN2[n].y -= half*ResN;

  kp = eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What20;

  kp = eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2);
  ResN += kp*What21;

  kp = eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3);
  ResN += kp*What22;

  kp = eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  ResN += kp*What23;

  pTresN2[n].z -= half*ResN;

  kp = eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2);
  ResN = kp*What20;

  kp = eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2);
  ResN += kp*What21;

  kp = eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2);
  ResN += kp*What22;

  kp = eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  ResN += kp*What23;

  pTresN2[n].w -= half*ResN;

  real dvxdx = (pVz[v1].y*Tnx1*tl1/pVz[v1].x +
                pVz[v2].y*Tnx2*tl2/pVz[v2].x +
                pVz[v3].y*Tnx3*tl3/pVz[v3].x)/area;
  real dvydy = (pVz[v1].z*Tny1*tl1/pVz[v1].x +
                pVz[v2].z*Tny2*tl2/pVz[v2].x +
                pVz[v3].z*Tny3*tl3/pVz[v3].x)/area;
  if (abs(dvxdx + dvydy) > 10.0) {
    // No stationary extrapolation for strong velocity divergence
    pTresN0[n] = ResBefore0;
    pTresN1[n] = ResBefore1;
    pTresN2[n] = ResBefore2;
  }

}



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//-----------------------------------------------------------------------------
//! Systems of 3 equations: Cartesian and cylindrical isothermal hydrodynamics
//-----------------------------------------------------------------------------
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



template<ConservationLaw CL>
__host__ __device__
void StatExpSingle(int n, const int3 *pTv, real3 *pVz,
                   TriangleNormals Tn,
                   real3 *pResSource,
                   real3 *pTresN0, real3 *pTresN1, real3 *pTresN2,
                   int nVertex, real G, real G1, real G2,
                   const real *pVp, const real cs0, const real cspow,
                   const real frameAngularVelocity, const real2 *pVc)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
  const real half  = (real) 0.5;
  const real one = (real) 1.0;
  const real two = (real) 2.0;
  const real three = (real) 3.0;
  const real four = (real) 4.0;

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

  // Average sound speed (note: cspow only supported in cylindrical geometry!)
  real ctilde = cs0;
  // Average radius triangle
  real r2 = one;
  real ir2 = one;
  real Omega = zero;

  // Average parameter vector
  real Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  real Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  real Z2 = (Zv02 + Zv12 + Zv22)*onethird;

  real tl1 = Tn.pTl[n].x;
  real tl2 = Tn.pTl[n].y;
  real tl3 = Tn.pTl[n].z;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;

  // Stationary state estimate
  real xa = one;
  real xb = one;
  real xc = one;

  if (CL == CL_CYL_ISO) {
    //######################################################################
    // Take isothermal cylindrical as a special case, as stationary flow in
    // the limit of no radial mass flux becomes underdefined. In stead, take
    // v_y to be the current vertex values and calculate the density gradient
    // from the x-momentum source term.
    //######################################################################

    xa = pVc[v1].x;
    xb = pVc[v2].x;
    xc = pVc[v3].x;

    // Average sound speed
    ctilde = cs0*exp(cspow*(xa + xb + xc)*onethird);

    // Average radius
    r2 = exp(two*(xa + xb + xc)*onethird);
    ir2 = one/r2;
    Omega = frameAngularVelocity;

    // Statexp source: centrifugal + central gravity
    real S1 = (Zv02*Zv02*exp(-four*xa) - Zv00*Zv00*exp(-three*xa) +
               Zv12*Zv12*exp(-four*xb) - Zv10*Zv10*exp(-three*xb) +
               Zv22*Zv22*exp(-four*xc) - Zv20*Zv20*exp(-three*xc))*onethird;

    // Make sure extrapolated densities are physical
    real alpha = one;
    real c1 = cs0*exp(cspow*xa);
    real c2 = cs0*exp(cspow*xb);
    real c3 = cs0*exp(cspow*xc);

    real Sdx2 = S1*(pVc[v2].x - pVc[v1].x);
    real Sdx3 = S1*(pVc[v3].x - pVc[v1].x);
    real d2 = Zv00*Zv00*c1*c1 + Sdx2;
    real d3 = Zv00*Zv00*c1*c1 + Sdx3;
    if (d2 < zero || d3 < zero) {
      d2 = zero;
      d3 = zero;
      alpha = zero;
    }

    real vx = Zv11/Zv10;
    real vy = Zv12/Zv10;
    Zv10 = alpha*sqrt(d2)/c2;
    Zv11 = alpha*Zv10*vx;
    Zv12 = alpha*Zv10*vy;

    vx = Zv21/Zv20;
    vy = Zv22/Zv20;
    Zv20 = alpha*sqrt(d3)/c3;
    Zv21 = alpha*Zv20*vx;
    Zv22 = alpha*Zv20*vy;

    Zv00 = alpha*Zv00;
    Zv01 = alpha*Zv01;
    Zv02 = alpha*Zv02;
  } else {
    //######################################################################
    // General case: Take stationary state to be given by dU/dx = A^(-1)S
    //######################################################################

    // Triangle area
    real s = half*(tl1 + tl2 + tl3);
    real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

    // Average source term = -ResSource/Area
    real3 S = pResSource[n];
    S.x /= -area;
    S.y /= -area;
    S.z /= -area;

    real dx = pVc[v2].x - pVc[v1].x;
    real fac = one/(Sq(ctilde) - Sq(utilde));

    Zv10 = sqrt(Zv00*Zv00 + dx*(S.y - two*utilde*S.x)*fac);
    Zv11 = (Zv00*Zv01 + dx*S.x)/Zv10;
    Zv12 = (Zv00*Zv02 + dx*((S.y - two*utilde*S.x)*vtilde*fac +
                            (S.z - vtilde*S.x)/utilde))/Zv10;

    dx = pVc[v3].x - pVc[v1].x;

    Zv20 = sqrt(Zv00*Zv00 + dx*(S.y - two*utilde*S.x)*fac);
    Zv21 = (Zv00*Zv01 + dx*S.x)/Zv10;
    Zv22 = (Zv00*Zv02 + dx*((S.y - two*utilde*S.x)*vtilde*fac +
                            (S.z - vtilde*S.x)/utilde))/Zv10;
  }

  Z0 = (Zv00 + Zv10 + Zv20)*onethird;
  Z1 = (Zv01 + Zv11 + Zv21)*onethird;
  Z2 = (Zv02 + Zv12 + Zv22)*onethird;

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

  real tnx1 = Tn.pTn1[n].x;
  real tnx2 = Tn.pTn2[n].x;
  real tnx3 = Tn.pTn3[n].x;
  real tny1 = Tn.pTn1[n].y;
  real tny2 = Tn.pTn2[n].y;
  real tny3 = Tn.pTn3[n].y;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K+*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real Wtemp0 = zero;
  real Wtemp1 = zero;
  real Wtemp2 = zero;

  real km;
  real ic = one/ctilde;

  real uc = utilde*ic;
  real vc = vtilde*ic;

  // First direction
  real nx = tnx1;
  real ny = tny1;
  real tl = half*tl1;
  real wtilde = utilde*nx + vtilde*ny*ir2;

  real l1 = max(zero, wtilde + ctilde - Omega*ny);
  real l2 = max(zero, wtilde - ctilde - Omega*ny);
  real l3 = max(zero, wtilde - Omega*ny);

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
  km = isoKMP02(ny*ir2, ic, l1l2);
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
  km = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What02;
  real nm12 = tl*km;

  // km[0][2][0]
  km = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What00;
  real nm20 = tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What01;
  real nm21 = tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What02;
  real nm22 = tl*km;

  // Second direction
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = max(wtilde + ctilde - Omega*ny, zero);
  l2 = max(wtilde - ctilde - Omega*ny, zero);
  l3 = max(wtilde - Omega*ny, zero);

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
  km = isoKMP02(ny*ir2, ic, l1l2);
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
  km = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What12;
  nm12 += tl*km;

  // km[0][2][0]
  km = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What10;
  nm20 += tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What11;
  nm21 += tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  Wtemp2 += tl*km*What12;
  nm22 += tl*km;

  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = max(wtilde + ctilde - Omega*ny, zero);
  l2 = max(wtilde - ctilde - Omega*ny, zero);
  l3 = max(wtilde - Omega*ny, zero);

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
  km = isoKMP02(ny*ir2, ic, l1l2);
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
  km = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  Wtemp1 += tl*km*What22;
  nm12 += tl*km;

  // km[0][2][0]
  km = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  Wtemp2 += tl*km*What20;
  nm20 += tl*km;

  // km[0][2][1]
  km = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  Wtemp2 += tl*km*What21;
  nm21 += tl*km;

  // km[0][2][2]
  km = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
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

  // PhiN = Kp*(What - Ninv*Sum(Km*What))
  real ResN, kp;

  real Tnx1 = Tn.pTn1[n].x;
  real Tnx2 = Tn.pTn2[n].x;
  real Tnx3 = Tn.pTn3[n].x;
  real Tny1 = Tn.pTn1[n].y;
  real Tny2 = Tn.pTn2[n].y;
  real Tny3 = Tn.pTn3[n].y;

  real3 ResBefore0 = pTresN0[n];
  real3 ResBefore1 = pTresN1[n];
  real3 ResBefore2 = pTresN2[n];

  // First direction
  nx = Tnx1;
  ny = Tny1;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = max(wtilde + ctilde - Omega*ny, zero);
  l2 = max(wtilde - ctilde - Omega*ny, zero);
  l3 = max(wtilde - Omega*ny, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN = kp*What00;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN += kp*What01;

  // kp[0][0][2]
  kp = isoKMP02(ny*ir2, ic, l1l2);
  ResN += kp*What02;

  pTresN0[n].x -= half*ResN;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What00;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What01;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  ResN += kp*What02;

  pTresN0[n].y -= half*ResN;

  // kp[0][2][0]
  kp = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What00;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  ResN += kp*What01;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  ResN += kp*What02;

  pTresN0[n].z -= half*ResN;

  // Second direction
  nx = Tnx2;
  ny = Tny2;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = max(wtilde + ctilde - Omega*ny, zero);
  l2 = max(wtilde - ctilde - Omega*ny, zero);
  l3 = max(wtilde - Omega*ny, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN = kp*What10;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN += kp*What11;

  // kp[0][0][2]
  kp = isoKMP02(ny*ir2, ic, l1l2);
  ResN += kp*What12;


  pTresN1[n].x -= half*ResN;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What10;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What11;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  ResN += kp*What12;

  pTresN1[n].y -= half*ResN;

  // kp[0][2][0]
  kp = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What10;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  ResN += kp*What11;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  ResN += kp*What12;

  pTresN1[n].z -= half*ResN;

  // Third direction
  nx = Tnx3;
  ny = Tny3;
  wtilde = (uc*nx + vc*ny*ir2)*ctilde;

  l1 = max(wtilde + ctilde - Omega*ny, zero);
  l2 = max(wtilde - ctilde - Omega*ny, zero);
  l3 = max(wtilde - Omega*ny, zero);

  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);

  // kp[0][0][0]
  kp = isoKMP00(ic, wtilde, l1l2, l2);
  ResN = kp*What20;

  // kp[0][0][1]
  kp = isoKMP01(nx, ic, l1l2);
  ResN += kp*What21;

  // kp[0][0][2]
  kp = isoKMP02(ny*ir2, ic, l1l2);
  ResN += kp*What22;

  pTresN2[n].x -= half*ResN;

  // kp[0][1][0]
  kp = isoKMP10(nx, ctilde, uc, wtilde, l1l2l3, l1l2);
  ResN = kp*What20;

  // kp[0][1][1]
  kp = isoKMP11(nx, uc, l1l2l3, l1l2, l3);
  ResN += kp*What21;

  // kp[0][1][2]
  kp = isoKMP12(nx, ny*ir2, uc, l1l2l3, l1l2);
  ResN += kp*What22;

  pTresN2[n].y -= half*ResN;

  // kp[0][2][0]
  kp = isoKMP20(ny*r2, ctilde, vc, wtilde, l1l2l3, l1l2);
  ResN = kp*What20;

  // kp[0][2][1]
  kp = isoKMP21(nx, ny*r2, vc, l1l2l3, l1l2);
  ResN += kp*What21;

  // kp[0][2][2]
  kp = isoKMP22(ny, vc*ir2, l1l2l3, l1l2, l3);
  ResN += kp*What22;

  pTresN2[n].z -= half*ResN;

  // It is not always wise to do stationary extrapolation
  /*
  if (CL == CL_CYL_ISO) {
    // Not stationary extrapolation too close to the planet
    real rp2 = r2 + one - two*sqrt(r2)*cos(pVc[v3].y - M_PI);
    if (rp2 < 0.01) {
      pTresN0[n] = ResBefore0;
      pTresN1[n] = ResBefore1;
      pTresN2[n] = ResBefore2;
    }
  }
  */

  // Triangle area
  real s = half*(tl1 + tl2 + tl3);
  real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

  real dvxdx = (pVz[v1].y*Tnx1*tl1/pVz[v1].x +
                pVz[v2].y*Tnx2*tl2/pVz[v2].x +
                pVz[v3].y*Tnx3*tl3/pVz[v3].x)/area;
  real dvydy = (pVz[v1].z*Tny1*tl1/pVz[v1].x +
                pVz[v2].z*Tny2*tl2/pVz[v2].x +
                pVz[v3].z*Tny3*tl3/pVz[v3].x)/area;
  if (abs(dvxdx + dvydy) > 10.0) {
    // No stationary extrapolation for strong velocity divergence
    pTresN0[n] = ResBefore0;
    pTresN1[n] = ResBefore1;
    pTresN2[n] = ResBefore2;
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//-----------------------------------------------------------------------------
//! Systems of 1 equation: Linear advection and Burgers equation
//-----------------------------------------------------------------------------
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


template<ConservationLaw CL>
__host__ __device__
void StatExpSingle(int n, const int3 *pTv, real *pVz,
                   TriangleNormals Tn,
                   real *pResSource,
                   real *pTresN0, real *pTresN1, real *pTresN2,
                   int nVertex, real G, real G1, real G2,
                   const real *pVp, const real cs0, const real cspow,
                   const real frameAngularVelocity, const real2 *pVc)
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
  real vx = one;
  real vy = zero;
  if (CL == CL_BURGERS) {
    const real onethird = (real) (1.0/3.0);
    real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
    vx = Z0;
    vy = Z0;
  }

  real tl1 = Tn.pTl[n].x;
  real tl2 = Tn.pTl[n].y;
  real tl3 = Tn.pTl[n].z;

  // Triangle area
  real s = half*(tl1 + tl2 + tl3);
  real area = sqrt(s*(s - tl1)*(s - tl2)*(s - tl3));

  // Average source term = -ResSource/Area
  real S = -pResSource[n]/area;

  // Approximation to stationary solution within triangle
  real What0 = Zv0;
  real What1 = Zv0 + S*(pVc[v2].x - pVc[v1].x)/vx;
  real What2 = Zv0 + S*(pVc[v3].x - pVc[v1].x)/vx;

  real tnx1 = Tn.pTn1[n].x;
  real tnx2 = Tn.pTn2[n].x;
  real tnx3 = Tn.pTn3[n].x;
  real tny1 = Tn.pTn1[n].y;
  real tny2 = Tn.pTn2[n].y;
  real tny3 = Tn.pTn3[n].y;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K+*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real Wtemp = zero;

  // First direction
  real nx = tnx1;
  real ny = tny1;
  real tl = half*tl1;

  real l1 = max(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What0;
  real nm = tl*l1;

  // Second direction
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;

  l1 = max(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What1;
  nm += tl*l1;

  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;

  l1 = max(zero, vx*nx + vy*ny);

  Wtemp += tl*l1*What2;
  nm += tl*l1;

  real invN = one;
  if (nm != zero) invN /= nm;

  real Wtilde = invN*Wtemp;

  What0 -= Wtilde;
  What1 -= Wtilde;
  What2 -= Wtilde;

  real ResN;

  real Tnx1 = Tn.pTn1[n].x;
  real Tnx2 = Tn.pTn2[n].x;
  real Tnx3 = Tn.pTn3[n].x;
  real Tny1 = Tn.pTn1[n].y;
  real Tny2 = Tn.pTn2[n].y;
  real Tny3 = Tn.pTn3[n].y;

  nx = Tnx1;
  ny = Tny1;
  l1 = max(zero, vx*nx + vy*ny);

  ResN = l1*What0;          // ResN = K^+*What = K^+*(What - N*Sum(K^-*What))

  pTresN0[n] -= half*ResN;

  // Second direction
  nx = Tnx2;
  ny = Tny2;
  l1 = max(zero, vx*nx + vy*ny);

  ResN = l1*What1;

  pTresN1[n] -= half*ResN;

  // Third direction
  nx = Tnx3;
  ny = Tny3;

  l1 = max(zero, vx*nx + vy*ny);

  ResN = l1*What2;   // [ResN] = phi/x

  pTresN2[n] -= half*ResN;
}

//######################################################################
/*! \brief Kernel adjusting N residuals for stationary flows

\param Ms Object containing mesh size (number of vertices, triangles,edges)
\param *pTv Pointer to triangle vertices
\param *pVz Pointer to parameter vector
\param Tn Class containing triangle normals
\param *pResSource Pointer to residual due to source terms
\param *pTresN0 Triangle residue N direction 0
\param *pTresN1 Triangle residue N direction 1
\param *pTresN2 Triangle residue N direction 2
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param *pVp Pointer to external potential at vertices
\param cs0 Soundspeed at x=0 (cylindrical isothermal only)
\param cspow Power law index of soundspeed (cylindrical isothermal only)
\param frameAngularVelocity Angular velocity coordinate frame (cylindrical geometry)
\param *pVc Pointer to vertex coordinates*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devStatExp(MeshSize Ms, const int3 *pTv, realNeq *pVz,
           TriangleNormals Tn,
           realNeq *pResSource,
           realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
           real G, real G1, real G2,
           const real *pVp, const real cs0, const real cspow,
           const real frameAngularVelocity, const real2 *pVc)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;


  while (n < Ms.nTriangle) {
    StatExpSingle<CL>(n, pTv, pVz, Tn,
                      pResSource,
                      pTresN0, pTresN1, pTresN2,
                      Ms.nVertex, G, G1, G2, pVp, cs0, cspow,
                      frameAngularVelocity, pVc);

    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Adjust spatial N residuals to reduce unwanted numerical diffusion in stationary flows. */
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::StatExp()
{
  realNeq *pResSource = triangleResidueSource->GetPointer();
  realNeq *pVz = vertexParameterVector->GetPointer();
  real *pVp = vertexPotential->GetPointer();
  real G = simulationParameter->specificHeatRatio;

  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);

  const int3 *pTv = mesh->TriangleVerticesData();

  MeshSize Ms(mesh);
  TriangleNormals Tn(mesh);

  const real2 *pVc = mesh->VertexCoordinatesData();

  const real Omega = simulationParameter->frameAngularVelocity;
  const real cs0 = simulationParameter->soundspeed0;
  const real cspow = simulationParameter->soundspeedPower;

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devStatExp<realNeq, CL>,
                                       (size_t) 0, 0);

    devStatExp<realNeq, CL><<<nBlocks, nThreads>>>
      (Ms, pTv, pVz, Tn,
       pResSource,
       pTresN0, pTresN1, pTresN2,
       G, G - 1.0, G - 2.0, pVp,
       cs0, cspow, Omega, pVc);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < Ms.nTriangle; n++)
      StatExpSingle<CL>(n, pTv, pVz, Tn,
                        pResSource,
                        pTresN0, pTresN1, pTresN2,
                        Ms.nVertex, G, G - 1.0, G - 2.0,
                        pVp, cs0, cspow, Omega, pVc);
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::StatExp();
template void Simulation<real, CL_BURGERS>::StatExp();
template void Simulation<real3, CL_CART_ISO>::StatExp();
template void Simulation<real3, CL_CYL_ISO>::StatExp();
template void Simulation<real4, CL_CART_EULER>::StatExp();

}  // namespace astrix
