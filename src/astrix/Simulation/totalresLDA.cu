// -*-c++-*-
/*! \file totalresII.cu
\brief File containing functions for calculating space-time LDA residue*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

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
\param G2 G - 2*/
//######################################################################

__host__ __device__
void CalcTotalResLDASingle(int n,  
			   const int3* __restrict__ pTv,
			   const real4* __restrict__ pVz, 
			   real4 *pTresLDA0, real4 *pTresLDA1, 
			   real4 *pTresLDA2, real4 *pTresTot,
			   const real2 *pTn1, const real2 *pTn2,
			   const real2 *pTn3, const real3 *pTl, int nVertex, 
			   real G, real G1, real G2)
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
  real absvt = half*(utilde*utilde + vtilde*vtilde);

  real htilde = Z3/Z0;
  real ctilde = G1*(htilde - absvt);
  //real invCtilde = rsqrtf(ctilde);
  //ctilde = one/invCtilde;

  ctilde = sqrt(ctilde);
  real invCtilde = one/ctilde;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  real km;
  //real invCtilde = 
  //  rsqrtf(G1*(htilde - half*(Sq(utilde) + Sq(vtilde))));
  //real ctilde = one/invCtilde;

  real hoverc = htilde*invCtilde;
  real uoverc = utilde*invCtilde;
  real voverc = vtilde*invCtilde;
  real absvtc = G1*absvt*invCtilde;
  
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
  real wtilde = utilde*nx + vtilde*ny;
  
  real l1 = min(zero, wtilde + ctilde);
  real l2 = min(zero, wtilde - ctilde);
  real l3 = min(zero, wtilde);
  //real l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  //real l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  //real l3 = half*(wtilde - fabs(wtilde));

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);
  
  // km000    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  real nm00 = tl*km;
  
  // km001
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  real nm01 = tl*km;
  
  // km002
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  real nm02 = tl*km;
  
  // km003
  km = G1*l1l2l3*Sq(invCtilde);
  real nm03 = tl*km;
  
  // km010
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  real nm10 = tl*km;
  
  // km011
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  real nm11 = tl*km;
  
  // km012
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  real nm12 = tl*km;
  
  // km013
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  real nm13 = tl*km;

  // km020
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  real nm20 = tl*km;
  
  // km021
  km = voverc*nx*l1l2 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2) + nx*ny*l1l2l3;
  real nm21 = tl*km;
  
  // km022
  km = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  real nm22 = tl*km;
  
  // km023
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  real nm23 = tl*km;
  
  // km030
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  real nm30 = tl*km;
  
  // km031
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  real nm31 = tl*km;
  
  // km032
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) - 
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  real nm32 = tl*km;
  
  // km033
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  real nm33 = tl*km;
  
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;

  // Second direction         
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);
  //l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  //l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  //l3 = half*(wtilde - fabs(wtilde));
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km100    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  nm00 += tl*km;
  
  // km101
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  nm01 += tl*km;
  
  // km102
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  nm02 += tl*km;
  
  // km103
  km = G1*l1l2l3*Sq(invCtilde);
  nm03 += tl*km;
  
  // km110
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  nm10 += tl*km;
  
  // km111
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  nm11 += tl*km;
  
  // km112
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  nm12 += tl*km;
  
  // km113
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  nm13 += tl*km;
  
  // km120
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  nm20 += tl*km;
  
  // km121
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  nm21 += tl*km;
  
  // km122
  km = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  nm22 += tl*km;
  
  // km123
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  nm23 += tl*km;
  
  // km130
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  nm30 += tl*km;
  
  // km131
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  nm31 += tl*km;
  
  // km132
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  nm32 += tl*km;
  
  // km133
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  nm33 += tl*km;
  
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  // Third direction
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);
  //l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  //l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  //l3 = half*(wtilde - fabs(wtilde));
 
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km200    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  nm00 += tl*km;
  
  // km201
  km = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  nm01 += tl*km;
  
  // km202
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  nm02 += tl*km;
  
  // km203
  km = G1*l1l2l3*Sq(invCtilde);
  nm03 += tl*km;
  
  // km210
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  nm10 += tl*km;
  
  // km211
  km = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  nm11 += tl*km;
  
  // km212
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  nm12 += tl*km;
  
  // km213
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  nm13 += tl*km;
  
  // km220
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  nm20 += tl*km;
  
  // km221
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  nm21 += tl*km;
  
  // km222
  km = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  nm22 += tl*km;
  
  // km223
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  nm23 += tl*km;
  
  // km230
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  nm30 += tl*km;
  
  // km231
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  nm31 += tl*km;
  
  // km232
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  nm32 += tl*km;
  
  // km233
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
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

  real ResLDA, kp;
  
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  //l1 = max(wtilde + ctilde, zero);
  //l2 = max(wtilde - ctilde, zero);
  //l3 = max(wtilde, zero);    
  
  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));

  // Auxiliary variables
  l1l2l3 = half*(l1+l2)-l3;
  l1l2 = half*(l1-l2);
  
  // kp000    
  kp = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  ResLDA = -kp*Wtilde0;
  
  // kp001
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp002
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp003
  kp = G1*l1l2l3*Sq(invCtilde);
  ResLDA -= kp*Wtilde3;

  pTresLDA0[n].x = ResLDA;
  
  // kp010
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResLDA =-kp*Wtilde0;
  
  // kp011
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp012
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp013
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA0[n].y = ResLDA;
  
  // kp020
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResLDA =-kp*Wtilde0;
  
  // kp021
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResLDA -= kp*Wtilde1;
  
  // kp022
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp023
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA0[n].z = ResLDA;
  
  // kp030
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResLDA =-kp*Wtilde0;
  
  // kp031
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp032
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp033
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA0[n].w = ResLDA;
  
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  // Second direction
  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  //l1 = max(wtilde + ctilde, zero);
  //l2 = max(wtilde - ctilde, zero);
  //l3 = max(wtilde, zero);    
  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp100    
  kp = absvtc*invCtilde*l1l2l3 + 
    l3 - l1l2*wtilde*invCtilde;
  ResLDA = -kp*Wtilde0;
  
  // kp101
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp102
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp103
  kp = G1*l1l2l3*Sq(invCtilde);
  ResLDA -= kp*Wtilde3;
  
  pTresLDA1[n].x = ResLDA;
  
  // kp110
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResLDA =-kp*Wtilde0;
  
  // kp111
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp112
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp113
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA1[n].y = ResLDA;
  
  // kp120
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResLDA = -kp*Wtilde0;
  
  // kp121
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResLDA -= kp*Wtilde1;
  
  // kp122
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp123
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA1[n].z = ResLDA;
  
  // kp130
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResLDA =-kp*Wtilde0;
  
  // kp131
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp132
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp133
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA1[n].w = ResLDA;
  
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  // Third direction
  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  //l1 = max(wtilde + ctilde, zero);
  //l2 = max(wtilde - ctilde, zero);
  //l3 = max(wtilde, zero);
  l1 = half*(wtilde + ctilde + fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde + fabs(wtilde - ctilde));
  l3 = half*(wtilde + fabs(wtilde));
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp200    
  kp = absvtc*invCtilde*l1l2l3 + 
    l3 - l1l2*wtilde*invCtilde;
  ResLDA = -kp*Wtilde0;
  
  // kp201
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp202
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp203
  kp = G1*l1l2l3*Sq(invCtilde);
  ResLDA -= kp*Wtilde3;
  
  pTresLDA2[n].x = ResLDA;
  
  // kp210
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResLDA =-kp*Wtilde0;
  
  // kp211
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp212
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp213
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA2[n].y = ResLDA;
  
  // kp220
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResLDA =-kp*Wtilde0;
  
  // kp221
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResLDA -= kp*Wtilde1;
  
  // kp222
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp223
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA2[n].z = ResLDA;
  
  // kp230
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResLDA = -kp*Wtilde0;
  
  // kp231
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde1;
  
  // kp232
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResLDA -= kp*Wtilde2;
  
  // kp233
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResLDA -= kp*Wtilde3;
  
  pTresLDA2[n].w = ResLDA;
}

__host__ __device__
void CalcTotalResLDASingle(int n,  
			   const int3* __restrict__ pTv,
			   const real* __restrict__ pVz, 
			   real *pTresLDA0, real *pTresLDA1, 
			   real *pTresLDA2, real *pTresTot,
			   const real2 *pTn1, const real2 *pTn2,
			   const real2 *pTn3, const real3 *pTl, int nVertex, 
			   real G, real G1, real G2)
{
  const real zero  = (real) 0.0;
  //const real onethird = (real) (1.0/3.0);
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

  // Parameter vector at vertices: 12 uncoalesced loads
  //real Zv0 = pVz[vs1];
  //real Zv1 = pVz[vs2];
  //real Zv2 = pVz[vs3];

  // Average parameter vector
  //real Z0 = (Zv0 + Zv1 + Zv2)*onethird;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  // Triangle edge lengths
  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  //real tny1 = pTn1[n].y;

  // First direction
  real nx = half*tnx1;
  //real ny = half*tny1;
  real tl = tl1;
  
  real l1 = min(zero, nx);  
  real nm = tl*l1;

  // Second direction         
  real tnx2 = pTn2[n].x;
  //real tny2 = pTn2[n].y;
  nx = half*tnx2;
  //ny = half*tny2;
  tl = tl2;
  
  l1 = min(zero, nx);
  nm += tl*l1;

  // Third direction
  real tnx3 = pTn3[n].x;
  //real tny3 = pTn3[n].y;

  nx = half*tnx3;
  //ny = half*tny3;
  tl = tl3;
  
  l1 = min(zero, nx);
  nm += tl*l1;

  real invN = one;
  if (nm != zero) invN /= nm;

  real Wtilde = invN*pTresTot[n];
 
  // First direction
  real Tnx1 = pTn1[n].x;
  //real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  //ny = half*Tny1;

  l1 = half*(nx + fabs(nx));
  pTresLDA0[n] = -l1*Wtilde;
  
  // Second direction
  real Tnx2 = pTn2[n].x;
  //real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  //ny = half*Tny2;
  
  l1 = half*(nx + fabs(nx));
  pTresLDA1[n] = -l1*Wtilde;
  
  // Third direction
  real Tnx3 = pTn3[n].x;
  //real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  //ny = half*Tny3;

  l1 = half*(nx + fabs(nx));
  pTresLDA2[n] = -l1*Wtilde;
}

//######################################################################
/*! \brief Kernel calculating space-time LDA residue for all triangles

\param nTriangle Total number of triangles in Mesh
\param *tv1 Pointer to first vertex of triangle 
\param *tv2 Pointer to second vertex of triangle 
\param *tv3 Pointer to third vertex of triangle
\param *pVz0 pointer to array of values for zeroth component of parameter vector
\param *pVz1 pointer to array of values for first component of parameter vector 
\param *pVz2 pointer to array of values for second component of parameter vector
\param *pVz3 pointer to array of values for third component of parameter vector
\param *pTresLDA00 Triangle residue LDA direction 0 state 0
\param *pTresLDA01 Triangle residue LDA direction 0 state 1
\param *pTresLDA02 Triangle residue LDA direction 0 state 2
\param *pTresLDA03 Triangle residue LDA direction 0 state 3
\param *pTresLDA10 Triangle residue LDA direction 1 state 0
\param *pTresLDA11 Triangle residue LDA direction 1 state 1
\param *pTresLDA12 Triangle residue LDA direction 1 state 2
\param *pTresLDA13 Triangle residue LDA direction 1 state 3
\param *pTresLDA20 Triangle residue LDA direction 2 state 0
\param *pTresLDA21 Triangle residue LDA direction 2 state 1
\param *pTresLDA22 Triangle residue LDA direction 2 state 2
\param *pTresLDA23 Triangle residue LDA direction 2 state 3
\param *pTresTot0 Triangle total residue state 0
\param *pTresTot1 Triangle total residue state 1
\param *pTresTot2 Triangle total residue state 2
\param *pTresTot3 Triangle total residue state 3
\param *triNx Pointer to x component of triangle edge normals
\param *triNy Pointer to y component of triangle edge normals
\param *pTl Pointer to triangle edge lengths
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats*/
//######################################################################

__global__ void
devCalcTotalResLDA(int nTriangle, const int3* __restrict__ pTv,
		   const realNeq* __restrict__ pVz, 
		   realNeq *pTresLDA0, realNeq *pTresLDA1, realNeq *pTresLDA2,
		   realNeq *pTresTot, const real2 *pTn1, const real2 *pTn2,
		   const real2 *pTn3, const real3 *pTl,
		   int nVertex, real G, real G1, real G2)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcTotalResLDASingle(n, pTv, pVz, pTresLDA0, pTresLDA1, pTresLDA2,
			  pTresTot, pTn1, pTn2, pTn3, pTl, nVertex, G, G1, G2);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate space-time LDA residue for all triangles; result in \a triangleResidueLDA.*/
//######################################################################

void Simulation::CalcTotalResLDA()
{
#ifdef TIME_ASTRIX
  cudaEvent_t start, stop;
  float elapsedTime = 0.0f;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
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
  
  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);
  
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
				       devCalcTotalResLDA, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devCalcTotalResLDA<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVz,
       pTresLDA0, pTresLDA1, pTresLDA2, pTresTot,
       pTn1, pTn2, pTn3, pTl, nVertex, 
       specificHeatRatio, 
       specificHeatRatio - 1.0,
       specificHeatRatio - 2.0);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  } else {
#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    for (int n = 0; n < nTriangle; n++) 
      CalcTotalResLDASingle(n, pTv, pVz,
			    pTresLDA0, pTresLDA1, pTresLDA2, pTresTot,
			    pTn1, pTn2, pTn3, pTl, nVertex, 
			    specificHeatRatio, specificHeatRatio - 1.0,
			    specificHeatRatio - 2.0);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
  }
  
#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Kernel: devCalcTotalResII, # of elements: "
	    << nTriangle << ", elapsed time: " << elapsedTime << std::endl;
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


}
