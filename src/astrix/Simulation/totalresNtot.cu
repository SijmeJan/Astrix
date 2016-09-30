// -*-c++-*-
/*! \file totalres.cu
\brief File containing functions for calculating space-time N + total residue*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

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
\param *pVpot Pointer to gravitational potential at vertices
\param *pTresN0 Triangle residue N direction 0 (output)
\param *pTresN1 Triangle residue N direction 1 (output)
\param *pTresN2 Triangle residue N direction 2 (output)
\param *pTresTot Triangle total residue (output)
\param nVertex Total number of vertices in Mesh
\param G Ratio of specific heats
\param G1 G - 1
\param G2 G - 2
\param iG 1/G*/
//######################################################################
  
__host__ __device__
void CalcTotalResNtotSingle(const int n, const real dt,
			    const int3* __restrict__ pTv, 
			    const real4* __restrict__ pVz, real4 *pDstate,
			    const real2 *pTn1, const real2 *pTn2,
			    const real2 *pTn3, const real3* __restrict__ pTl,
			    real *pVpot, real4 *pTresN0, real4 *pTresN1,
			    real4 *pTresN2, real4 *pTresTot, int nVertex,
			    const real G, const real G1,
			    const real G2, const real iG)
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
  real What03 = (Z3*Zv00 + G1*(Z1*Zv01 + Z2*Zv02) + Z0*Zv03)*iG;      
  real What10 = two*Z0*Zv10;
  real What11 = Z1*Zv10 + Z0*Zv11;
  real What12 = Z2*Zv10 + Z0*Zv12;
  real What13 = (Z3*Zv10 + G1*(Z1*Zv11 + Z2*Zv12) + Z0*Zv13)*iG;      
  real What20 = two*Z0*Zv20;
  real What21 = Z1*Zv20 + Z0*Zv21;
  real What22 = Z2*Zv20 + Z0*Zv22;
  real What23 = (Z3*Zv20 + G1*(Z1*Zv21 + Z2*Zv22) + Z0*Zv23)*iG;      

  // Source term residual
  real rhoAve  = Z0*Z0;
  real momxAve = Z0*Z1;
  real momyAve = Z0*Z2;

  real tnx1 = pTn1[n].x;
  real tny1 = pTn1[n].y;
  real tnx2 = pTn2[n].x;
  real tny2 = pTn2[n].y;
  real tnx3 = pTn3[n].x;
  real tny3 = pTn3[n].y;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real dPotdx =
    tnx1*tl1*pVpot[v1] + tnx2*tl2*pVpot[v2] + tnx3*tl3*pVpot[v3];
  real dPotdy =
    tny1*tl1*pVpot[v1] + tny2*tl2*pVpot[v2] + tny3*tl3*pVpot[v3];

  // -integral(Source*dS)
  real ResSource0 = zero;
  real ResSource1 = half*rhoAve*dPotdx;
  real ResSource2 = half*rhoAve*dPotdy;
  real ResSource3 = -half*momxAve*dPotdx - half*momyAve*dPotdy;
  
  ResTot0 += ResSource0;
  ResTot1 += ResSource1;
  ResTot2 += ResSource2;
  ResTot3 += ResSource3;
  
  // Matrix element K- + K+
  real kk;

  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real htilde = Z3/Z0;
  real absvt = half*(Sq(utilde) + Sq(vtilde));
  real absvtc = G1*absvt;
    
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme 
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  // First direction
  real nx = half*tnx1*tl1;
  real ny = half*tny1*tl1;
  real wtilde = utilde*nx + vtilde*ny;
    
  // kk000    
  //kk = 0.0f;
  //ResTot0 +=  t_l*kk*What00;
    
  // kk001
  kk = nx;
  ResTot0 += kk*What01;
  
  // kk002
  kk = ny;
  ResTot0 += kk*What02;
  
  // kk003
  //kk = 0.0f;
  //ResTot0 += tl*kk*What03;
  
  // kk010
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 +=  kk*What00;
  
  // kk011
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What01;
  
  // kk012
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What02;
  
  // kk013
  kk = G1*nx;
  ResTot1 += kk*What03;
  
  // kk020
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 +=  kk*What00;
  
  // kk021
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What01;
  
  // kk022
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What02;
  
  // kk023
  kk = G1*ny;
  ResTot2 += kk*What03;
  
  // kk030
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 +=  kk*What00;
  
  // kk031
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What01;
  
  // kk032
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What02;
  
  // kk033
  kk = G*wtilde;
  ResTot3 += kk*What03;
  
  // Second direction
  nx = half*tnx2*tl2;
  ny = half*tny2*tl2;
  wtilde = utilde*nx + vtilde*ny;
  
  // kk100    
  //kk = 0.0f;
  //ResTot0 += tl*kk*What10];
  
  // kk101
  kk = nx;
  ResTot0 += kk*What11;
  
  // kk102
  kk = ny;
  ResTot0 += kk*What12;
  
  // kk103
  //kk = 0.0f;
  //ResTot0 += tl*kk*What13];
  
  // kk110
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 += kk*What10;
  
  // kk111
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What11;
  
  // kk112
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What12;
  
  // kk113
  kk = G1*nx;
  ResTot1 += kk*What13;
  
  // kk120
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 += kk*What10;
  
  // kk121
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What11;
  
  // kk122
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What12;
  
  // kk123
  kk = G1*ny;
  ResTot2 += kk*What13;
  
  // kk130
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 += kk*What10;
  
  // kk131
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What11;
  
  // kk132
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What12;
  
  // kk133
  kk = G*wtilde;
  ResTot3 += kk*What13; 
  
  // Third direction
  nx = half*tnx3*tl3;
  ny = half*tny3*tl3;
  wtilde = utilde*nx + vtilde*ny;
  
  // kk200    
  //kk = 0.0f;
  //ResTot0 += tl*kk*What20];
  
  // kk201
  kk = nx;
  ResTot0 += kk*What21;
  
  // kk202
  kk = ny;
  ResTot0 += kk*What22;
  
  // kk203
  //kk = 0.0f;
  //ResTot0 += tl*kk*What23;
  pTresTot[n].x = half*ResTot0;
  
  // kk210
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 += kk*What20;
  
  // kk211
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What21;
  
  // kk212
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What22;
  
  // kk213
  kk = G1*nx;
  ResTot1 += kk*What23;
  
  pTresTot[n].y = half*ResTot1;
  
  // kk220
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 += kk*What20;
  
  // kk221
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What21;
  
  // kk222
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What22;
  
  // kk223
  kk = G1*ny;
  ResTot2 += kk*What23;
  
  pTresTot[n].z = half*ResTot2;
  
  // kk230
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 += kk*What20;
    
  // kk231
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What21;
    
  // kk232
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What22;
  
  // kk233
  kk = G*wtilde;
  ResTot3 += kk*What23;

  pTresTot[n].w = half*ResTot3;   

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real Wtemp0 = ResSource0;
  real Wtemp1 = ResSource1;
  real Wtemp2 = ResSource2;
  real Wtemp3 = ResSource3;

  real km;
  //real invCtilde = 
  //  rsqrtf(G1*(htilde - half*(Sq(utilde) + Sq(vtilde))));
  real invCtilde = rsqrtf(G1*htilde - absvtc);
  real ctilde = one/invCtilde;
  
  real hoverc = htilde*invCtilde;
  real uoverc = utilde*invCtilde;
  real voverc = vtilde*invCtilde;
  absvtc = absvtc*invCtilde;
  
  // First direction
  nx = half*tnx1;
  ny = half*tny1;
  real tl = tl1;
  wtilde = utilde*nx + vtilde*ny;
  
  //real l1 = min(zero, wtilde + ctilde);
  //real l2 = min(zero, wtilde - ctilde);
  //real l3 = min(zero, wtilde);
  real l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  real l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  real l3 = half*(wtilde - fabs(wtilde));

  // Auxiliary variables
  real l1l2l3 = half*(l1 + l2) - l3;
  real l1l2 = half*(l1 - l2);
  
  // km000    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What00;
  real nm00 = tl*km;
    
  // km001
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  Wtemp0 += tl*km*What01;
  real nm01 = tl*km;
    
  // km002
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What02;
  real nm02 = tl*km;
    
  // km003
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What03;
  real nm03 = tl*km;
    
  // km010
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What00;
  real nm10 = tl*km;
    
  // km011
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What01;
  real nm11 = tl*km;
  
  // km012
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What02;
  real nm12 = tl*km;
    
  // km013
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What03;
  real nm13 = tl*km;
    
  // km020
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What00;
  real nm20 = tl*km;
    
  // km021
  km = voverc*nx*l1l2 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2) + nx*ny*l1l2l3;
  Wtemp2 += tl*km*What01;
  real nm21 = tl*km;
  
  // km022
  km = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What02;
  real nm22 = tl*km;
  
  // km023
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What03;
  real nm23 = tl*km;
  
  // km030
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What00;
  real nm30 = tl*km;
  
  // km031
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What01;
  real nm31 = tl*km;
  
  // km032
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What02;
  real nm32 = tl*km;
  
  // km033
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  Wtemp3 += tl*km*What03;
  real nm33 = tl*km;
  
  // Second direction         
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  //l1 = min(wtilde + ctilde, zero);
  //l2 = min(wtilde - ctilde, zero);
  //l3 = min(wtilde, zero);
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
 
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km100    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What10;
  nm00 += tl*km;
  
  // km101
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  Wtemp0 += tl*km*What11;
  nm01 += tl*km;
  
  // km102
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What12;
  nm02 += tl*km;
  
  // km103
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What13;
  nm03 += tl*km;
  
  // km110
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What10;
  nm10 += tl*km;
  
  // km111
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What11;
  nm11 += tl*km;
  
  // km112
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What12;
  nm12 += tl*km;
  
  // km113 
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What13;
  nm13 += tl*km;
  
  // km120
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What10;
  nm20 += tl*km;
  
  // km121
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  Wtemp2 += tl*km*What11;
  nm21 += tl*km;
  
  // km122
  km = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What12;
  nm22 += tl*km;
  
  // km123
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What13;
  nm23 += tl*km;
  
  // km130
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What10;
  nm30 += tl*km;
  
  // km131
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What11;
  nm31 += tl*km;
  
  // km132
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What12;
  nm32 += tl*km;
  
  // km133
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  Wtemp3 += tl*km*What13;
  nm33 += tl*km;
  
  // Third direction
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  //l1 = min(wtilde + ctilde, zero);
  //l2 = min(wtilde - ctilde, zero);
  //l3 = min(wtilde, zero);
  l1 = half*(wtilde + ctilde - fabs(wtilde + ctilde));
  l2 = half*(wtilde - ctilde - fabs(wtilde - ctilde));
  l3 = half*(wtilde - fabs(wtilde));
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km200    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What20;
  nm00 += tl*km;
  
  // km201
  km = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  Wtemp0 += tl*km*What21;
  nm01 += tl*km;
  
  // km202
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What22;
  nm02 += tl*km;
  
  // km203
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What23;
  nm03 += tl*km;
  
  // km210
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What20;
  nm10 += tl*km;
  
  // km211
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What21;
  nm11 += tl*km;
  
  // km212
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What22;
  nm12 += tl*km;
  
  // km213
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What23;
  nm13 += tl*km;
  
  // km220
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What20;
  nm20 += tl*km;
  
  // km221
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  Wtemp2 += tl*km*What21;
  nm21 += tl*km;
  
  // km222
  km = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What22;
  nm22 += tl*km;
  
  // km223
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What23;
  nm23 += tl*km;
  
  // km230
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What20;
  nm30 += tl*km;
  
  // km231
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What21;
  nm31 += tl*km;
  
  // km232
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What22;
  nm32 += tl*km;
  
  // km233
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
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
  real ResN, kp;
  
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;
  
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);    
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp000    
  kp = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  ResN = kp*What00;
  
  // kp001
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN += kp*What01;
  
  // kp002
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN += kp*What02;
  
  // kp003
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN += kp*What03;
  
  pTresN0[n].x += half*ResN;
  
  // kp010
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN = kp*What00;
  
  // kp011
  kp = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN += kp*What01;
  
  // kp012
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN += kp*What02;
  
  // kp013
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN += kp*What03;
  
  pTresN0[n].y += half*ResN;
  
  // kp020
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN = kp*What00;
  
  // kp021
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN += kp*What01;
  
  // kp022
  kp = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN += kp*What02;
  
  // kp023
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN += kp*What03;
  
  pTresN0[n].z += half*ResN;
  
  // kp030
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN = kp*What00;
  
  // kp031
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What01;
  
  // kp032
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What02;
  
  // kp033
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN += kp*What03;
  
  pTresN0[n].w += half*ResN;
  
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  // Second direction
  nx = half*Tnx2;
  ny = half*Tny2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);    
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp100    
  kp = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  ResN = kp*What10;
  
  // kp101
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN += kp*What11;
  
  // kp102
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN += kp*What12;
  
  // kp103
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN += kp*What13;
  
  pTresN1[n].x += half*ResN;
  
  // kp110
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN = kp*What10;
  
  // kp111
  kp = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN += kp*What11;
  
  // kp112
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN += kp*What12;
  
  // kp113
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN += kp*What13;
  
  pTresN1[n].y += half*ResN;
  
  // kp120
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN = kp*What10;
  
  // kp121
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN += kp*What11;
  
  // kp122
  kp = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN += kp*What12;
  
  // kp123
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN += kp*What13;
  
  pTresN1[n].z += half*ResN;
  
  // kp130
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN = kp*What10;
  
  // kp131
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What11;
  
  // kp132
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What12;
  
  // kp133
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN += kp*What13;
  
  pTresN1[n].w   += half*ResN;
  
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  // Third direction
  nx = half*Tnx3;
  ny = half*Tny3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp200    
  kp = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  ResN = kp*What20;
  
  // kp201
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN += kp*What21;
  
  // kp202
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN += kp*What22;
  
  // kp203
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN += kp*What23;
  
  pTresN2[n].x += half*ResN;
  
  // kp210
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN = kp*What20;
  
  // kp211
  kp = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN += kp*What21;
  
  // kp212
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN += kp*What22;
  
  // kp213
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN += kp*What23;
  
  pTresN2[n].y += half*ResN;
  
  // kp220
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN = kp*What20;
  
  // kp221
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN += kp*What21;
  
  // kp222
  kp = Sq(ny)*l1l2l3 + l3 - voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN += kp*What22;
  
  // kp223
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN += kp*What23;
  
  pTresN2[n].z += half*ResN;
  
  // kp230
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN = kp*What20;
  
  // kp231
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What21;
  
  // kp232
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN += kp*What22;
  
  // kp233
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN += kp*What23;
  
  pTresN2[n].w += half*ResN;
}

__host__ __device__
void CalcTotalResNtotSingle(const int n, const real dt,
			    const int3* __restrict__ pTv, 
			    const real* __restrict__ pVz, real *pDstate,
			    const real2 *pTn1, const real2 *pTn2,
			    const real2 *pTn3, const real3* __restrict__ pTl,
			    real *pVpot, real *pTresN0, real *pTresN1,
			    real *pTresN2, real *pTresTot, int nVertex,
			    const real G, const real G1,
			    const real G2, const real iG)
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
  
  // Parameter vector at vertices: 12 uncoalesced loads
  real Zv0 = pVz[v1];
  real Zv1 = pVz[v2];
  real Zv2 = pVz[v3];
  
  // Average parameter vector
#if BURGERS == 1
  real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
  real vx = Z0;
  real vy = Z0;
#else
  real vx = one;
  real vy = zero;
#endif

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
  
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme 
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
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

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  real Wtemp = zero;
  
  // First direction
  nx = half*tnx1;
  ny = half*tny1;
  real tl = tl1;

  real l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
  Wtemp += tl*l1*What0;
  real nm = tl*l1;
    
  // Second direction
  nx = half*tnx2;
  ny = half*tny2;
  tl = tl2;

  l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
  Wtemp += tl*l1*What1;
  nm += tl*l1;

  // Third direction
  nx = half*tnx3;
  ny = half*tny3;
  tl = tl3;

  l1 = half*(vx*nx + vy*ny - fabs(vx*nx + vy*ny));
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
\param iG 1/G*/
//######################################################################

__global__ void
devCalcTotalResNtot(int nTriangle, real dt,
		    const int3* __restrict__ pTv, 
		    const realNeq* __restrict__ pVz, realNeq *pDstate,
		    const real2 *pTn1, const real2 *pTn2, const real2 *pTn3,
		    const real3* __restrict__  pTl, real *pVpot,
		    realNeq *pTresN0, realNeq *pTresN1, realNeq *pTresN2,
		    realNeq *pTresTot, int nVertex,
		    real G, real G1, real G2, real iG)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    CalcTotalResNtotSingle(n, dt, pTv, pVz, pDstate,
			   pTn1, pTn2, pTn3, pTl, pVpot,
			   pTresN0, pTresN1, pTresN2,
			   pTresTot, nVertex, G, G1, G2, iG);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate space-time residue (N + total) for all triangles; result in  \a triangleResidueN and \a triangleResidueTotal.

\param dt Time step*/
//######################################################################

void Simulation::CalcTotalResNtot(real dt)
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
      vertexPotential->TransformToDevice();
      vertexParameterVector->TransformToDevice();
      vertexStateDiff->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueTotal->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexPotential->TransformToHost();
      vertexParameterVector->TransformToHost();
      vertexStateDiff->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueTotal->TransformToHost();
      
      cudaFlag = 0;
    }
  }

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pDstate = vertexStateDiff->GetPointer();
  
  real *pVpot = vertexPotential->GetPointer();
  realNeq *pVz = vertexParameterVector->GetPointer();
  
  realNeq *pTresN0 = triangleResidueN->GetPointer(0);
  realNeq *pTresN1 = triangleResidueN->GetPointer(1);
  realNeq *pTresN2 = triangleResidueN->GetPointer(2);
  
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
				       devCalcTotalResNtot, 
				       (size_t) 0, 0);

#ifdef TIME_ASTRIX
    cudaEventRecord(start, 0);
#endif
    devCalcTotalResNtot<<<nBlocks, nThreads>>>
      (nTriangle, dt, pTv, pVz, pDstate,
       pTn1, pTn2, pTn3, pTl, pVpot,
       pTresN0, pTresN1, pTresN2, 
       pTresTot, nVertex, specificHeatRatio,
       specificHeatRatio - 1.0, specificHeatRatio - 2.0,
       1.0/specificHeatRatio);
    
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
      CalcTotalResNtotSingle(n, dt, pTv, pVz, pDstate,
			     pTn1, pTn2, pTn3, pTl, pVpot,
			     pTresN0, pTresN1, pTresN2, 
			     pTresTot, nVertex, specificHeatRatio,
			     specificHeatRatio - 1.0, specificHeatRatio - 2.0,
			     1.0/specificHeatRatio);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
  }
  
#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Kernel: devCalcTotalRes, # of elements: "
	    << nTriangle << ", elapsed time: " << elapsedTime << std::endl;
#endif
  
  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 0) {
      vertexPotential->TransformToDevice();
      vertexStateDiff->TransformToDevice();
      vertexParameterVector->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueTotal->TransformToDevice();

      cudaFlag = 1;
    } else {
      vertexPotential->TransformToHost();
      vertexStateDiff->TransformToHost();
      vertexParameterVector->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueTotal->TransformToHost();
      
      cudaFlag = 0;
    }
  }
}

}
