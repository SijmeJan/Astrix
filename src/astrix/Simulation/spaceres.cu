// -*-c++-*-
/*! \file spaceres.cu
\brief File containing functions for calculating spatial residue*/
#include <iostream>
#include <fstream>
#include <iomanip>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

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
\param *pVpot Pointer to gravitational potential at vertices
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
			const real2 *pTn3, const real3 *pTl, real *pVpot,
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
  
  // Source term residual 
  real rhoAve  = Z0*Z0;
  real momxAve = Z0*Z1;
  real momyAve = Z0*Z2;

  real tl1 = pTl[n].x;
  real tl2 = pTl[n].y;
  real tl3 = pTl[n].z;

  real tnx1 = pTn1[n].x;
  real tnx2 = pTn2[n].x;
  real tnx3 = pTn3[n].x;
  real tny1 = pTn1[n].y;
  real tny2 = pTn2[n].y;
  real tny3 = pTn3[n].y;

  real dPotdx =
    tnx1*tl1*pVpot[v1] + tnx2*tl2*pVpot[v2] + tnx3*tl3*pVpot[v3];
  real dPotdy =
    tny1*tl1*pVpot[v1] + tny2*tl2*pVpot[v2] + tny3*tl3*pVpot[v3];
  
  // -integral(Source*dS)
  real ResSource0 = zero;
  real ResSource1 = half*rhoAve*dPotdx;
  real ResSource2 = half*rhoAve*dPotdy;
  real ResSource3 = -half*momxAve*dPotdx - half*momyAve*dPotdy;

  // Total residue
  real ResTot0 = ResSource0;
  real ResTot1 = ResSource1;
  real ResTot2 = ResSource2;
  real ResTot3 = ResSource3;
  
  real Wtemp0 = ResSource0;
  real Wtemp1 = ResSource1;
  real Wtemp2 = ResSource2;
  real Wtemp3 = ResSource3;
  
  // Matrix element K- + K+
  real kk;
  
  real utilde = Z1/Z0;
  real vtilde = Z2/Z0;
  real htilde = Z3/Z0;
  real absvt  = half*(Sq(utilde) + Sq(vtilde));
  real absvtc = G1*absvt;
  
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate the total residue = Sum(K*What)
  // Not necessary for first-order N scheme 
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  // First direction
  real tl = tl1;
  real nx = half*tl*tnx1;
  real ny = half*tl*tny1;
  real wtilde = utilde*nx + vtilde*ny;
  
  // kk[0][0][0]    
  //kk = 0.0f;
  //ResTot[0] +=  t_l*kk*What[0][0];
  
  // kk[0][0][1]
  kk = nx;
  ResTot0 += kk*What01;
  
  // kk[0][0][2]
  kk = ny;
  ResTot0 += kk*What02; 

  // kk[0][0][3]
  //kk = 0.0f;
  //ResTot[0] += kk*What0][3];
  
  // kk[0][1][0]
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 +=  kk*What00;

  // kk[0][1][1]
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What01;
  
  // kk[0][1][2]
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What02;
  
  // kk[0][1][3]
  kk = G1*nx;
  ResTot1 += kk*What03;
  
  // kk[0][2][0]
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 +=  kk*What00;
  
  // kk[0][2][1]
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What01;
  
  // kk[0][2][2]
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What02;
  
  // kk[0][2][3]
  kk = G1*ny;
  ResTot2 += kk*What03;
  
  // kk[0][3][0]
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 +=  kk*What00;
  
  // kk[0][3][1]
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What01;
  
  // kk[0][3][2]
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What02;
  
  // kk[0][3][3]
  kk = G*wtilde;
  ResTot3 += kk*What03;
  
  // Second direction
  tl = tl2;
  nx = half*tl*tnx2;
  ny = half*tl*tny2;
  wtilde = utilde*nx + vtilde*ny;
  
  // kk[1][0][0]    
  //kk = 0.0f;
  //ResTot[0] += kk*What1][0];
  
  // kk[1][0][1]
  kk = nx;
  ResTot0 += kk*What11;
  
  // kk[1][0][2]
  kk = ny;
  ResTot0 += kk*What12;
  
  // kk[1][0][3]
  //kk = 0.0f;
  //ResTot[0] += kk*What1][3];
  
  // kk[1][1][0]
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 += kk*What10;
  
  // kk[1][1][1]
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What11;
  
  // kk[1][1][2]
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What12;
  
  // kk[1][1][3]
  kk = G1*nx;
  ResTot1 += kk*What13;
  
  // kk[1][2][0]
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 += kk*What10;
  
  // kk[1][2][1]
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What11;
  
  // kk[1][2][2]
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What12;
  
  // kk[1][2][3]
  kk = G1*ny;
  ResTot2 += kk*What13;
  
  // kk[1][3][0]
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 += kk*What10;
  
  // kk[1][3][1]
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What11;
  
  // kk[1][3][2]
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What12;
  
  // kk[1][3][3]
  kk = G*wtilde;
  ResTot3 += kk*What13;
  
  // Third direction
  tl = tl3;
  nx = half*tl*tnx3;
  ny = half*tl*tny3;
  wtilde = utilde*nx + vtilde*ny;
  
  // kk[2][0][0]    
  //kk = 0.0f;
  //ResTot[0] += kk*What2][0];
  
  // kk[2][0][1]
  kk = nx;
  ResTot0 += kk*What21;

  // kk[2][0][2]
  kk = ny;
  ResTot0 += kk*What22;

  // kk[2][0][3]
  //kk = 0.0f;
  //ResTot[0] += kk*What2][3];
  pTresTot[n].x = ResTot0;
  
  // kk[2][1][0]
  kk = absvtc*nx - wtilde*utilde; 
  ResTot1 += kk*What20;
  
  // kk[2][1][1]
  kk = wtilde - utilde*nx*G2;
  ResTot1 += kk*What21;
  
  // kk[2][1][2]
  kk = utilde*ny - vtilde*G1*nx;
  ResTot1 += kk*What22;
  
  // kk[2][1][3]
  kk = G1*nx;
  ResTot1 += kk*What23;
  pTresTot[n].y = ResTot1;
  
  // kk[2][2][0]
  kk = absvtc*ny - wtilde*vtilde;
  ResTot2 += kk*What20;
  
  // kk[2][2][1]
  kk = vtilde*nx - G1*utilde*ny;
  ResTot2 += kk*What21;
  
  // kk[2][2][2]
  kk = wtilde - vtilde*G2*ny;
  ResTot2 += kk*What22;
  
  // kk[2][2][3]
  kk = G1*ny;
  ResTot2 += kk*What23;
  pTresTot[n].z = ResTot2;
  
  // kk[2][3][0]
  kk = absvtc*wtilde - wtilde*htilde;
  ResTot3 += kk*What20;
  
  // kk[2][3][1]
  kk = nx*htilde - G1*utilde*wtilde;
  ResTot3 += kk*What21;
  
  // kk[2][3][2]
  kk = ny*htilde - G1*vtilde*wtilde;
  ResTot3 += kk*What22;
  
  // kk[2][3][3]
  kk = G*wtilde;
  ResTot3 += kk*What23;
  pTresTot[n].w = ResTot3;   
  
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  real km;
  //real invCtilde = 
  //  rsqrtf(G1*(htilde - half*(Sq(utilde) + Sq(vtilde))));
  real invCtilde = 
    1.0/sqrt(G1*(htilde - half*(Sq(utilde) + Sq(vtilde))));
  real ctilde = one/invCtilde;
  
  real hoverc = htilde*invCtilde;
  real uoverc = utilde*invCtilde;
  real voverc = vtilde*invCtilde;
  absvtc = absvtc*invCtilde;
  
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
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What00;
  real nm00 = tl*km;
  
  // km[0][0][1]
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  Wtemp0 += tl*km*What01;
  real nm01 = tl*km;
  
  // km[0][0][2]
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What02;
  real nm02 = tl*km;
  
  // km[0][0][3]
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What03;
  real nm03 = tl*km;
  
  // km[0][1][0]
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What00;
  real nm10 = tl*km;
  
  // km[0][1][1]
  km = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What01;
  real nm11 = tl*km;
  
  // km[0][1][2]
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What02;
  real nm12 = tl*km;
  
  // km[0][1][3]
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What03;
  real nm13 = tl*km;

  // km[0][2][0]
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What00;
  real nm20 = tl*km;
  
  // km[0][2][1]
  km = voverc*nx*l1l2 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2) +
    nx*ny*l1l2l3;
  Wtemp2 += tl*km*What01;
  real nm21 = tl*km;
  
  // km[0][2][2]
  km = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What02;
  real nm22 = tl*km;
  
  // km[0][2][3]
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What03;
  real nm23 = tl*km;
  
  // km[0][3][0]
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What00;
  real nm30 = tl*km;
  
  // km[0][3][1]
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What01;
  real nm31 = tl*km;
  
  // km[0][3][2]
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What02;
  real nm32 = tl*km;
  
  // km[0][3][3]
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  Wtemp3 += tl*km*What03;
  real nm33 = tl*km;
  
  // Second direction         
  nx = tnx2;
  ny = tny2;
  tl = half*tl2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km[1][0][0]    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What10;
  nm00 += tl*km;
  
  // km[1][0][1]
  km = invCtilde*nx*l1l2 - invCtilde*G1*uoverc*l1l2l3;
  Wtemp0 += tl*km*What11;
  nm01 += tl*km;
  
  // km[1][0][2]
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What12;
  nm02 += tl*km;
  
  // km[1][0][3]
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What13;
  nm03 += tl*km;
  
  // km[1][1][0]
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) - wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What10;
  nm10 += tl*km;
  
  // km[1][1][1]
  km = l3 + Sq(nx)*l1l2l3 - uoverc*(nx*(G-two)*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What11;
  nm11 += tl*km;
  
  // km[1][1][2]
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 - voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What12;
  nm12 += tl*km;
  
  // km[1][1][3]
  km=G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What13;
  nm13 += tl*km;
  
  // km[1][2][0]
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What10;
  nm20 += tl*km;
  
  // km[1][2][1]
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 - G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  Wtemp2 += tl*km*What11;
  nm21 += tl*km;
  
  // km[1][2][2]
  km = Sq(ny)*l1l2l3 + l3 - voverc*((G-two)*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What12;
  nm22 += tl*km;
  
  // km[1][2][3]
  km=G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What13;
  nm23 += tl*km;
  
  // km[1][3][0]
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What10;
  nm30 += tl*km;
  
  // km[1][3][1]
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What11;
  nm31 += tl*km;
  
  // km[1][3][2]
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What12;
  nm32 += tl*km;
  
  // km[1][3][3]
  km = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  Wtemp3 += tl*km*What13;
  nm33 += tl*km;
  
  // Third direction
  nx = tnx3;
  ny = tny3;
  tl = half*tl3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = min(wtilde + ctilde, zero);
  l2 = min(wtilde - ctilde, zero);
  l3 = min(wtilde, zero);
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // km[2][0][0]    
  km = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  Wtemp0 += tl*km*What20;
  nm00 += tl*km;
  
  // km[2][0][1]
  km = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  Wtemp0 += tl*km*What21;
  nm01 += tl*km;
  
  // km[2][0][2]
  km = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  Wtemp0 += tl*km*What22;
  nm02 += tl*km;
  
  // km[2][0][3]
  km = G1*l1l2l3*Sq(invCtilde);
  Wtemp0 += tl*km*What23;
  nm03 += tl*km;
  
  // km[2][1][0]
  km = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  Wtemp1 += tl*km*What20;
  nm10 += tl*km;
  
  // km[2][1][1]
  km = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  Wtemp1 += tl*km*What21;
  nm11 += tl*km;
  
  // km[2][1][2]
  km = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  Wtemp1 += tl*km*What22;
  nm12 += tl*km;
  
  // km[2][1][3]
  km = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  Wtemp1 += tl*km*What23;
  nm13 += tl*km;
  
  // km[2][2][0]
  km = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  Wtemp2 += tl*km*What20;
  nm20 += tl*km;
  
  // km[2][2][1]
  km = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  Wtemp2 += tl*km*What21;
  nm21 += tl*km;
  
  // km[2][2][2]
  km = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  Wtemp2 += tl*km*What22;
  nm22 += tl*km;
  
  // km[2][2][3]
  km = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  Wtemp2 += tl*km*What23;
  nm23 += tl*km;
  
  // km[2][3][0]
  km = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  Wtemp3 += tl*km*What20;
  nm30 += tl*km;
  
  // km[2][3][1]
  km = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What21;
  nm31 += tl*km;
  
  // km[2][3][2]
  km = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  Wtemp3 += tl*km*What22;
  nm32 += tl*km;
  
  // km[2][3][3]
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

  // Wtilde = Nm*Wtemp
  real Wtilde0 = invN00*Wtemp0 + invN01*Wtemp1 +
    invN02*Wtemp2 + invN03*Wtemp3;
  real Wtilde1 = invN10*Wtemp0 + invN11*Wtemp1 +
    invN12*Wtemp2 + invN13*Wtemp3;
  real Wtilde2 = invN20*Wtemp0 + invN21*Wtemp1 +
    invN22*Wtemp2 + invN23*Wtemp3;
  real Wtilde3 = invN30*Wtemp0 + invN31*Wtemp1 +
    invN32*Wtemp2 + invN33*Wtemp3;

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

  nx = Tnx1;
  ny = Tny1;

  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);    
  
  // Auxiliary variables
  l1l2l3 = half*(l1+l2)-l3;
  l1l2 = half*(l1-l2);
  
  // kp[0][0][0]    
  kp = absvtc*invCtilde*l1l2l3 + l3 - l1l2*wtilde*invCtilde;
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;
  
  // kp[0][0][1]
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;
  
  // kp[0][0][2]
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;
  
  // kp[0][0][3]
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;
  
  pTresN0[n].x   = half*ResN;
  pTresLDA0[n].x = half*ResLDA;
  
  // kp[0][1][0]
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;
  
  // kp[0][1][1]
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;
  
  // kp[0][1][2]
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;
  
  // kp[0][1][3]
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;
  
  pTresN0[n].y   = half*ResN;
  pTresLDA0[n].y = half*ResLDA;
  
  // kp[0][2][0]
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;
  
  // kp[0][2][1]
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;
  
  // kp[0][2][2]
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;
  
  // kp[0][2][3]
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;
  
  pTresN0[n].z   = half*ResN;
  pTresLDA0[n].z = half*ResLDA;
  
  // kp[0][3][0]
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN   = kp*What00;
  ResLDA =-kp*Wtemp0;
  
  // kp[0][3][1]
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What01;
  ResLDA -= kp*Wtemp1;
  
  // kp[0][3][2]
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What02;
  ResLDA -= kp*Wtemp2;
  
  // kp[0][3][3]
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN   += kp*What03;
  ResLDA -= kp*Wtemp3;
  
  pTresN0[n].w   = half*ResN;
  pTresLDA0[n].w = half*ResLDA;
  
  // Second direction
  nx = Tnx2;
  ny = Tny2;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);    
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp[1][0][0]    
  kp = absvtc*invCtilde*l1l2l3 + 
    l3 - l1l2*wtilde*invCtilde;
  ResN   =  kp*What10;
  ResLDA = -kp*Wtemp0;
  
  // kp[1][0][1]
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;
  
  // kp[1][0][2]
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;
  
  // kp[1][0][3]
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;
  
  pTresN1[n].x   = half*ResN;
  pTresLDA1[n].x = half*ResLDA;
  
  // kp[1][1][0]
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;
  
  // kp[1][1][1]
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;
  
  // kp[1][1][2]
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;
  
  // kp[1][1][3]
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;
  
  pTresN1[n].y   = half*ResN;
  pTresLDA1[n].y = half*ResLDA;
  
  // kp[1][2][0]
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN   =  kp*What10;
  ResLDA = -kp*Wtemp0;
  
  // kp[1][2][1]
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;
  
  // kp[1][2][2]
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;
 
  // kp[1][2][3]
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;
  
  pTresN1[n].z   = half*ResN;
  pTresLDA1[n].z = half*ResLDA;
  
  // kp[1][3][0]
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN   = kp*What10;
  ResLDA =-kp*Wtemp0;
  
  // kp[1][3][1]
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What11;
  ResLDA -= kp*Wtemp1;
  
  // kp[1][3][2]
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What12;
  ResLDA -= kp*Wtemp2;
  
  // kp[1][3][3]
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN   += kp*What13;
  ResLDA -= kp*Wtemp3;
  
  pTresN1[n].w   = half*ResN;
  pTresLDA1[n].w = half*ResLDA;
  
  // Third direction
  nx = Tnx3;
  ny = Tny3;
  wtilde = (uoverc*nx + voverc*ny)*ctilde;
  
  l1 = max(wtilde + ctilde, zero);
  l2 = max(wtilde - ctilde, zero);
  l3 = max(wtilde, zero);
  
  // Auxiliary variables
  l1l2l3 = half*(l1 + l2) - l3;
  l1l2 = half*(l1 - l2);
  
  // kp[2][0][0]    
  kp = absvtc*invCtilde*l1l2l3 + 
    l3 - l1l2*wtilde*invCtilde;
  ResN   =  kp*What20;
  ResLDA = -kp*Wtemp0;
  
  // kp[2][0][1]
  kp = invCtilde*(nx*l1l2 - G1*uoverc*l1l2l3);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;
  
  // kp[2][0][2]
  kp = invCtilde*(ny*l1l2 - G1*voverc*l1l2l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;
  
  // kp[2][0][3]
  kp = G1*l1l2l3*Sq(invCtilde);
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;
  
  pTresN2[n].x   = half*ResN;
  pTresLDA2[n].x = half*ResLDA;
  
  // kp[2][1][0]
  kp = absvtc*(uoverc*l1l2l3 + nx*l1l2) -
    wtilde*(nx*l1l2l3 + uoverc*l1l2); 
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;
  
  // kp[2][1][1]
  kp = l3 + Sq(nx)*l1l2l3 -
    uoverc*(nx*G2*l1l2 + G1*uoverc*l1l2l3);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;
  
  // kp[2][1][2]
  kp = uoverc*ny*l1l2 + nx*ny*l1l2l3 -
    voverc*G1*(nx*l1l2 + uoverc*l1l2l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;
  
  // kp[2][1][3]
  kp = G1*(uoverc*l1l2l3 + nx*l1l2)*invCtilde;
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;
  
  pTresN2[n].y   = half*ResN;
  pTresLDA2[n].y = half*ResLDA;
  
  // kp[2][2][0]
  kp = absvtc*(voverc*l1l2l3 + ny*l1l2) - 
    wtilde*(voverc*l1l2 + ny*l1l2l3);
  ResN   = kp*What20;
  ResLDA =-kp*Wtemp0;
  
  // kp[2][2][1]
  kp = voverc*nx*l1l2 + nx*ny*l1l2l3 -
    G1*uoverc*(voverc*l1l2l3 + ny*l1l2);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;
  
  // kp[2][2][2]
  kp = Sq(ny)*l1l2l3 + l3 -
    voverc*(G2*ny*l1l2 + G1*voverc*l1l2l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;
  
  // kp[2][2][3]
  kp = G1*(voverc*l1l2l3 + ny*l1l2)*invCtilde;
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;
  
  pTresN2[n].z   = half*ResN;
  pTresLDA2[n].z = half*ResLDA;
  
  // kp[2][3][0]
  kp = absvtc*(wtilde*l1l2 + hoverc*l1l2l3) -
    wtilde*(hoverc*l1l2 + wtilde*l1l2l3);
  ResN   =  kp*What20;
  ResLDA = -kp*Wtemp0;
  
  // kp[2][3][1]
  kp = nx*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*uoverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What21;
  ResLDA -= kp*Wtemp1;

  // kp[2][3][2]
  kp = ny*(hoverc*l1l2 + wtilde*l1l2l3) -
    G1*voverc*(wtilde*l1l2 + hoverc*l1l2l3);
  ResN   += kp*What22;
  ResLDA -= kp*Wtemp2;
  
  // kp[2][3][3]
  kp = G1*invCtilde*(hoverc*l1l2l3 + wtilde*l1l2) + l3;
  ResN   += kp*What23;
  ResLDA -= kp*Wtemp3;
  
  pTresN2[n].w   = half*ResN;
  pTresLDA2[n].w = half*ResLDA;
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
\param *pVpot Pointer to gravitational potential at vertices
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
devCalcSpaceRes(int nTriangle, const int3 *pTv, real4 *pVz,
		const real2 *pTn1, const real2 *pTn2,
		const real2 *pTn3, const real3 *pTl, real *pVpot,
		real4 *pTresN0, real4 *pTresN1, real4 *pTresN2,
		real4 *pTresLDA0, real4 *pTresLDA1, real4 *pTresLDA2,
		real4 *pTresTot, int nVertex, real G, real G1, real G2)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;


  while (n < nTriangle) {
    CalcSpaceResSingle(n, pTv, pVz, pTn1, pTn2, pTn3, pTl, pVpot,
    		       pTresN0, pTresN1, pTresN2,
                       pTresLDA0, pTresLDA1, pTresLDA2,
    		       pTresTot, nVertex, G, G1, G2);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate spatial residue for all triangles; result in \a triangleResidueN, \a triangleResidueLDA and \a triangleResidueTotal.*/
//######################################################################

void Simulation::CalcResidual()
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

      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
     
      cudaFlag = 1;
    } else {
      vertexPotential->TransformToHost();
      vertexParameterVector->TransformToHost();

      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleResidueTotal->TransformToHost();

      cudaFlag = 0;
    }
  }

  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  real *pVpot = vertexPotential->GetPointer();
  real4 *pVz = vertexParameterVector->GetPointer();
  
  real4 *pTresN0 = triangleResidueN->GetPointer(0);
  real4 *pTresN1 = triangleResidueN->GetPointer(1);
  real4 *pTresN2 = triangleResidueN->GetPointer(2);

  real4 *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  real4 *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  real4 *pTresLDA2 = triangleResidueLDA->GetPointer(2);

  real4 *pTresTot = triangleResidueTotal->GetPointer();
  
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
    cudaEventRecord(start, 0);
#endif
    devCalcSpaceRes<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pVz,
       pTn1, pTn2, pTn3, pTl, pVpot,
       pTresN0, pTresN1, pTresN2, 
       pTresLDA0, pTresLDA1, pTresLDA2,
       pTresTot, nVertex, specificHeatRatio,
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
      CalcSpaceResSingle(n, pTv, pVz, 
			 pTn1, pTn2, pTn3, pTl, pVpot,
			 pTresN0, pTresN1, pTresN2,
			 pTresLDA0, pTresLDA1, pTresLDA2,
			 pTresTot, nVertex, specificHeatRatio,
			 specificHeatRatio - 1.0,
			 specificHeatRatio - 2.0);
#ifdef TIME_ASTRIX
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
#endif
  }
  
#ifdef TIME_ASTRIX
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Kernel: devCalcSpaceRes, # of elements: "
	    << nTriangle << ", elapsed time: " << elapsedTime << std::endl;
#endif

  if (transformFlag == 1) {
    mesh->Transform();
    if (cudaFlag == 1) {
      vertexPotential->TransformToHost();
      vertexParameterVector->TransformToHost();
      
      triangleResidueN->TransformToHost();
      triangleResidueLDA->TransformToHost();
      triangleResidueTotal->TransformToHost();
      
      cudaFlag = 0;
    } else {
      vertexPotential->TransformToDevice();
      vertexParameterVector->TransformToDevice();

      triangleResidueN->TransformToDevice();
      triangleResidueLDA->TransformToDevice();
      triangleResidueTotal->TransformToDevice();
      
      cudaFlag = 1;
    }
  }
}

}
