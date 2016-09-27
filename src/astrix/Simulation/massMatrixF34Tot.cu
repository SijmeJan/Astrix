// -*-c++-*-
/*! \file massMatrix.cu
\brief File containing functions for F3 and F4 mass matrix*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

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
\param G2 G - 2*/
//######################################################################

__host__ __device__
void MassMatrixF34TotSingle(int n, real dt, int massMatrix, 
			    const int3* __restrict__ pTv,
			    const real4* __restrict__ pVz, 
			    const real4* __restrict__ pDstate, 
			    real4 *pTresTot, const real2 *pTn1,
			    const real2 *pTn2, const real2 *pTn3,
			    const real3 *pTl, int nVertex,
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
  
  real resTot0 = (dW00 + dW10 + dW20)*Adt;
  real resTot1 = (dW01 + dW11 + dW21)*Adt;
  real resTot2 = (dW02 + dW12 + dW22)*Adt;
  real resTot3 = (dW03 + dW13 + dW23)*Adt;
  
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

  ctilde = sqrt(ctilde);
  real invCtilde = one/ctilde;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate Wtemp = Sum(K-*What)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  real km;

  real hoverc = htilde*invCtilde;
  real uoverc = utilde*invCtilde;
  real voverc = vtilde*invCtilde;
  real absvtc = G1*absvt*invCtilde;
  
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

  // Wtilde = N*phi
  real Wtilde0 =
    invN00*resTot0 +  
    invN01*resTot1 +  
    invN02*resTot2 +  
    invN03*resTot3;  
  real Wtilde1 =
    invN10*resTot0 +  
    invN11*resTot1 +  
    invN12*resTot2 +  
    invN13*resTot3;  
  real Wtilde2 =
    invN20*resTot0 +  
    invN21*resTot1 +  
    invN22*resTot2 +  
    invN23*resTot3;  
  real Wtilde3 =
    invN30*resTot0 +  
    invN31*resTot1 +  
    invN32*resTot2 +  
    invN33*resTot3;  

  if (det != zero) det = one/det;

  Wtilde0 *= det;
  Wtilde1 *= det;
  Wtilde2 *= det;
  Wtilde3 *= det;

  // ResLDA = -Kp*Wtilde
  real ResLDA, kp;
  
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  wtilde = (uoverc*nx + voverc*ny)*ctilde;
    
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

  pTresTot[n].x += tl1*ResLDA;
  
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
  
  pTresTot[n].y += tl1*ResLDA;
  
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
  
  pTresTot[n].z += tl1*ResLDA;
  
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
  
  pTresTot[n].w += tl1*ResLDA;
  
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
  
  pTresTot[n].x += tl2*ResLDA;
  
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
  
  pTresTot[n].y += tl2*ResLDA;
  
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
  
  pTresTot[n].z += tl2*ResLDA;
  
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
  
  pTresTot[n].w += tl2*ResLDA;
  
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
  
  pTresTot[n].x += tl3*ResLDA;
  
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
  
  pTresTot[n].y += tl3*ResLDA;
  
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
  
  pTresTot[n].z += tl3*ResLDA;
  
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
  
  pTresTot[n].w += tl3*ResLDA;
}

__host__ __device__
void MassMatrixF34TotSingle(int n, real dt, int massMatrix, 
			    const int3* __restrict__ pTv,
			    const real* __restrict__ pVz, 
			    const real* __restrict__ pDstate, 
			    real *pTresTot, const real2 *pTn1,
			    const real2 *pTn2, const real2 *pTn3,
			    const real3 *pTl, int nVertex,
			    real G, real G1, real G2)
{
  const real zero  = (real) 0.0;
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
  
  // Parameter vector at vertices: 12 uncoalesced loads
  //real Zv0 = pVz[vs1];
  //real Zv1 = pVz[vs2];
  //real Zv2 = pVz[vs3];

  // Average parameter vector
  //real Z0 = (Zv0 + Zv1 + Zv2)*onethird;

  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Calculate N
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  real tnx1 = pTn1[n].x;
  //real tny1 = pTn1[n].y;

  // First direction
  real nx = half*tnx1;
  //real ny = half*tny1;
  real tl = tl1;
  
  real l1 = min(zero, nx);
  real nm = l1*tl;
    
  // Second direction         
  real tnx2 = pTn2[n].x;
  //real tny2 = pTn2[n].y;

  nx = half*tnx2;
  //ny = half*tny2;
  tl = tl2;
  
  l1 = min(zero, nx);
  nm += l1*tl;

  // Third direction
  real tnx3 = pTn3[n].x;
  //real tny3 = pTn3[n].y;

  nx = half*tnx3;
  //ny = half*tny3;
  tl = tl3;
  
  l1 = min(zero, nx);
  nm += l1*tl;

  real invN = (real) 1.0;
  if (nm != zero) invN /= nm;

  // Wtilde = N*phi
  real Wtilde = invN*dW0;
  real ResLDA;
  
  real Tnx1 = pTn1[n].x;
  //real Tny1 = pTn1[n].y;

  nx = half*tl1*Tnx1;
  //ny = half*Tny1;

  l1 = half*(nx + fabs(nx));
  ResLDA = -l1*Wtilde;

  pTresTot[n] += ResLDA;

  // Wtilde = N*dW1
  Wtilde = invN*dW1;

  real Tnx2 = pTn2[n].x;
  //real Tny2 = pTn2[n].y;

  // Second direction
  nx = half*tl2*Tnx2;
  //ny = half*Tny2;

  l1 = half*(nx + fabs(nx));
  ResLDA = -l1*Wtilde;

  pTresTot[n] += ResLDA;
  
  // Wtilde = N*dW1
  Wtilde = invN*dW2;

  real Tnx3 = pTn3[n].x;
  //real Tny3 = pTn3[n].y;

  // Third direction
  nx = half*tl3*Tnx3;
  //ny = half*Tny3;

  l1 = half*(nx + fabs(nx));
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
\param G2 G - 2*/
//######################################################################

__global__ void
devMassMatrixF34Tot(int nTriangle, real dt, int massMatrix,
		    const int3* __restrict__ pTv,
		    const realNeq* __restrict__ pVz, 
		    const realNeq* __restrict__ pDstate, 
		    realNeq *pTresTot, const real2 *pTn1, const real2 *pTn2,
		    const real2 *pTn3, const real3 *pTl,
		    int nVertex, real G, real G1, real G2)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    MassMatrixF34TotSingle(n, dt, massMatrix, pTv, pVz, pDstate,
			   pTresTot, pTn1, pTn2, pTn3, pTl,
			   nVertex, G, G1, G2);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate mass matrix contribution F3/F4 to total residual.*/
//######################################################################

void Simulation::MassMatrixF34Tot(real dt, int massMatrix)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pVz = vertexParameterVector->GetPointer();
  realNeq *pDstate = vertexStateDiff->GetPointer();
  
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
				       devMassMatrixF34Tot, 
				       (size_t) 0, 0);

    devMassMatrixF34Tot<<<nBlocks, nThreads>>>
      (nTriangle, dt, massMatrix, pTv, pVz, pDstate,
       pTresTot, pTn1, pTn2, pTn3, pTl, nVertex, 
       specificHeatRatio, 
       specificHeatRatio - 1.0,
       specificHeatRatio - 2.0);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

  } else {
    for (int n = 0; n < nTriangle; n++) 
      MassMatrixF34TotSingle(n, dt, massMatrix, pTv, pVz, pDstate,
			     pTresTot, pTn1, pTn2, pTn3, pTl, nVertex, 
			     specificHeatRatio, specificHeatRatio - 1.0,
			     specificHeatRatio - 2.0);
  }
}

}
