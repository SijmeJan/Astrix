// -*-c++-*-
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./upwind.h"

namespace astrix {

__host__ __device__
void AddTemporalResidualSingle(int n,  
			       const int3* __restrict__ pTv,
			       const real4* __restrict__ pState, 
			       const real4* __restrict__ pVz, 
			       real4 *pTresLDA0, real4 *pTresLDA1, 
			       real4 *pTresLDA2,
			       const real2 *pTn1, const real2 *pTn2,
			       const real2 *pTn3, const real3 *pTl,
			       int nVertex, real G, real G1, real G2, real dt)
{
  const real zero  = (real) 0.0;
  const real onethird = (real) (1.0/3.0);
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

  real W00 = pState[v1].x;
  real W01 = pState[v1].y;
  real W02 = pState[v1].z;
  real W03 = pState[v1].w;
  real W10 = pState[v2].x;
  real W11 = pState[v2].y;
  real W12 = pState[v2].z;
  real W13 = pState[v2].w;
  real W20 = pState[v3].x;
  real W21 = pState[v3].y;
  real W22 = pState[v3].z;
  real W23 = pState[v3].w;
  
  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*onethird/dt;

  real ResTot0 = -(W00 + W10 + W20)*Adt;
  real ResTot1 = -(W01 + W11 + W21)*Adt;
  real ResTot2 = -(W02 + W12 + W22)*Adt;
  real ResTot3 = -(W03 + W13 + W23)*Adt;

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
  real alpha = G1*half*(utilde*utilde + vtilde*vtilde);

  real htilde = Z3/Z0;
  real ctilde = sqrt(G1*htilde - alpha);
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
    invN00*ResTot0 + invN01*ResTot1 + invN02*ResTot2 + invN03*ResTot3;  
  real Wtilde1 =
    invN10*ResTot0 + invN11*ResTot1 + invN12*ResTot2 + invN13*ResTot3;  
  real Wtilde2 =
    invN20*ResTot0 + invN21*ResTot1 + invN22*ResTot2 + invN23*ResTot3;  
  real Wtilde3 =
    invN30*ResTot0 + invN31*ResTot1 + invN32*ResTot2 + invN33*ResTot3;  

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
  pTresLDA0[n].x = 0.5*pTresLDA0[n].x + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA0[n].y = 0.5*pTresLDA0[n].y + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA0[n].z = 0.5*pTresLDA0[n].z + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA0[n].w = 0.5*pTresLDA0[n].w + ResLDA;

  // Second direction  
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

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
  pTresLDA1[n].x = 0.5*pTresLDA1[n].x + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA1[n].y = 0.5*pTresLDA1[n].y + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA1[n].z = 0.5*pTresLDA1[n].z + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA1[n].w = 0.5*pTresLDA1[n].w + ResLDA;
  
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
  pTresLDA2[n].x = 0.5*pTresLDA2[n].x + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP10(nx, ac, uc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP11(G1, G2, nx, uc, l1l2l3, l1l2, l3)
    -Wtilde2*eulerKMP12(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde3*eulerKMP13(G1, nx, ic, uc, l1l2l3, l1l2);
  pTresLDA2[n].y = 0.5*pTresLDA2[n].y + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP20(ny, ac, vc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP21(G1, nx, ny, uc, vc, l1l2l3, l1l2)
    -Wtilde2*eulerKMP22(G1, G2, ny, vc, l1l2l3, l1l2, l3)
    -Wtilde3*eulerKMP23(G1, ny, ic, vc, l1l2l3, l1l2);
  pTresLDA2[n].z = 0.5*pTresLDA2[n].z + ResLDA;

  ResLDA =
    -Wtilde0*eulerKMP30(ac, hc, wtilde, l1l2l3, l1l2)
    -Wtilde1*eulerKMP31(G1, nx, hc, uc, wtilde, l1l2l3, l1l2)
    -Wtilde2*eulerKMP32(G1, ny, hc, vc, wtilde, l1l2l3, l1l2)
    -Wtilde3*eulerKMP33(G1, ic, hc, wtilde, l1l2l3, l1l2, l3);
  pTresLDA2[n].w = 0.5*pTresLDA2[n].w + ResLDA;
}
  
__host__ __device__
void AddTemporalResidualSingle(int n,  
			       const int3* __restrict__ pTv,
			       const real* __restrict__ pState, 
			       const real* __restrict__ pVz, 
			       real *pTresLDA0, real *pTresLDA1, 
			       real *pTresLDA2,
			       const real2 *pTn1, const real2 *pTn2,
			       const real2 *pTn3, const real3 *pTl,
			       int nVertex, real G, real G1, real G2, real dt)
{
  const real zero  = (real) 0.0;
  const real half  = (real) 0.5;
  const real one = (real) 1.0;
  const real onethird = (real) (1.0/3.0);

  int v1 = pTv[n].x;
  int v2 = pTv[n].y;
  int v3 = pTv[n].z;
  while (v1 >= nVertex) v1 -= nVertex;
  while (v2 >= nVertex) v2 -= nVertex;
  while (v3 >= nVertex) v3 -= nVertex;
  while (v1 < 0) v1 += nVertex;
  while (v2 < 0) v2 += nVertex;
  while (v3 < 0) v3 += nVertex;

  real W0 = pState[v1];
  real W1 = pState[v2];
  real W2 = pState[v3];
  
  real Tl1 = pTl[n].x;
  real Tl2 = pTl[n].y;
  real Tl3 = pTl[n].z;

  // Calculate triangle area
  real s = half*(Tl1 + Tl2 + Tl3);
  real Adt = sqrt(s*(s - Tl1)*(s - Tl2)*(s - Tl3))*onethird/dt;

  real ResTot = -(W0 + W1 + W2)*Adt;
  
  // Average parameter vector
#if BURGERS == 1  
  real Zv0 = pVz[v1];
  real Zv1 = pVz[v2];
  real Zv2 = pVz[v3];

  real Z0 = (Zv0 + Zv1 + Zv2)*onethird;
  real vx = Z0;
  real vy = Z0;
#else
  real vx = one;
  real vy = zero;
#endif

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

  real Wtilde = invN*ResTot;
 
  // First direction
  real Tnx1 = pTn1[n].x;
  real Tny1 = pTn1[n].y;

  nx = half*Tnx1;
  ny = half*Tny1;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA0[n] = 0.5*pTresLDA0[n] - l1*Wtilde;
  
  // Second direction
  real Tnx2 = pTn2[n].x;
  real Tny2 = pTn2[n].y;

  nx = half*Tnx2;
  ny = half*Tny2;
  
  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA1[n] = 0.5*pTresLDA1[n] - l1*Wtilde;
  
  // Third direction
  real Tnx3 = pTn3[n].x;
  real Tny3 = pTn3[n].y;

  nx = half*Tnx3;
  ny = half*Tny3;

  l1 = half*(vx*nx + vy*ny + fabs(vx*nx + vy*ny));
  pTresLDA2[n] = 0.5*pTresLDA2[n] - l1*Wtilde;
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
devAddTemporalResidual(int nTriangle, const int3* __restrict__ pTv,
		       const realNeq* __restrict__ pState, 
		       const realNeq* __restrict__ pVz, 
		       realNeq *pTresLDA0, realNeq *pTresLDA1,
		       realNeq *pTresLDA2,
		       const real2 *pTn1, const real2 *pTn2,
		       const real2 *pTn3, const real3 *pTl,
		       int nVertex, real G, real G1, real G2, real dt)
{
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nTriangle) {
    AddTemporalResidualSingle(n, pTv, pState, pVz, pTresLDA0, pTresLDA1,
                              pTresLDA2, pTn1, pTn2, pTn3, pTl,
                              nVertex, G, G1, G2, dt);
    
    // Next triangle
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Calculate space-time LDA residue for all triangles; result in \a triangleResidueLDA.*/
//######################################################################

void Simulation::AddTemporalResidual(real dt)
{
  int nTriangle = mesh->GetNTriangle();
  int nVertex = mesh->GetNVertex();

  realNeq *pState = vertexStateOld->GetPointer();
  realNeq *pVz = vertexParameterVector->GetPointer();
  
  realNeq *pTresLDA0 = triangleResidueLDA->GetPointer(0);
  realNeq *pTresLDA1 = triangleResidueLDA->GetPointer(1);
  realNeq *pTresLDA2 = triangleResidueLDA->GetPointer(2);
  
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
				       devAddTemporalResidual, 
				       (size_t) 0, 0);

    devAddTemporalResidual<<<nBlocks, nThreads>>>
      (nTriangle, pTv, pState, pVz,
       pTresLDA0, pTresLDA1, pTresLDA2, 
       pTn1, pTn2, pTn3, pTl, nVertex, 
       specificHeatRatio, 
       specificHeatRatio - 1.0,
       specificHeatRatio - 2.0, dt);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nTriangle; n++) 
      AddTemporalResidualSingle(n, pTv, pState, pVz,
				pTresLDA0, pTresLDA1, pTresLDA2, 
				pTn1, pTn2, pTn3, pTl, nVertex, 
				specificHeatRatio, specificHeatRatio - 1.0,
				specificHeatRatio - 2.0, dt);
  }
}

}
