// -*-c++-*-
/*! \file initial.cu
\brief Functions to set initial conditions*/
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"

namespace astrix {

//##############################################################################
/*! \brief Set initial conditions at vertex \a n

\param n Vertex to consider
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVpot Pointer to gravitational potential at vertices
\param *pState Pointer to state vector (output)
\param G Ratio of specific heats*/
//##############################################################################
  
__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
		      real *pVpot, real4 *state, real G)
{
  const real onethird = (real) (1.0/3.0);
  const real zero = (real) 0.0;
  const real half = (real) 0.5;
  const real one = (real) 1.0;
  const real two = (real) 2.0;
  const real five = (real) 5.0;
  
  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = zero;
  real momx = zero;
  real momy = zero;
  real ener = zero;
  
  if(problemDef == PROBLEM_LINEAR){
    real amp = (real) 1.0e-4;
    real k = two*M_PI;
    real c0 = one;
    real p0 = one/G;
    
    dens = one;
    momx = (real) 1.0e-10;
    momy = (real) 2.0e-10;
    
    momx += amp*cos(k*vertX);
    momy += amp*c0*cos(k*vertX);
    
    ener = 
      half*(Sq(momx) + Sq(momy))/dens + 
      (p0 + c0*c0*amp*cos(k*vertX))/(G - one);
  }
  
  if(problemDef == PROBLEM_VORTEX){
    /*
    real xc = half;
    real yc = half;
    
    real x = vertX;
    real y = vertY;
    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    real omega = five*r;
    if (r > 0.2f) omega = two - five*r;
    if (r > 0.4f) omega = zero;

    real vx = -omega*(y - yc)/r;
    real vy =  omega*(x - xc)/r;

    real pres = five + 12.5f*r*r;
    if (r > 0.2f)
      pres = 9.0f - 4.0f*log(0.2f) + 12.5f*r*r - 20.0f*r + 4.0f*log(r);
    if (r > 0.4f) pres = 3.0f + 4.0f*log(two);

    dens = one;
    momx = (real) 1.0e-10 + dens*vx;
    momy = (real) 2.0e-10 + dens*vy;
    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
    */

    real xc = zero;
    real yc = zero;

    real x = vertX;
    real y = vertY;
    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;
    real d = 4.0*M_PI*r;

    real w = 15.0f*(cos(d) + one);
    if (r >= 0.25) w = 0;
    real vx = -y*w + 6.0f;
    real vy = x*w;

    dens = 1.4;
    
    real pm = 100.0;
    real C = -15.0*15.0*dens*(0.75*M_PI*M_PI - 2.0 + 0.125)/(16.0*M_PI*M_PI);
    real dp =
      (2.0*cos(d) + 2.0*d*sin(d) + 0.125*cos(2.0*d) + 0.25*d*sin(2.0*d) +
       0.75*d*d)*15.0*15.0*dens/(16.0*M_PI*M_PI) + C;
    if (r >= 0.25) dp = 0.0;
    real pres = pm + dp;
    
    momx = (real) 1.0e-10 + dens*vx;
    momy = (real) 2.0e-10 + dens*vy;
    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);    
  }

  if(problemDef == PROBLEM_YEE){
    real xc = zero;
    real yc = zero;

    real beta = five;
    
    real x = vertX;
    real y = vertY;
    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    real T = one - (G - one)*Sq(beta)*exp(one - r*r)/(8.0f*G*M_PI*M_PI);

    real vx = -half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
    real vy =  half*(x - xc)*beta*exp(half - half*r*r)/M_PI;

    dens = std::pow(T, (real)(one/(G - one)));
    momx = (real)1.0e-10 + dens*(vx + 2.0/sqrt(5.0));
    momy = (real)2.0e-10 + dens*(vy + 1.0/sqrt(5.0));
    ener = half*(Sq(momx) + Sq(momy))/dens + dens*T/(G - one);
  }

  if(problemDef == PROBLEM_RT){
    real vel_amp = zero;
    real rhoH = two;
    real rhoL = one;
    
    real yPert = 0.01f*cosf(6.0f*M_PI*vertX);

    real d = 0.0000005f;
    real xi = (vertY - yPert)*M_PI/d;
    if (abs(xi) > half*M_PI) xi = half*M_PI*xi/abs(xi);

    real y = vertY;
    real y1 = yPert - half*d;
    real y2 = yPert + half*d;
    
    real I = rhoL*y;
    if (y > y1)
      I = rhoL*y1 + half*(rhoL + rhoH)*(y - y1) -
	half*(rhoH - rhoL)*cos(xi)*(d/M_PI);
    if (y > y2)
      I = rhoL*y1 + half*(rhoL + rhoH)*(y2 - y1) + rhoH*(y - y2);

    real p = 2.5f - 0.1f*I; // 2.5f - vertW0[n]*pVertexPotential[n]
    
    dens =  half*(rhoL + rhoH) + half*(rhoH - rhoL)*sin(xi);
    momx = (real) 1.0e-15;
    momy = dens*vel_amp*0.25f*
      (one + cosf(4.0f*(real)(M_PI)*vertX))*
      (one + cosf((one + onethird)*(real)(M_PI)*vertY)) + (real) 1.1e-15;
    ener = half*(Sq(momx) + Sq(momy))/dens + p/(G - one);
  }
  
  if (problemDef == PROBLEM_KH) {
    /*
    real vel_amp = 0.1;
    real dy = 0.005;
    
    vertW0[n] = 1.0f -
      atan((vertY - 0.25f)/dy)/M_PI +
      atan((vertY + 0.25f)/dy)/M_PI;
    vertW1[n] = vertW0[n]*(0.5f +
			     atan((vertY - 0.25f)/dy)/M_PI -
			     atan((vertY + 0.25f)/dy)/M_PI);
    vertW2[n] = 
      vertW0[n]*vel_amp*sinf(4.0f*(real)(M_PI)*vertX)*
      (exp(-Sq(vertY-0.25f)/Sq(0.05)) +
       exp(-Sq(vertY+0.25f)/Sq(0.05)));
    vertW3[n] = 
      0.5f*(Sq(vertW1[n])+Sq(vertW2[n]))/vertW0[n] +
      2.5f/((real)(G)-1.0f);
    */

    real kx = one;
    real ky = two;
    real r = kx*vertX + ky*vertY;
    real s = ky*vertX - kx*vertY;
    
    real uBulk = one;
    real d1 = one;
    real d2 = two;
    real dm = half*(d1 - d2);
    real L = 0.025f;

    // Ramp from d1 to 0.5*(d1+d2)
    real d = d1 - dm*exp((r - 0.25f)/L);
    // Ramp from 0.5*(d1+d2) to d2
    if (r > 0.25f) d = d2 + dm*exp((-r + 0.25f)/L);
    // Ramp from d2 to 0.5*(d1+d2)
    if (r > 0.50f) d = d2 + dm*exp(-(0.75f - r)/L);
    // Ramp from 0.5*(d1+d2) to d1
    if (r > 0.75f) d = d1 - dm*exp(-(r - 0.75f)/L);
    for (int nn = 1; nn < kx + ky; nn++) {
      real N = (real) nn;
      if (r > N + 0.00f) d = d1 - dm*exp(-(N + 0.25f - r)/L);
      if (r > N + 0.25f) d = d2 + dm*exp((-r + N + 0.25f)/L);
      if (r > N + 0.50f) d = d2 + dm*exp(-(N + 0.75f - r)/L);
      if (r > N + 0.75f) d = d1 - dm*exp(-(r - N - 0.75f)/L);
    }
    
    real u1 = half;
    real u2 = -half;
    real um = half*(u1 - u2);
    
    real u = u1 - um*exp((r - 0.25f)/L);
    if (r > 0.25f) u = u2 + um*exp((-r + 0.25f)/L);
    if (r > 0.50f) u = u2 + um*exp(-(0.75f - r)/L);
    if (r > 0.75f) u = u1 - um*exp(-(r - 0.75f)/L);
    for (int nn = 1; nn < kx + ky; nn++) {
      real N = (real) nn;
      if (r > N + 0.00f) u = u1 - um*exp(-(N + 0.25f - r)/L);
      if (r > N + 0.25f) u = u2 + um*exp((-r + N + 0.25f)/L);
      if (r > N + 0.50f) u = u2 + um*exp(-(N + 0.75f - r)/L);
      if (r > N + 0.75f) u = u1 - um*exp(-(r - N + 0.75f)/L);
    }

    real v = 0.01f*sin(two*M_PI*s);

    u = zero;
    v = zero;
    
    u += uBulk;
    
    real cosa = kx/sqrt(kx*kx + ky*ky);
    real sina = ky/sqrt(kx*kx + ky*ky);

    real vx = u*sina + v*cosa;
    real vy = -u*cosa + v*sina;

    dens = d;
    momx = d*vx;
    momy = d*vy;
    ener = half*d*(vx*vx + vy*vy) + 2.5f/(G - one);
  }
  
  if (problemDef == PROBLEM_SOD) {
    if (vertX < 0.5f) {
      dens = 1.0f;
      momx = 1.0e-10f;
      momy = 1.0e-10f;
      ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 1.0f/(G - 1.0f);
    } else {
      dens = 0.125f;
      momx = 1.2e-10f;
      momy = 2.0e-10f;
      ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 0.1f/(G - 1.0f);
    }
  }
  
  if(problemDef == PROBLEM_BLAST){
    // Interacting blast waves
    real p = 0.01f;
    if (vertX < 0.1f) p = 1000.0f;
    if (vertX > 0.9f) p = 100.0f;
    
    dens = 1.0f;
    momx = 1.0e-10f;
    momy = 1.0e-10f;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + p/(G - 1.0f);
  }
  
  if(problemDef == PROBLEM_RIEMANN){
    real f = 1.0f - (vertX > 0.8f);
    real g = 1.0f - (vertY > 0.8f);

    // CASE 3
    real densRB = 1.5f;
    real momxRB = 1.0e-10f;
    real momyRB = 1.0e-10f;
    real enerRB = 0.5f*(Sq(momxRB) + Sq(momyRB))/densRB + 1.5f/(G - 1.0f);

    real densLB = 0.5322581f;
    real momxLB = 1.2060454f*densLB;
    real momyLB = 0.0f;
    real enerLB = 0.5f*(Sq(momxLB) + Sq(momyLB))/densLB + 0.3f/(G - 1.0f);
    
    real densRO = 0.5322581f;
    real momxRO = 0.0f;
    real momyRO = 1.2060454f*densRO;
    real enerRO = 0.5f*(Sq(momxRO) + Sq(momyRO))/densRO + 0.3f/(G - 1.0f);

    real densLO = 0.1379928f;
    real momxLO = 1.2060454f*densLO;
    real momyLO = 1.2060454f*densLO;
    real enerLO = 0.5f*(Sq(momxLO) + Sq(momyLO))/densLO + 0.0290323f/(G - 1.0f);
    
    /*
    // CASE 6 
    real densRB = 1.0f;
    real momxRB = 0.75f*densRB;
    real momyRB = -0.5f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 2.0f;
    real momxLB = 0.75f*densLB;
    real momyLB = 0.5f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 1.0f/(G-1.0f);
    
    real densRO = 3.0f;
    real momxRO = -0.75f*densRO;
    real momyRO = -0.5f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 1.0f/(G-1.0f);

    real densLO = 1.0f;
    real momxLO = -0.75f*densLO;
    real momyLO = 0.5f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 1.0f/(G-1.0f);
    */
    /*
    // CASE 15 
    real densRB = 1.0f;
    real momxRB = 0.1f*densRB;
    real momyRB = -0.3f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 0.5197f;
    real momxLB = -0.6259f*densLB;
    real momyLB = -0.3f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 0.4f/(G-1.0f);
    
    real densRO = 0.5313f;
    real momxRO = 0.1f*densRO;
    real momyRO = 0.4276f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 0.4f/(G-1.0f);

    real densLO = 0.8f;
    real momxLO = 0.1f*densLO;
    real momyLO = -0.3f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 0.4f/(G-1.0f);
    */
    /*
    // Case 17
    real densRB = 1.0f;
    real momxRB = 1.0e-10f*densRB;
    real momyRB = -0.4f*densRB;
    real enerRB = 0.5f*(Sq(momxRB)+Sq(momyRB))/densRB + 1.0f/(G-1.0f);

    real densLB = 2.0f;
    real momxLB = 1.0e-10f*densLB;
    real momyLB = -0.3f*densLB;
    real enerLB = 0.5f*(Sq(momxLB)+Sq(momyLB))/densLB + 1.0f/(G-1.0f);
    
    real densRO = 0.5197f;
    real momxRO = 1.0e-10f*densRO;
    real momyRO = -1.1259f*densRO;
    real enerRO = 0.5f*(Sq(momxRO)+Sq(momyRO))/densRO + 0.4f/(G-1.0f);

    real densLO = 1.0625f;
    real momxLO = 1.0e-10f*densLO;
    real momyLO = 0.2145f*densLO;
    real enerLO = 0.5f*(Sq(momxLO)+Sq(momyLO))/densLO + 0.4f/(G-1.0f);
    */
    
    dens =
      densRB*(1.0f - f)*(1.0f - g) + densLB*f*(1.0f - g) +
      densRO*(1.0f - f)*g + densLO*f*g;
    momx =
      momxRB*(1.0f - f)*(1.0f - g) + momxLB*f*(1.0f - g) +
      momxRO*(1.0f - f)*g + momxLO*f*g;
    momy =
      momyRB*(1.0f - f)*(1.0f - g) + momyLB*f*(1.0f - g) +
      momyRO*(1.0f - f)*g + momyLO*f*g;
    ener =
      enerRB*(1.0f - f)*(1.0f - g) + enerLB*f*(1.0f - g) +
      enerRO*(1.0f - f)*g + enerLO*f*g; 
  }

  state[n].x = dens;
  state[n].y = momx;
  state[n].z = momy;
  state[n].w = ener;
}

__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
		      real *pVpot, real *state, real G)
{
  real half = (real) 0.5;
  
  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = (real) 1.0;
  
  if(problemDef == PROBLEM_ADVECT) {
    real x = vertX;
    real y = vertY;
    real r = sqrt(Sq(x - half) + Sq(y - half));

    if (r <= (real) 0.25) dens += Sq(cos(2.0*M_PI*r));
  }

  state[n] = dens;
}

//######################################################################
/*! \brief Kernel setting initial conditions

\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVpot Pointer to gravitational potential at vertices
\param *pState Pointer to state vector (output)
\param G Ratio of specific heats*/
//######################################################################

__global__ void 
devSetInitial(int nVertex, const real2 *pVc, ProblemDefinition problemDef,
	      real *pVertexPotential, realNeq *state, real G)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x; 

  while(n < nVertex){
    SetInitialSingle(n, pVc, problemDef, pVertexPotential, state, G);
 
    n += blockDim.x*gridDim.x;
  }
}
  
//######################################################################
/*! Set initial conditions for all vertices based on problemSpecification*/
//######################################################################

void Simulation::SetInitial()
{
  int nVertex = mesh->GetNVertex();

  realNeq *state = vertexState->GetPointer();
  real *pVertexPotential = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();
  
  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
				       devSetInitial, 
				       (size_t) 0, 0);

    devSetInitial<<<nBlocks, nThreads>>>
      (nVertex, pVc, problemDef, pVertexPotential, state, specificHeatRatio);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for(int n = 0; n < nVertex; n++) 
      SetInitialSingle(n, pVc, problemDef, pVertexPotential, state,
		       specificHeatRatio);
  }
}

}
