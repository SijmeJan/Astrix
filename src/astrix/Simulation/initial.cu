// -*-c++-*-
/*! \file initial.cu
\brief Functions to set initial conditions

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#include <iostream>

#include <gsl/gsl_sf_bessel.h>
#include <boost/math/special_functions/bessel.hpp>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Common/cudaLow.h"
#include "../Common/inlineMath.h"
#include "./Param/simulationparameter.h"

namespace astrix {

//! Bump helper function
__host__ __device__
real funcFbump(real t)
{
  if (t <= 0.0f) return (real) 0.0;
  return exp(-(real) 1.0/(t + 1.0e-10));
}

//! Bump main function
__host__ __device__
real funcBump(real t)
{
  return funcFbump(t)/(funcFbump(t) + funcFbump((real) 1.0 - t));
}

//##############################################################################
/*! \brief Set initial conditions at vertex \a n

\param n Vertex to consider
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVpot Pointer to gravitational potential at vertices
\param *state Pointer to state vector (output)
\param G Ratio of specific heats
\param time Simulation time
\param Px Length of x domain
\param Py Length of y domain
\param denspow Density power law index (cylindrical isothermal only)
\param cs0 Soundspeed at x = 0 (cylindrical isothermal only)
\param cspow Soundspeed power law index (cylindrical isothermal only)*/
//##############################################################################

template <ConservationLaw CL>
__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
                      real *pVpot, real4 *state, real G, real time,
                      real Px, real Py, real denspow, real cs0, real cspow)
{
  // const real onethird = (real) (1.0/3.0);
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

  if (problemDef == PROBLEM_GAUSS) {
    real x = vertX;
    //real y = vertY;

    real X_VELOCITY = 2.0, pres = 100.0;
    real CENTRE = 0.2;
    real S,W,RHO_0 = 10.0,RHO_PULSE = 50.0;

    S = abs(CENTRE - x);
    W = 0.1;

    dens = RHO_PULSE*exp(-S*S/(W*W)) + RHO_0*(1.0-exp(-S*S/(W*W)));
    momx = dens*X_VELOCITY;
    momy = dens*0.0000000001;

    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
  }

  if (problemDef == PROBLEM_LINEAR) {
    real amp = (real) 1.0e-4;
    real k = two*M_PI;
    real c0 = one;
    real p0 = c0*c0/G;

    dens = one;
    momx = zero;
    momy = (real) 1.0e-10;

    dens += amp*cos(k*vertX);
    momx += amp*cos(k*vertX);
    p0   += amp*cos(k*vertX);

    ener = half*(Sq(momx) + Sq(momy))/dens + p0/(G - one);
  }

  if (problemDef == PROBLEM_VORTEX) {
    real x = vertX;
    real y = vertY;
    real vx = 1.0;
    real vy = 0.0;

    // Temperature
    real temp = 1.0;

    //for (int i = -1; i < 2; i++) {
    //real xc = five + time + i*Px;
      real xc = five + vx*time;
      real yc = five;

      real beta = five;

      real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

      temp -= (G - one)*Sq(beta)*exp(one - r*r)/(8.0f*G*M_PI*M_PI);

      vx -= half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
      vy += half*(x - xc)*beta*exp(half - half*r*r)/M_PI;
      //}

    dens = std::pow(temp, (real)(one/(G - one)));
    real pres = dens*temp;

    momx = (real)1.0e-10 + dens*vx;
    momy = (real)2.0e-10 + dens*vy;
    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
  }

  if (problemDef == PROBLEM_CYL) {
    dens = 1.0;
    momx = 2.0;
    momy = 1.0e-10;
    ener = half*(Sq(momx) + Sq(momy))/dens + 2.5f/(G - one);
  }

  if (problemDef == PROBLEM_KH) {
    real a = 4.0f;
    real smoothHat =
      funcBump(a*(vertY - 0.25f) + 0.5f)*funcBump(-a*(vertY - 0.75f) + 0.5f);

    dens = 1.0f + smoothHat;
    momx = dens*(-0.5f + smoothHat);
    momy = 0.0f;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 2.5f/(G - 1.0f);
  }

  if (problemDef == PROBLEM_SOD) {
    /*
    real v = 0.5;
    dens = 1.0;
    if (vertX > 0.25 + v*time)
      dens = 0.5;
    momx = dens*v;
    momy = 0.0;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 1.0f/(G - 1.0f);
    */

    /*
    real v = sqrt(G/(3.0 - G));
    if (vertX > 0.0) v = -v;

    dens = 1.0;
    momx = v;
    momy = 0.0;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + 1.0f/(G - 1.0f);

    if (abs(vertX) < abs(v)*time) {
      dens = 2.0;
      momx = 0.0;
      momy = 0.0;
      ener = (1.0f + dens*v*v)/(G - 1.0f);
    }
    */


    // Left state
    real dL = 1.0;
    real pL = 1.0;
    real cL = sqrt(G*pL/dL);

    // Right state
    real dR = 0.125;
    real pR = 0.1;
    real cR = sqrt(G*pR/dR);

    real G1 = G - 1.0;
    real m = G1/(G + 1.0);

    real pmid = 0.30313017805;

    real vpost = 2.0*sqrt(G)*(1.0 - std::pow(pmid, 0.5*G1/G))/G1;

    real dmidR = dR*(pmid + m*pR)/(pR + m*pmid);
    real dmidL = dL*std::pow(pmid/pL, 1.0/G);

    real umid = (pmid - pR)/sqrt(0.5*dR*((G + 1.0)*pmid + G1*pR));
    real vshock = dmidR*umid/(dmidR - dR);

    real x0 = 0.5;
    real x1 = x0 - cL*time;
    real x2 = x0 - cL*time*(std::pow(dmidL/dL, 0.5*G1) + m - 1.0)/m;
    real x3 = x0 + vpost*time;
    real x4 = x0 + vshock*time;

    dens = dL;
    momx = 1.0e-10f;
    momy = 1.0e-10f;
    real pres = pL;

    if (vertX > x1) {
      real c = m*(x0 - vertX)/time + (1.0 - m)*cL;
      dens = dL*std::pow(c/cL, 2.0/G1);
      momx = dens*((vertX - x1)*umid/(x2 - x1));
      momy = 1.0e-10f;
      pres = pL*std::pow(dens/dL, G);
    }
    if (vertX > x2) {
      dens = dmidL;
      momx = dens*umid;
      momy = 1.0e-10f;
      pres = pmid;
    }
    if (vertX > x3) {
      dens = dmidR;
      momx = dens*umid;
      momy = 1.0e-10f;
      pres = pmid;
    }
    if (vertX > x4) {
      dens = dR;
      momx = 1.0e-10f;
      momy = 1.0e-10f;
      pres = pR;
    }

    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + pres/G1;


  }

  if (problemDef == PROBLEM_NOH) {
    real x = vertX;
    real y = vertY;
    real r = sqrt(x*x + y*y);

    real pres = 1.0e-6;

    if (r < time/3.0) {
      dens = 16.0;
      momx = zero;
      momy = zero;
      pres = 16.0/3.0;
    } else {
      dens = one + time/(r + 1.0e-10);
      momx = dens*(-x/(r + 1.0e-10) + 1.0e-10);
      momy = dens*(-y/(r + 1.0e-10) - 2.0e-10);
    }

    ener = half*(Sq(momx) + Sq(momy))/dens + pres/(G - one);
  }

  if (problemDef == PROBLEM_BLAST) {
    // Interacting blast waves
    real p = 0.01f;
    if (vertX < 0.1f) p = 1000.0f;
    if (vertX > 0.9f) p = 100.0f;

    dens = 1.0f;
    momx = 1.0e-10f;
    momy = 1.0e-10f;
    ener = 0.5f*(Sq(momx)+Sq(momy))/dens + p/(G - 1.0f);
  }

  if (problemDef == PROBLEM_RIEMANN) {
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

  if (problemDef == PROBLEM_SOURCE) {
    real rhoBot = 1.0;
    real rhoTop = 2.0;
    real Ly = 1.0;
    dens = rhoBot + 0.5*(rhoTop - rhoBot)*(vertY + Ly)/Ly;
    momx = 1.0e-10;
    momy = 0.0;
    real pres = 1.0 - 0.1*(0.5*(rhoTop + rhoBot)*vertY +
                           0.25*vertY*vertY*(rhoTop - rhoBot)/Ly);
    ener = half*(Sq(momx) + Sq(momy))/dens + dens*pVpot[n] + pres/(G - one);

    /*
    real x = vertX;
    real y = vertY;

    real xc = five;
    real yc = five;

    real beta = five;

    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    real vx = -half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
    real vy = half*(x - xc)*beta*exp(half - half*r*r)/M_PI;

    dens = 1.0;
    momx = (real)1.0e-10 + dens*vx;
    momy = (real)2.0e-10 + dens*vy;
    real pres = 1.0;
    ener = half*(Sq(momx) + Sq(momy))/dens + dens*pVpot[n] + pres/(G - one);
    */

    /*
    real x = vertX;
    real y = vertY;

    real xc = five;
    real yc = five;

    real beta = five;

    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    real pot = y;//-0.125*beta*beta*exp(1.0 - r*r)/(M_PI*M_PI);

    dens = std::pow(1.0 - (G - 1.0)*pot/G, 1.0/(G - 1.0));
    momx = 1.0e-10;
    momy = 0.0;
    real pres = std::pow(dens, G);
    ener = half*(Sq(momx) + Sq(momy))/dens + dens*pVpot[n] + pres/(G - one);
    */
  }

  state[n].x = dens;
  state[n].y = momx;
  state[n].z = momy;
  state[n].w = ener;
}

//! Version for three equations
template <ConservationLaw CL>
__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
                      real *pVpot, real3 *state, real G, real time,
                      real Px, real Py, real denspow, real cs0, real cspow)
{
  real zero = (real) 0.0;
  real one = (real) 1.0;
  real five = (real) 5.0;
  real half = (real) 0.5;

  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = (real) 1.0;
  real momx = (real) 0.0;
  real momy = (real) 0.0;

  if (problemDef == PROBLEM_VORTEX) {
    real x = vertX;
    real y = vertY;
    real vx = one;
    real vy = zero;

    real xc = five + vx*time;
    real yc = five;

    real beta = five;

    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    dens = exp(-Sq(beta/M_PI)*exp(one - r*r)/8.0);
    vx -= half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
    vy += half*(x - xc)*beta*exp(half - half*r*r)/M_PI;

    momx = (real)1.0e-10 + dens*vx;
    momy = (real)2.0e-10 + dens*vy;
  }

  if (problemDef == PROBLEM_SOURCE) {
    //dens = exp(-0.1*vertY);

    real x = vertX;
    real y = vertY;

    real xc = five;
    real yc = five;

    real beta = five;

    real r = sqrt(Sq(x - xc)+Sq(y - yc)) + (real) 1.0e-10;

    real vx = -half*(y - yc)*beta*exp(half - half*r*r)/M_PI;
    real vy = half*(x - xc)*beta*exp(half - half*r*r)/M_PI;

    momx = (real)1.0e-10 + dens*vx;
    momy = (real)2.0e-10 + dens*vy;
  }

  if (problemDef == PROBLEM_DISC) {
    real x = vertX;
    real y = vertY;

    // Cylindrical radius
    real r = exp(x);

    dens = pow(r, denspow);
    real vx = 0.0e-10;
    // Angular momentum
    real vy = sqrt(1.0*r +
                   (denspow + 2.0*cspow)*cs0*cs0*pow(r, 4.0 + 2.0*cspow));

    // Linear perturbations
    real A = 1.0e-5;            // Amplitude
    // Surface density \propto r^{-a}
    real a = 2.0 - denspow;

    /*
    // NO GRAVITY
    real k = 4.0*M_PI/Px;
    real w = 0.5*sqrt(4.0*k*k + a*a - 20.0*a + 36.0)*cs0;
    real f = A*sin(k*x);
    real df = A*k*cos(k*x);

    real u1 = f*cos(w*time)/sqrt(dens);
    real d1 = -sqrt(dens)*(df + (1.0 - 0.5*a)*f)*sin(w*time)/w;
    real v1 = -r*r*f*sin(w*time)*2.0*sqrt(2.0-a)*cs0/(w*sqrt(dens));

    */

    /*
    // GRAVITY: needs denspow=0, soundspeed0=1/sqrt(3), soundspeedPower=-1.5
    real b0 = 5.40278878940795; //1.3196250078464;
    real c1 = 2.6674287565997314; //66.561534022766;
    real w = cs0*b0;

    real q = 2.0*b0*r*sqrt(r)/3.0;
    real g = A*sqrt(r)*(gsl_sf_bessel_J0(q) + c1*gsl_sf_bessel_Y0(q));
    real dg = 0.5*g/r - A*b0*r*(gsl_sf_bessel_J1(q) + c1*gsl_sf_bessel_Y1(q));

    real u1 = g*cos(w*time)/sqrt(dens*r);
    real d1 = sqrt(dens/r)*(0.5*(a - 1.0)*g - r*dg)*sin(w*time)/w;
    //real v1 = -0.5*g*sqrt(1.0 - (a + 1.0)*cs0*cs0)*sin(w*time)/(w*sqrt(dens));
    real v1 = 0.0;
    */

    // DISC: needs denspow=0, soundspeed0=0.1, soundspeedPower=-1.5
    real b0 = 20.829436532817862;
    real c1 = 2.831729237113438;
    real w = cs0*b0;

    real nu = sqrt(1.0 + 4.0/(cs0*cs0) - 8.0*a + a*a - 1.0)/3.0;
    real q = 2.0*b0*r*sqrt(r)/3.0;

    real Jplus = boost::math::cyl_bessel_j(nu, q);
    real Jmin  = boost::math::cyl_bessel_j(-nu, q);
    real dJplus = 0.5*(boost::math::cyl_bessel_j(nu - 1.0, q) -
                       boost::math::cyl_bessel_j(nu + 1.0, q));
    real dJmin  = 0.5*(boost::math::cyl_bessel_j(-nu - 1.0, q) -
                       boost::math::cyl_bessel_j(-nu + 1.0, q));

    real g = A*sqrt(r)*(Jmin + c1*Jplus);

    real dg = 0.5*g/r + A*b0*r*(dJmin + c1*dJplus);

    real u1 = g*cos(w*time)/sqrt(dens*r);
    real d1 = sqrt(dens/r)*(0.5*(a - 1.0)*g - r*dg)*sin(w*time)/w;
    real v1 = -0.5*g*sqrt(1.0 - (a + 1.0)*cs0*cs0)*sin(w*time)/(w*sqrt(dens));
    //real v1 = 0.0;

    /*
#ifndef __CUDA_ARCH__
    std::cout << "Period: " << 2.0*M_PI/w << std::endl;
    std::cout << u1 << " " << d1 << " " << v1 << std::endl;
    //std::cout << vy << std::endl;
    //std::cout << r + (denspow + 2.0*cspow)*cs0*cs0*pow(r, 4.0 + 2.0*cspow)
    //          << std::endl;
    int qq; std::cin >> qq;
#endif
    */

    dens += d1;
    vx += u1;
    vy += v1;


    //if (r > 0.5 && r < 2.4)
    //  dens *= (1.0 + 0.1*exp(-Sq(y - 3.14)/0.01));
    momx = dens*vx;
    momy = dens*vy;
  }

  state[n].x = dens;
  state[n].y = momx;
  state[n].z = momy;
}

//! Version for single equation
template <ConservationLaw CL>
__host__ __device__
void SetInitialSingle(int n, const real2 *pVc, ProblemDefinition problemDef,
                      real *pVpot, real *state, real G, real time,
                      real Px, real Py, real denspow, real cs0, real cspow)
{
  real vertX = pVc[n].x;
  real vertY = pVc[n].y;

  real dens = (real) 1.0;

  if (problemDef == PROBLEM_LINEAR) {
    for (int i = -1; i < 2; i++) {
      real x = vertX;
      real xc = time + i*Px;

      if (fabs(x - xc) <= (real) 0.25) dens += Sq(cos(2.0*M_PI*(x - xc)));
    }
  }

  if (problemDef == PROBLEM_VORTEX) {
    real half = (real) 0.5;

    for (int i = -1; i < 2; i++) {
      real x = vertX;
      real y = vertY;

      real xc = half + i*Px + time;
      real yc = half;

      real r = sqrt(Sq(x - xc) + Sq(y - yc));

      if (r <= (real) 0.25) dens += Sq(cos(2.0*M_PI*r));
    }
  }

  if (problemDef == PROBLEM_RIEMANN) {
    real x = vertX;
    real y = vertY;

    dens = (real) 1.0e-10;
    if (x  > (real) -0.6 && x < (real) - 0.1 &&
        y > (real) -0.35 && y < (real) 0.15) dens += 1.0;
  }

  if (problemDef == PROBLEM_SOURCE) {
    if (CL == CL_ADVECT) {
      // Source = u, periodic
      //dens = cos(2.0*M_PI*(vertX - time))*exp(time);

      // Source = u, not periodic
      //dens = exp(0.5*(vertX + time));

      // Source = u^2, not periodic
      dens = -(real) 2.0/(vertX + time + 2.0);
    }
    if (CL == CL_BURGERS) {
      // Source = u
      //dens = vertX/((real) 1.0 + exp(-time));
      //dens = 0.5*exp(2.0*time)*(sqrt(1.0 + 4.0*vertX*exp(-2.0*time)) - 1.0);

      // Source = u^2
      dens = -((real) 1.0 + exp(vertX))/((real) 1.0 + time);
    }
  }

  state[n] = dens;
}

//######################################################################
/*! \brief Kernel setting initial conditions

\param nVertex Total number of vertices in Mesh
\param *pVc Pointer to coordinates of vertices
\param problemDef Problem definition
\param *pVertexPotential Pointer to gravitational potential at vertices
\param *state Pointer to state vector (output)
\param G Ratio of specific heats
\param time Current simulation time
\param Px Length of x domain
\param Py Length of y domain
\param denspow Density power law index (cylindrical isothermal only)
\param cs0 Soundspeed at x = 0 (cylindrical isothermal only)
\param cspow Soundspeed power law index (cylindrical isothermal only)
\param *pVertexFlag Pointer to flags indicating whether vertex is part of boundary
\param boundaryFlag Flag whether to set only boundary vertices*/
//######################################################################

template<class realNeq, ConservationLaw CL>
__global__ void
devSetInitial(int nVertex, const real2 *pVc, ProblemDefinition problemDef,
              real *pVertexPotential, realNeq *state,
              real G, real time, real Px, real Py,
              real denspow, real cs0, real cspow,
              const int *pVertexFlag, int boundaryFlag)
{
  // n = vertex number
  int n = blockIdx.x*blockDim.x + threadIdx.x;

  while (n < nVertex) {
    // Do only specific vertices
    int do_flag = 1;
    if (boundaryFlag != 0)
      if (pVertexFlag[n] == 0) do_flag = 0;

    if (do_flag)
      SetInitialSingle<CL>(n, pVc, problemDef, pVertexPotential, state,
                           G, time, Px, Py, denspow, cs0, cspow);

    n += blockDim.x*gridDim.x;
  }
}

//######################################################################
/*! Set initial conditions for all vertices based on problemSpecification

\param time Current simulation time, to get exact solution at this time (if available)
\param boundaryFlag Flag whether to set only boundary vertices*/
//######################################################################

template <class realNeq, ConservationLaw CL>
void Simulation<realNeq, CL>::SetInitial(real time,
                                         int boundaryFlag)
{
  int nVertex = mesh->GetNVertex();

  realNeq *state = vertexState->GetPointer();
  real *pVertexPotential = vertexPotential->GetPointer();
  const real2 *pVc = mesh->VertexCoordinatesData();

  real Px = mesh->GetPx();
  real Py = mesh->GetPy();
  real G = simulationParameter->specificHeatRatio;
  ProblemDefinition p = simulationParameter->problemDef;

  const real denspow = simulationParameter->densityPower;
  const real cs0 = simulationParameter->soundspeed0;
  const real cspow = simulationParameter->soundspeedPower;

  const int *pVertexFlag = mesh->VertexBoundaryFlagData();

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads,
                                       devSetInitial<realNeq, CL>,
                                       (size_t) 0, 0);

    devSetInitial<realNeq, CL><<<nBlocks, nThreads>>>
      (nVertex, pVc, p, pVertexPotential, state, G, time, Px, Py,
       denspow, cs0, cspow,
       pVertexFlag, boundaryFlag);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int n = 0; n < nVertex; n++) {
      // Do only specific vertices
      int do_flag = 1;
      if (boundaryFlag != 0) {
        if (pVertexFlag[n] == 0) do_flag = 0;
      }

      if (do_flag)
        SetInitialSingle<CL>(n, pVc, p, pVertexPotential,
                             state, G, time, Px, Py,
                             denspow, cs0, cspow);
    }

  }

  try {
    // Add KH eigenvector
    if (p == PROBLEM_KH)
      KHAddEigenVector();
    if (p == PROBLEM_SOURCE && CL == CL_CART_EULER)
      RTAddEigenVector();
  }
  catch (...) {
    std::cout << "Warning: reading eigenvector file failed!" << std::endl;
    std::cout << "Running simulation without adding eigenvector!"
              << std::endl;
  }
}

//##############################################################################
// Instantiate
//##############################################################################

template void Simulation<real, CL_ADVECT>::SetInitial(real time,
                                                      int boundaryFlag);
template void Simulation<real, CL_BURGERS>::SetInitial(real time,
                                                       int boundaryFlag);
template void Simulation<real3, CL_CART_ISO>::SetInitial(real time,
                                                         int boundaryFlag);
template void Simulation<real3, CL_CYL_ISO>::SetInitial(real time,
                                                        int boundaryFlag);
template void Simulation<real4, CL_CART_EULER>::SetInitial(real time,
                                                           int boundaryFlag);

}  // namespace astrix
