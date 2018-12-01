// -*-c++-*-
/*! \file bicgstab.cpp

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cuda_runtime_api.h>
#include <iostream>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "../Device/device.h"

namespace astrix {

void LinSys::BiCGStab(Array<real> *x, Array<real> *b, int l)
{
  // r = b - A*x
  MultiplyByMatrix(x, v);
  r->SetToDiff(b, v);

  // rhat = r
  rhat->SetEqual(r);

  real rho0  = 1.0;
  real alpha = 0.0;
  real omega = 1.0;

  u->SetToValue(0.0);

  for (int k = 0; k <= maxIter; k += l) {
    // Array<real> *uhat = new Array<real>(1,
    real rho1 = r->InnerProduct(rhat);
    real beta = alpha*rho1/rho0;

    // u = r - beta*u
    CalcU(beta);

    // v = A*u
    MultiplyByMatrix(u, v);

    real gamma = v->InnerProduct(rhat);

    // alpha = rho/gamma
    alpha = rho0/gamma;

    // r = ri -alpha*v
    CalcR(alpha);

    // s = A*r
    MultiplyByMatrix(r, s);

    // x = xi + alpha*u
    CalcX(alpha);

    // rho1 = (rhat, s)
    rho1 = s->InnerProduct(rhat);

    beta = alpha*rho1/rho0;
    rho0 = rho1;

    // v = s - beta*v
    CalcV(beta);

    // w = A*v
    MultiplyByMatrix(v, w);

    // gamma = (w, rhat)
    real gamma = w->InnerProduct(rhat);

    alpha = rho0/gamma;

    // u = r - beta*u
    // r = r - alpha*v
    // s = s - alpha*w
    // t = A*s

    // omega1 = (r, s)
    // mu = (s, s)
    // nu = (s, t)
    // tau = (t, t)
    // omega2 = (r, t)
    // tau = tau - nu*nu/mu
    // omega2 = (omega2 - nu*omega1/mu)/tau
    // omega1 = (omega1 - nu*omega2)/mu
    // xi+2 = x + omega1*r + omega2*s + alpha*u
    // ri+2 = r - omega1*s - omega2*t
    // If accurate enough quit
    // u = u - omega1*v - omega2*w
  }
}

}  // namespace astrix
