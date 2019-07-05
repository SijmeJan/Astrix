// -*-c++-*-
/*! \file diagnostics.cpp
\brief Constructor, destructor and initialization of the Diagnostics class

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>

#include "../../Common/definitions.h"
#include "../../Array/array.h"
#include "../../Mesh/mesh.h"
#include "./diagnostics.h"

namespace astrix {

//#########################################################################
//#########################################################################

template<class T, ConservationLaw CL>
Diagnostics<T, CL>::Diagnostics(Array<T> *state,
                                Array<real> *pot,
                                Mesh *mesh)
{
  // Size of output for various conservation laws
  int diagSize = 1;
  if (CL == CL_CART_ISO) diagSize = 2;
  if (CL == CL_CYL_ISO) diagSize = 2;
  if (CL == CL_CART_EULER) diagSize = 4;

  // Create result vector on host
  result = new Array<real>(1, 0, diagSize);
  real *pResult = result->GetPointer();


  pResult[0] = TotalMass(state, mesh);
  if (CL == CL_CART_ISO || CL == CL_CART_EULER)
    pResult[1] = KineticEnergy(state, mesh);
  if (CL == CL_CART_EULER) {
    pResult[2] = ThermalEnergy(state, pot, mesh);
    pResult[3] = TotalEnergy(state, mesh);
  }
  if (CL == CL_CYL_ISO) {
    pResult[1] = Torque(state, mesh);
  }
}

//#########################################################################
//#########################################################################

template<class T, ConservationLaw CL>
Diagnostics<T, CL>::~Diagnostics()
{
  delete result;
}

//##############################################################################
// Instantiate
//##############################################################################

template
Diagnostics<real, CL_ADVECT>::Diagnostics(Array<real> *state,
                                          Array<real> *pot,
                                          Mesh *mesh);
template
Diagnostics<real, CL_BURGERS>::Diagnostics(Array<real> *state,
                                           Array<real> *pot,
                                           Mesh *mesh);
template
Diagnostics<real3, CL_CART_ISO>::Diagnostics(Array<real3> *state,
                                             Array<real> *pot,
                                             Mesh *mesh);
template
Diagnostics<real3, CL_CYL_ISO>::Diagnostics(Array<real3> *state,
                                            Array<real> *pot,
                                            Mesh *mesh);
template
Diagnostics<real4, CL_CART_EULER>::Diagnostics(Array<real4> *state,
                                               Array<real> *pot,
                                               Mesh *mesh);

//##############################################################################

template Diagnostics<real, CL_ADVECT>::~Diagnostics();
template Diagnostics<real, CL_BURGERS>::~Diagnostics();
template Diagnostics<real3, CL_CART_ISO>::~Diagnostics();
template Diagnostics<real3, CL_CYL_ISO>::~Diagnostics();
template Diagnostics<real4, CL_CART_EULER>::~Diagnostics();

}  // namespace astrix
