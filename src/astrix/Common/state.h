/*! \file state.h
    \brief Header file for templated operations on state vector.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_STATE_H
#define ASTRIX_STATE_H

namespace astrix {

  namespace state {

    template<class realNeq, ConservationLaw CL>
      inline real GetDensity(realNeq state) {
      return 0.0;
    }
    template<>
      inline real GetDensity<real, CL_ADVECT>(real state) {
      return state;
    }
    template<>
      inline real GetDensity<real, CL_BURGERS>(real state) {
      return state;
    }
    template<>
      inline real GetDensity<real3, CL_CART_ISO>(real3 state) {
      return state.x;
    }
    template<>
      inline real GetDensity<real4, CL_CART_EULER>(real4 state) {
      return state.x;
    }

    template<class realNeq, ConservationLaw CL>
      inline real GetMomX(realNeq state) {
      return 0.0;
    }
    template<>
      inline real GetMomX<real3, CL_CART_ISO>(real3 state) {
      return state.y;
    }
    template<>
      inline real GetMomX<real4, CL_CART_EULER>(real4 state) {
      return state.y;
    }

    template<class realNeq, ConservationLaw CL>
      inline real GetMomY(realNeq state) {
      return 0.0;
    }
    template<>
      inline real GetMomY<real3, CL_CART_ISO>(real3 state) {
      return state.z;
    }
    template<>
      inline real GetMomY<real4, CL_CART_EULER>(real4 state) {
      return state.z;
    }

    template<class realNeq, ConservationLaw CL>
      inline real GetEnergy(realNeq state) {
      return 0.0;
    }
    template<>
      inline real GetEnergy<real4, CL_CART_EULER>(real4 state) {
      return state.w;
    }


    template<class realNeq, ConservationLaw CL>
      inline void SetDensity(realNeq& state,
                             real dens) { }
    template<>
      inline void SetDensity<real, CL_ADVECT>(real& state, real dens) {
      state = dens;
    }
    template<>
      inline void SetDensity<real, CL_BURGERS>(real& state, real dens) {
      state = dens;
    }
    template<>
      inline void SetDensity<real3, CL_CART_ISO>(real3& state, real dens) {
      state.x = dens;
    }
    template<>
      inline void SetDensity<real4, CL_CART_EULER>(real4& state, real dens) {
      state.x = dens;
    }

    template<class realNeq, ConservationLaw CL>
      inline void SetMomX(realNeq& state, real momx) { }
    template<>
      inline void SetMomX<real3, CL_CART_ISO>(real3& state, real momx) {
      state.y = momx;
    }
    template<>
      inline void SetMomX<real4, CL_CART_EULER>(real4& state, real momx) {
      state.y = momx;
    }

    template<class realNeq, ConservationLaw CL>
      inline void SetMomY(realNeq& state, real momy) { }
    template<>
      inline void SetMomY<real3, CL_CART_ISO>(real3& state, real momy) {
      state.z = momy;
    }
    template<>
      inline void SetMomY<real4, CL_CART_EULER>(real4& state, real momy) {
      state.z = momy;
    }

    template<class realNeq, ConservationLaw CL>
      inline void SetEnergy(realNeq& state, real ener) { }
    template<>
      inline void SetEnergy<real4, CL_CART_EULER>(real4& state, real ener) {
      state.w = ener;
    }

  }

}

#endif  // ASTRIX_STATE_H
