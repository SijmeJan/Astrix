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

    //! Return density from state vector: by default, return zero
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline real GetDensity(realNeq state) {
      return 0.0;
    }
    //! Return density from state vector: return scalar in case of scalar advect
    template<>
      __host__ __device__
      inline real GetDensity<real, CL_ADVECT>(real state) {
      return state;
    }
    //! Return density from state vector: return scalar in case of Burgers
    template<>
      __host__ __device__
      inline real GetDensity<real, CL_BURGERS>(real state) {
      return state;
    }
    //! Return density from state vector in case of isothermal hydro
    template<>
      __host__ __device__
      inline real GetDensity<real3, CL_CART_ISO>(real3 state) {
      return state.x;
    }
    //! Return density from state vector in case of Euler equations
    template<>
      __host__ __device__
      inline real GetDensity<real4, CL_CART_EULER>(real4 state) {
      return state.x;
    }

    //! Return x momentum from state vector: by default, return zero
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline real GetMomX(realNeq state) {
      return 0.0;
    }
    //! Return x momentum from state vector in case of isothermal hydro
    template<>
      __host__ __device__
      inline real GetMomX<real3, CL_CART_ISO>(real3 state) {
      return state.y;
    }
    //! Return x momentum from state vector in case of Euler equations
    template<>
      __host__ __device__
      inline real GetMomX<real4, CL_CART_EULER>(real4 state) {
      return state.y;
    }

    //! Return y momentum from state vector: by default, return zero
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline real GetMomY(realNeq state) {
      return 0.0;
    }
    //! Return y momentum from state vector in case of isothermal hydro
    template<>
      __host__ __device__
      inline real GetMomY<real3, CL_CART_ISO>(real3 state) {
      return state.z;
    }
    //! Return y momentum from state vector in case of Euler equations
    template<>
      __host__ __device__
      inline real GetMomY<real4, CL_CART_EULER>(real4 state) {
      return state.z;
    }

    //! Return energy from state vector: by default, return zero
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline real GetEnergy(realNeq state) {
      return 0.0;
    }
    //! Return energy from state vector in case of Euler equations
    template<>
      __host__ __device__
      inline real GetEnergy<real4, CL_CART_EULER>(real4 state) {
      return state.w;
    }

    //! Set density in state vector: by default, do nothing
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline void SetDensity(realNeq& state,
                             real dens) { }
    //! Set density in state vector: set scalar in case of advection
    template<>
      __host__ __device__
      inline void SetDensity<real, CL_ADVECT>(real& state, real dens) {
      state = dens;
    }
    //! Set density in state vector: set scalar in case of Burgers
    template<>
      __host__ __device__
      inline void SetDensity<real, CL_BURGERS>(real& state, real dens) {
      state = dens;
    }
    //! Set density in state vector in case of isothermal hydrodynamics
    template<>
      __host__ __device__
      inline void SetDensity<real3, CL_CART_ISO>(real3& state, real dens) {
      state.x = dens;
    }
    //! Set density in state vector in case of Euler equations
    template<>
      __host__ __device__
      inline void SetDensity<real4, CL_CART_EULER>(real4& state, real dens) {
      state.x = dens;
    }

    //! Set x momentum in state vector: by default, do nothing
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline void SetMomX(realNeq& state, real momx) { }
    //! Set x momentum in state vector in case of isothermal hydrodynamics
    template<>
      __host__ __device__
      inline void SetMomX<real3, CL_CART_ISO>(real3& state, real momx) {
      state.y = momx;
    }
    //! Set x momentum in state vector in case of Euler equations
    template<>
      __host__ __device__
      inline void SetMomX<real4, CL_CART_EULER>(real4& state, real momx) {
      state.y = momx;
    }

    //! Set y momentum in state vector: by default, do nothing
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline void SetMomY(realNeq& state, real momy) { }
    //! Set y momentum in state vector in case of isothermal hydrodynamics
    template<>
      __host__ __device__
      inline void SetMomY<real3, CL_CART_ISO>(real3& state, real momy) {
      state.z = momy;
    }
    //! Set y momentum in state vector in case of Euler equations
    template<>
      __host__ __device__
      inline void SetMomY<real4, CL_CART_EULER>(real4& state, real momy) {
      state.z = momy;
    }

    //! Set energy in state vector: by default, do nothing
    template<class realNeq, ConservationLaw CL>
      __host__ __device__
      inline void SetEnergy(realNeq& state, real ener) { }
    //! Set energy in state vector in case of Euler equations
    template<>
      __host__ __device__
      inline void SetEnergy<real4, CL_CART_EULER>(real4& state, real ener) {
      state.w = ener;
    }

  }  // namespace state

}  // namespace astrix

#endif  // ASTRIX_STATE_H
