/*! \file definitions.h
\brief Header file containing some basic definitions.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_DEFINITIONS_H
#define ASTRIX_DEFINITIONS_H

/*! \namespace astrix
\brief Namespace encapsulating all of Astrix
*/
namespace astrix {
  //! Use single or double precision
#if USE_DOUBLE == 1
  typedef double real;
  typedef double4 real4;
  typedef double3 real3;
  typedef double2 real2;
#else
  typedef float real;
  typedef float4 real4;
  typedef float3 real3;
  typedef float2 real2;
#endif

#if N_EQUATION == 1
  typedef real realNeq;
#endif
#if N_EQUATION == 3
  typedef real3 realNeq;
#endif
#if N_EQUATION == 4
  typedef real4 realNeq;
#endif

  //! Problem definitions
  /*! Enumeration of predefined test problems*/
  enum ProblemDefinition {PROBLEM_UNDEFINED, /*!< Undefined, leads to error*/
                          PROBLEM_LINEAR,    /*!< Linear wave*/
                          PROBLEM_SOD,       /*!< Sod shock tube*/
                          PROBLEM_BLAST,     /*!< Interacting blast waves*/
                          PROBLEM_KH,        /*!< Kelvin-Helmholz instability*/
                          PROBLEM_CYL,       /*!< Flow around cylinder*/
                          PROBLEM_RIEMANN,   /*!< 2D Riemann problem*/
                          PROBLEM_VORTEX,    /*!< Isentropic vortex*/
                          PROBLEM_NOH        /*!< Noh test problem*/
  };

  //! Integration schemes
  /*! Enumeration of available integration schemes */
  enum IntegrationScheme {SCHEME_UNDEFINED, /*!< Undefined, will lead to error*/
                          SCHEME_N,         /*!< N scheme, diffusive 1st order*/
                          SCHEME_LDA,       /*!< LDA scheme, 2nd order*/
                          SCHEME_B,         /*!< Blended N-LDA scheme*/
                          SCHEME_BX         /*!< Blended X scheme*/
  };
}  // namespace astrix

#endif
