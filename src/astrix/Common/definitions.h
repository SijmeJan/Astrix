/*! \file definitions.h
\brief Header file containing some basic definitions.
*/
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

//#define TIME_ASTRIX

#define USE_DOUBLE -1

/*! \namespace astrix
\brief Namespace encapsulating all of Astrix
*/ 
namespace astrix {
  //! Use single or double precision
#if USE_DOUBLE==1
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
  
  //! Problem definitions
  /*! Enumeration of predefined test problems*/
  enum ProblemDefinition {PROBLEM_UNDEFINED, /*!< Undefined, will lead to error*/
			  PROBLEM_LINEAR, /*!< Linear wave*/ 
			  PROBLEM_VORTEX, /*!< Vortex advection*/
			  PROBLEM_SOD,    /*!< Sod shock tube*/
			  PROBLEM_BLAST,  /*!< Interacting blast waves*/
			  PROBLEM_KH,     /*!< Kelvin-Helmholz instability*/
			  PROBLEM_RT,     /*!< Rayleigh-Taylor instability*/ 
			  PROBLEM_RIEMANN,/*!< 2D Riemann problem*/
			  PROBLEM_YEE,    /*!< Yee vortex*/
			  PROBLEM_ADVECT  /*!< Advection problem*/
  };

  //! Integration schemes
  /*! Enumeration of available integration schemes */
  enum IntegrationScheme {SCHEME_UNDEFINED, /*!< Undefined, will lead to error*/
			  SCHEME_N,         /*!< N scheme, diffusive 1st order*/
			  SCHEME_LDA,       /*!< LDA scheme, 2nd order*/
			  SCHEME_B,          /*!< Blended N-LDA scheme*/
			  SCHEME_BX          /*!< Blended X scheme*/    
  };
}

#endif
