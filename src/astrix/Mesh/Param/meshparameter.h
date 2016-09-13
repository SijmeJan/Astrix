/*! \file meshparameter.h
\brief Header file for MeshParameter class*/
#ifndef ASTRIX_MESH_PARAM_H
#define ASTRIX_MESH_PARAM_H

namespace astrix {

//! Class containing parameters for the Mesh
/*! Various parameters governing the resolution and quality of the Mesh are read from an input file and stored in this class*/  
class MeshParameter
{
 public:
  //! Constructor, set all data to invalid values
  MeshParameter();
  //! Destructor
  ~MeshParameter();  

  //! Definition of test problem
  ProblemDefinition problemDef;
  //! Approximate number of vertices in x direction
  int equivalentPointsX;
  //! Quality bound on triangles; must be >= 1
  real qualityBound;
  //! Flag to create periodic domain in x
  int periodicFlagX;
  //! Flag to create periodic domain in y
  int periodicFlagY;
  //! Position of left x boundary
  real minx;
  //! Position of right x boundary
  real maxx;
  //! Position of left y boundary
  real miny;
  //! Position of right y boundary
  real maxy;
  //! Flag whether using structured mesh
  int structuredFlag;
  //! Flag whether mesh is adaptive
  int adaptiveMeshFlag;
  //! Maximum factor to increase resolution over base resolution if adaptive mesh is used
  int maxRefineFactor;
  //! Number of time steps without checking if refinement is needed 
  int nStepSkipRefine;
  //! Number of time steps without checking if coarsening is needed
  int nStepSkipCoarsen;
  //! If discretization error smaller than minError, coarsen Mesh
  real minError;
  //! If discretization error larger than maxError, refine Mesh
  real maxError;
  
  //! Triangle size for initial Mesh (derived from \a equivalentPointsX)
  real baseResolution;
  //! Triangle size for adaptive mesh (derived from \a baseResolution and \a maxRefineFactor)
  real maxResolution;

  //! Read in data from file
  void ReadFromFile(const char *fileName);
  
 private:
  
  //! Check if contents are valid
  void CheckValidity();
};

}

#endif

