/*! \file meshparameter.h
\brief Header file for MeshParameter class

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_MESH_PARAM_H
#define ASTRIX_MESH_PARAM_H

namespace astrix {

//! Class containing parameters for the Mesh
/*! Various parameters governing the resolution and quality of the Mesh are
read from an input file and stored in this class*/
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

  //! Triangle size for initial Mesh (derived from \a equivalentPointsX)
  real baseResolution;
  //! Triangle size for adaptive mesh (derived from \a baseResolution and \a maxRefineFactor)
  real maxResolution;
  //! Filename for input boundary vertices
  std::string vertexBoundaryInputFile;
  //! Filename for input internal vertices
  std::string vertexInputFile;

  //! Read in data from file
  void ReadFromFile(const char *fileName);

 private:
  //! Check if contents are valid
  void CheckValidity();
};

}  // namespace astrix

#endif  // ASTRIX_MESH_PARAM_H
