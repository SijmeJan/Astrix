/*! \file simulationparameter.h
\brief Header file for SimulationParameter class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_SIMULATION_PARAM_H
#define ASTRIX_SIMULATION_PARAM_H

namespace astrix {

//! Class containing parameters for the Simulation
/*! Various parameters governing the Simulation  are
read from an input file and stored in this class*/
class SimulationParameter
{
 public:
  //! Constructor, set all data to invalid values
  SimulationParameter();
  //! Destructor
  ~SimulationParameter();

  //! Problem specification (see Common/definitions.h)
  /*! Converted to int using definitions in definitions.h.*/
  ProblemDefinition problemDef;

  //! Number of space dimensions (fixed to 2)
  int nSpaceDim;

  //! Integration scheme (see Common/definitions.h)
  /*! Converted to int using definitions in definitions.h.*/
  IntegrationScheme intScheme;
  //! Order of accuracy in time (1 or 2)
  int integrationOrder;
  //! Mass matrix formulation to use (1, 2, 3 or 4)
  int massMatrix;
  //! Flag whether to use selective lumping
  int selectiveLumpFlag;
  //! Courant number
  real CFLnumber;
  //! Preference for using minimum/maximum value of blend parameter
  int preferMinMaxBlend;

  //! Ratio of specific heats
  real specificHeatRatio;

  //! Maximum simulation time
  real maxSimulationTime;
  //! Time between 2D saves
  real saveIntervalTime;
  //! Time between 0D saves
  real saveIntervalTimeFine;
  //! Flag whether do output VTK files
  int writeVTK;

  //! Read in data from file
  void ReadFromFile(const char *fileName, ConservationLaw CL);

 private:
  //! Check if contents are valid
  void CheckValidity(ConservationLaw CL);
};

}  // namespace astrix

#endif  // ASTRIX_SIMULATION_PARAM_H
