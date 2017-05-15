/*! \file checkvalidity.cpp
\brief Function for checking validity of data in SimulationParameter object

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "../../Common/definitions.h"
#include "./simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Perform basic checks of the content read from the input file is valid.
An error is thrown if any problems are detected.*/
//#########################################################################

void SimulationParameter::CheckValidity(ConservationLaw CL)
{
  if (problemDef == PROBLEM_UNDEFINED) {
    std::cout << "Invalid value for problemDefinition" << std::endl;
    throw std::runtime_error("");
  }
  if (problemDef == PROBLEM_SOD ||
      problemDef == PROBLEM_BLAST ||
      problemDef == PROBLEM_KH ||
      problemDef == PROBLEM_NOH ||
      problemDef == PROBLEM_CYL) {
    if (CL != CL_CART_EULER) {
      std::cout << "Problem requires Conservation law CL_CART_EULER"
                << std::endl;
      throw std::runtime_error("");
    }
  }

  if (maxSimulationTime < 0.0 ||
      std::isinf(maxSimulationTime) ||
      std::isnan(maxSimulationTime)) {
    std::cout << "Invalid value for maxSimulationTime" << std::endl;
    throw std::runtime_error("");
  }
  if (saveIntervalTime < 0.0 ||
      std::isinf(saveIntervalTime) ||
      std::isnan(saveIntervalTime)) {
    std::cout << "Invalid value for saveIntervalTime" << std::endl;
    throw std::runtime_error("");
  }
  if (saveIntervalTimeFine < 0.0 ||
      std::isinf(saveIntervalTimeFine) ||
      std::isnan(saveIntervalTimeFine)) {
    std::cout << "Invalid value for saveIntervalTimeFine" << std::endl;
    throw std::runtime_error("");
  }
  if (integrationOrder != 1 && integrationOrder != 2) {
    std::cout << "Invalid value for integrationOrder" << std::endl;
    throw std::runtime_error("");
  }
  if (massMatrix < 1 || massMatrix > 4) {
    std::cout << "Invalid value for massMatrix" << std::endl;
    throw std::runtime_error("");
  }
  if (selectiveLumpFlag != 0 && selectiveLumpFlag != 1) {
    std::cout << "Invalid value for selectiveLumpFlag" << std::endl;
    throw std::runtime_error("");
  }
  if (intScheme == SCHEME_UNDEFINED) {
    std::cout << "Invalid value for integrationScheme" << std::endl;
    throw std::runtime_error("");
  }
  if (CFLnumber <= 0.0 || CFLnumber > 1.0) {
    std::cout << "Invalid value for CFLnumber" << std::endl;
    throw std::runtime_error("");
  }
  if (preferMinMaxBlend != -1 &&
      preferMinMaxBlend != 0 &&
      preferMinMaxBlend != 1) {
    std::cout << "Invalid value for preferMinMaxBlend" << std::endl;
    throw std::runtime_error("");
  }
  if (specificHeatRatio < 0.0) {
    std::cout << "Invalid value for specificHeatRatio" << std::endl;
    throw std::runtime_error("");
  }
}

}  // namespace astrix
