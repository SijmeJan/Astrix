/*! \file checkvalidity.cpp
\brief Function for checking validity of data in MeshParameter object

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
#include "./meshparameter.h"

namespace astrix {

//#########################################################################
/*! Perform basic checks of the content read from the input file is valid.
An error is thrown if any problems are detected.*/
//#########################################################################

void MeshParameter::CheckValidity()
{
  if (problemDef == PROBLEM_UNDEFINED) {
    std::cout << "Invalid value for problemDefinition" << std::endl;
    throw std::runtime_error("");
  }
  if (equivalentPointsX <= 1) {
    std::cout << "Invalid value for equivalentPointsX" << std::endl;
    throw std::runtime_error("");
  }
  if (std::isinf(minx) || std::isnan(minx)) {
    std::cout << "Invalid value for minx" << std::endl;
    throw std::runtime_error("");
  }
  if (std::isinf(maxx) || std::isnan(maxx)) {
    std::cout << "Invalid value for maxx" << std::endl;
    throw std::runtime_error("");
  }
  if (std::isinf(miny) || std::isnan(miny)) {
    std::cout << "Invalid value for miny" << std::endl;
    throw std::runtime_error("");
  }
  if (std::isinf(maxy) || std::isnan(maxy)) {
    std::cout << "Invalid value for maxy" << std::endl;
    throw std::runtime_error("");
  }
  if (minx > maxx) {
    std::cout << "Invalid minx/maxx combination" << std::endl;
    throw std::runtime_error("");
  }
  if (miny > maxy) {
    std::cout << "Invalid miny/maxy combination" << std::endl;
    throw std::runtime_error("");
  }
  if (periodicFlagX != 0 && periodicFlagX != 1) {
    std::cout << "Invalid value for periodicFlagX" << std::endl;
    throw std::runtime_error("");
  }
  if (periodicFlagY != 0 && periodicFlagY != 1) {
    std::cout << "Invalid value for periodicFlagY" << std::endl;
    throw std::runtime_error("");
  }
  if (adaptiveMeshFlag != 0 && adaptiveMeshFlag != 1) {
    std::cout << "Invalid value for adaptiveMeshFlag" << std::endl;
    throw std::runtime_error("");
  }
  if (maxRefineFactor <= 0 || maxRefineFactor > 1000000) {
    std::cout << "Invalid value for maxRefineFactor" << std::endl;
    throw std::runtime_error("");
  }
  if (nStepSkipRefine <= 0) {
    std::cout << "Invalid value for nStepSkipRefine" << std::endl;
    throw std::runtime_error("");
  }
  if (nStepSkipCoarsen <= 0) {
    std::cout << "Invalid value for nStepSkipCoarsen" << std::endl;
    throw std::runtime_error("");
  }
  if (qualityBound < 1.0 ||
      std::isinf(qualityBound) ||
      std::isnan(qualityBound)) {
    std::cout << "Invalid value for qualityBound" << std::endl;
    throw std::runtime_error("");
  }
  if (structuredFlag != 0 &&
      structuredFlag != 1 &&
      structuredFlag != 2) {
    std::cout << "Invalid value for structuredFlag" << std::endl;
    throw std::runtime_error("");
  }
}

}  // namespace astrix
