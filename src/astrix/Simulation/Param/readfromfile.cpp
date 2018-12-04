/*! \file Simulation/Param/readfromfile.cpp
\brief Function for reading data into SimulationParameter object

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include "../../Common/definitions.h"
#include "./simulationparameter.h"

namespace astrix {

//#########################################################################
/*! Read in data from input file. File is read line by line; input parameters can appear in any order but all must be present. An exception is thrown if any parameter is missing or has an invalid value.

\param *fileName Pointer to input file name
\param CL Conservation law used*/
//#########################################################################

void SimulationParameter::ReadFromFile(const char *fileName,
                                       ConservationLaw CL)
{
  // Open input file
  std::ifstream inFile(fileName);
  if (!inFile) {
    std::cout << "Error opening file " << fileName << std::endl;
    throw std::runtime_error("");
  }

  // Read file line by line
  std::string line;
  while (getline(inFile, line)) {
    std::string firstWord, secondWord;

    // Extract first two words from line
    std::istringstream iss(line);
    iss >> firstWord;
    iss >> secondWord;

    // Problem definition
    if (firstWord == "problemDefinition") {
      if (secondWord == "LIN") problemDef = PROBLEM_LINEAR;
      if (secondWord == "CYL") problemDef = PROBLEM_CYL;
      if (secondWord == "KH") problemDef = PROBLEM_KH;
      if (secondWord == "RIEMANN") problemDef = PROBLEM_RIEMANN;
      if (secondWord == "SOD") problemDef = PROBLEM_SOD;
      if (secondWord == "BLAST") problemDef = PROBLEM_BLAST;
      if (secondWord == "VORTEX") problemDef = PROBLEM_VORTEX;
      if (secondWord == "NOH") problemDef = PROBLEM_NOH;
      if (secondWord == "SOURCE") problemDef = PROBLEM_SOURCE;
      if (secondWord == "GAUSS") problemDef = PROBLEM_GAUSS;
      if (secondWord == "DISC") problemDef = PROBLEM_DISC;
    }

    // Time to stop simulation; check that secondWord is number
    if (firstWord == "maxSimulationTime") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
        maxSimulationTime = atof(secondWord.c_str());
    }

    // Save interval; check that secondWord is number
    if (firstWord == "saveIntervalTime") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
        saveIntervalTime = atof(secondWord.c_str());
    }
    // Fine grain save interval; check that secondWord is number
    if (firstWord == "saveIntervalTimeFine") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
        saveIntervalTimeFine = atof(secondWord.c_str());
    }
    // Flag whether do output VTK files
    if (firstWord == "writeVTK") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("01") == std::string::npos)
        writeVTK = atof(secondWord.c_str());
    }

    // Integration scheme
    if (firstWord == "integrationScheme") {
      if (secondWord == "N") intScheme = SCHEME_N;
      if (secondWord == "LDA") intScheme = SCHEME_LDA;
      if (secondWord == "B") intScheme = SCHEME_B;
      if (secondWord == "BX") intScheme = SCHEME_BX;
    }

    // Integration order (should be 1 or 2)
    if (firstWord == "integrationOrder") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("12") == std::string::npos)
        integrationOrder = atof(secondWord.c_str());
    }

    // Mass matrix formulation (should be 1, 2, 3 or 4)
    if (firstWord == "massMatrix") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("1234") == std::string::npos)
        massMatrix = atof(secondWord.c_str());
    }

    // Flag to use selective lumping
    if (firstWord == "selectiveLumpFlag") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("01") == std::string::npos)
        selectiveLumpFlag = atof(secondWord.c_str());
    }

    // Courant number
    if (firstWord == "CFLnumber") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
        CFLnumber = atof(secondWord.c_str());
    }

    // Preference for minimum/maximum blend parameter (-1, 0 or 1)
    if (firstWord == "preferMinMaxBlend") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("01-") == std::string::npos)
        preferMinMaxBlend = atof(secondWord.c_str());
    }

    // SpecificHeatRatio
    if (firstWord == "specificHeatRatio") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
        specificHeatRatio = atof(secondWord.c_str());
    }
    if (firstWord == "minError") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789.-e") == std::string::npos)
        minError = atof(secondWord.c_str());
    }
    if (firstWord == "maxError") {
      if (!secondWord.empty() &&
          secondWord.find_first_not_of("0123456789.-e") == std::string::npos)
        maxError = atof(secondWord.c_str());
    }

  }

  inFile.close();

  // Check validity of parameters
  try {
    CheckValidity(CL);
  }
  catch(...) {
    std::cout << "Some Simulation parameters not valid, exiting..."
              << std::endl;
    throw;
  }

}

}  // namespace astrix
