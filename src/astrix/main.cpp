/*! \file main.cpp
\brief Main body of Astrix run

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/

#include <cuda_runtime_api.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cstddef>         // std::size_t
#include <string>         // std::string

#include "./Common/definitions.h"
#include "./Simulation/simulation.h"
#include "./Device/device.h"

//###########################################################################
// main
//###########################################################################

int main(int argc, char *argv[])
{
  // Parse command line arguments
  int verboseLevel = 0;                  // How much screen output
  int debugLevel = 0;                    // Level of debugging
  int nSwitches = 0;                     // Number of command line switches
  int cudaFlag = 0;                      // Flag whether to use CUDA device
  int restartNumber = 0;                 // Save number to restart from
  double maxWallClockHours = 1.0e10;     // Maximum wallclock hours to run

  // Walk through all command line arguments
  for (int i = 1; i < argc; ++i) {
    // Set flag for executing on device
    if (strcmp(argv[i], "--device") == 0 ||
        strcmp(argv[i], "-d") == 0) {
      std::cout << "Will use CUDA" << std::endl;
      cudaFlag = 1;
      nSwitches++;
    }
    // Set verbose level
    if (strcmp(argv[i], "--verbose") == 0 ||
        strcmp(argv[i], "-v") == 0) {
      verboseLevel = atoi(argv[i+1]);
      std::cout << "Verbose level: " << verboseLevel << std::endl;
      nSwitches += 2;
    }
    // Set debug level
    if (strcmp(argv[i], "--Debug") == 0 ||
        strcmp(argv[i], "-D") == 0) {
      debugLevel = atoi(argv[i+1]);
      std::cout << "Debug level: " << debugLevel << std::endl;
      nSwitches += 2;
    }
    // Allow restart
    if (strcmp(argv[i], "--restart") == 0 ||
        strcmp(argv[i], "-r") == 0) {
      restartNumber = atoi(argv[i+1]);
      std::cout << "Restart number: " << restartNumber << std::endl;
      nSwitches += 2;
    }
    // Max wall clock hours
    if (strcmp(argv[i], "--wallclocklimit") == 0 ||
        strcmp(argv[i], "-wcl") == 0) {
      maxWallClockHours = atof(argv[i+1]);
      std::cout << "Maximum wall clock time: " << maxWallClockHours
                << " hours" << std::endl;
      nSwitches += 2;
    }
  }

  // Check for correct number of arguments
  if (argc != 2 + nSwitches) {
    // Strip possible directory from command
    std::string cmand = argv[0];
    std::size_t found = cmand.find_last_of("/");

    // Print usage
    std::cout << "Usage: " << cmand.substr(found + 1)
              << " [-d]"
              << " [-v verboseLevel]"
              << " [-D debugLevel]"
              << " [-r restartNumber]"
              << " filename"
              << std::endl;
    std::cout << "-d               : run on GPU device" << std::endl;
    std::cout << "-v verboseLevel  : amount of output to stdout (0 - 2)"
              << std::endl;
    std::cout << "-D debugLevel    : amount of extra checks for debugging"
              << std::endl;
    std::cout << "-r restartNumber : try to restart from previous dump"
              << std::endl;
    std::cout << "filename         : input file name" << std::endl;

    return 1;
  }

  std::cout << "Welcome to Astrix!" << std::endl;


  // Initialise CUDA device
  astrix::Device *device;
  try {
    device = new astrix::Device(cudaFlag);
  }
  catch (...) {
    std::cout << "Device initialisation failed; exiting..." << std::endl;
    return 0;
  }

  // Last argument should be input file name
  char *fileName = argv[argc-1];

  // Create simulation from input file

  astrix::Simulation<astrix::real4, astrix::CL_CART_EULER> *simulation;
  try {
    simulation =
      new astrix::Simulation<astrix::real4,
                             astrix::CL_CART_EULER>(verboseLevel, debugLevel,
                                                    fileName, device,
                                                    restartNumber);
  }
  catch (...) {
    std::cout << "Could not create Simulation object, exiting..." << std::endl;
    delete device;
    return 1;
  }

  try {
    // Run simulation
    simulation->Run(maxWallClockHours);
  }
  catch (...) {
    std::cout << "Exiting with error!" << std::endl;
    return 1;
  }

  // Clean up
  delete simulation;
  delete device;

  return 0;
}
