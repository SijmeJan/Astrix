/*! \file main.cpp
\brief Main body of Astrix run

*/ /* \section LICENSE
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
//! Main function: parse command line arguments and run simulation
/*! First, parse command line arguments, then create and run an Astrix simulation.*/
//###########################################################################

int main(int argc, char *argv[])
{
  std::cout << "Welcome to Astrix!" << std::endl;

  // Parse command line arguments
  int verboseLevel = 0;                  // How much screen output
  int debugLevel = 0;                    // Level of debugging
  int nSwitches = 0;                     // Number of command line switches
  int cudaFlag = 0;                      // Flag whether to use CUDA device
  int restartNumber = 0;                 // Save number to restart from
  double maxWallClockHours = 1.0e10;     // Maximum wallclock hours to run
  astrix::ConservationLaw CL =
    astrix::CL_CART_EULER;

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
    // Select conservation law from command line
    if (strcmp(argv[i], "--conservationlaw") == 0 ||
        strcmp(argv[i], "-cl") == 0) {
      CL = astrix::CL_UNDEFINED;
      if (strcmp(argv[i+1], "advect") == 0) CL = astrix::CL_ADVECT;
      if (strcmp(argv[i+1], "burgers") == 0) CL = astrix::CL_BURGERS;
      if (strcmp(argv[i+1], "cart_iso") == 0) CL = astrix::CL_CART_ISO;
      if (strcmp(argv[i+1], "cyl_iso") == 0) CL = astrix::CL_CYL_ISO;
      if (strcmp(argv[i+1], "cart_euler") == 0) CL = astrix::CL_CART_EULER;

      std::cout << "Conservation law: ";
      switch (CL) {
      case astrix::CL_ADVECT :
        std::cout << "linear advection" << std::endl;
        break;
      case astrix::CL_BURGERS :
        std::cout << "Burgers equation" << std::endl;
        break;
      case astrix::CL_CART_ISO :
        std::cout << "Cartesian isothermal hydrodynamics" << std::endl;
        break;
      case astrix::CL_CYL_ISO :
        std::cout << "Cylindrical isothermal hydrodynamics" << std::endl;
        break;
      case astrix::CL_CART_EULER :
        std::cout << "Cartesian hydrodynamics" << std::endl;
        break;
      default :
        std::cout << "Invalid conservation law specified" << std::endl;
        std::cout << "Valid conservation laws: " << std::endl;
        std::cout << "  advect: linear advection" << std::endl;
        std::cout << "  burgers: Burgers equation" << std::endl;
        std::cout << "  cart_iso: Cartesian isothermal hydrodynamics"
                  << std::endl;
        std::cout << "  cyl_iso: Cylindrical isothermal hydrodynamics"
                  << std::endl;
        std::cout << "  cart_euler: Cartesian hydrodynamics" << std::endl;
        return 1;
      }

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
              << " [-cl conservationLaw]"
              << " filename"
              << std::endl;
    std::cout << "-d                  : run on GPU device" << std::endl;
    std::cout << "-v verboseLevel     : amount of output to stdout (0 - 2)"
              << std::endl;
    std::cout << "-D debugLevel       : amount of extra checks for debugging"
              << std::endl;
    std::cout << "-r restartNumber    : try to restart from previous dump"
              << std::endl;
    std::cout << "-cl conservationLaw : use different conservation law. Can be"
              << std::endl
              << "                      either \"advect\", \"burgers\" "
              << std::endl
              << "                      \"cart_iso\" or \"cart_euler\" "
              << std::endl;
    std::cout << "filename            : input file name" << std::endl;

    return 1;
  }

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

  // Linear advection
  if (CL == astrix::CL_ADVECT) {
    astrix::Simulation<astrix::real, astrix::CL_ADVECT> *simulation;
    try {
      simulation =
        new astrix::Simulation<astrix::real,
                               astrix::CL_ADVECT>(verboseLevel, debugLevel,
                                                  fileName, device,
                                                  restartNumber);
    }
    catch (...) {
      std::cout << "Could not create Simulation object, exiting..."
                << std::endl;
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
  }
  // Burgers equation
  if (CL == astrix::CL_BURGERS) {
    astrix::Simulation<astrix::real, astrix::CL_BURGERS> *simulation;
    try {
      simulation =
        new astrix::Simulation<astrix::real,
                               astrix::CL_BURGERS>(verboseLevel, debugLevel,
                                                   fileName, device,
                                                   restartNumber);
    }
    catch (...) {
      std::cout << "Could not create Simulation object, exiting..."
                << std::endl;
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
  }
  // Cartesian isothermal hydrodynamics
  if (CL == astrix::CL_CART_ISO) {
    astrix::Simulation<astrix::real3, astrix::CL_CART_ISO> *simulation;
    try {
      simulation =
        new astrix::Simulation<astrix::real3,
                               astrix::CL_CART_ISO>(verboseLevel, debugLevel,
                                                    fileName, device,
                                                    restartNumber);
    }
    catch (...) {
      std::cout << "Could not create Simulation object, exiting..."
                << std::endl;
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
  }
  // Cylindrical isothermal hydrodynamics
  if (CL == astrix::CL_CYL_ISO) {
    astrix::Simulation<astrix::real3, astrix::CL_CYL_ISO> *simulation;
    try {
      simulation =
        new astrix::Simulation<astrix::real3,
                               astrix::CL_CYL_ISO>(verboseLevel, debugLevel,
                                                   fileName, device,
                                                   restartNumber);
    }
    catch (...) {
      std::cout << "Could not create Simulation object, exiting..."
                << std::endl;
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
  }
  if (CL == astrix::CL_CART_EULER) {
    astrix::Simulation<astrix::real4, astrix::CL_CART_EULER> *simulation;
    try {
      simulation =
        new astrix::Simulation<astrix::real4,
                               astrix::CL_CART_EULER>(verboseLevel, debugLevel,
                                                      fileName, device,
                                                      restartNumber);
    }
    catch (...) {
      std::cout << "Could not create Simulation object, exiting..."
                << std::endl;
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
  }


  delete device;

  return 0;
}
