#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>

#include "./Common/definitions.h"
#include "./Simulation/simulation.h"
#include "./Device/device.h"
#include "./Common/timer.h"

//###########################################################################
// main
//###########################################################################

int main(int argc, char *argv[])
{
  std::cout << "Welcome to Astrix!" << std::endl;  
  
  // Parse command line arguments
  int checkDeviceFlag = 0;               // Exit after check for devices
  int verboseLevel = 0;                  // How much screen output
  int debugLevel = 0;                    // Level of debugging
  int nSwitches = 0;                     // Number of command line switches
  int cudaFlag = 0;                      // Flag whether to use CUDA device
  int restartNumber = 0;                 // Save number to restart from
  double maxWallClockHours = 1.0e10;     // Maximum wallclock hours to run
  int extraFlag = 0;

  //astrix::Timer *timer = new astrix::Timer("main.prof", 100000, cudaFlag);
  //delete timer;
  //return 0;
  
  // Walk through all command line arguments
  for (int i = 1; i < argc; ++i) {
    // Set flag for just checking for device
    if (strcmp(argv[i], "--checkdevice") == 0 ||
        strcmp(argv[i], "-c") == 0) {
      std::cout << "Will quit after checking for CUDA devices" << std::endl;
      checkDeviceFlag = 1;
      nSwitches++;
    }
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
    // Extra flag
    if (strcmp(argv[i], "--extraflag") == 0 ||
        strcmp(argv[i], "-e") == 0) {
      extraFlag = atoi(argv[i+1]);
      std::cout << "Extra flag: " << extraFlag << std::endl;
      nSwitches += 2;
    }
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
  
  // Exit if just checking device
  if (checkDeviceFlag == 1) {
    std::cout << "Devices checked, exiting..." << std::endl;
    delete device;
    return 0;
  }
  
  // Check for correct number of arguments
  if (argc != 2 + nSwitches) {
    std::cout << "Usage: " << argv[0] 
	      << " [-c]"
	      << " [-d]"
	      << " [-v verboseLevel]"
	      << " [-D debugLevel]" 
	      << " [-r restartNumber]"
	      << " [-e extraFlag]"
	      << " filename" 
	      << std::endl;
    delete device;
    return 1;
  }

  // Last argument should be input file name
  char *fileName = argv[argc-1];

  // Create simulation from input file
  astrix::Simulation *simulation;
  try {
    simulation =
      new astrix::Simulation(verboseLevel, debugLevel,
			     fileName, device, restartNumber,
			     extraFlag);
  }
  catch (...) {
    std::cout << "Could not create Simulation object, exiting..." << std::endl;
    delete device;
    return 1;
  }

  try {
    // Run simulation
    simulation->Run(restartNumber, maxWallClockHours);
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
