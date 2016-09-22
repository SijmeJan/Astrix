// -*-c++-*-
/*! \file simulation.cpp
\brief Constructor, destructor and initialization of the Simulation class*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cuda_runtime_api.h>

#include "../Common/definitions.h"
#include "../Array/array.h"
#include "../Mesh/mesh.h"
#include "./simulation.h"
#include "../Device/device.h"

namespace astrix {
  
extern int nPredicatesFast;
extern int nPredicatesSlow;

//#########################################################################
/*! Define Arrays, create Mesh and setup simulation.
  \param _verboseLevel How much information to output to stdout in Astrix. 
  \param _debugLevel Level of extra checks for correct mesh.
  \param *fileName Input file name 
  \param *device Device to be used for computation.
  \param restartNumber Number of saved file to restore from*/
//#########################################################################

Simulation::Simulation(int _verboseLevel,
		       int _debugLevel,
		       char *fileName,
		       Device *_device,
		       int restartNumber)
{
  // How much to output to screen
  verboseLevel = _verboseLevel;
  debugLevel = _debugLevel;
  device = _device;
  
  cudaFlag = device->GetCudaFlag();

  // Only run in 2D for now
  nSpaceDim = 2;
  // Velocity components + density + energy equation
  nEquation = nSpaceDim + 2;

  nTimeStep = 0;
  
  // Define arrays
  vertexState           = new Array<real4>(1, cudaFlag);
  vertexStateOld        = new Array<real4>(1, cudaFlag);
  vertexPotential       = new Array<real>(1, cudaFlag);
  vertexStateDiff       = new Array<real4>(1, cudaFlag);
  vertexParameterVector = new Array<real4>(1, cudaFlag);
  
  triangleResidueN  = new Array<real4>(nSpaceDim + 1, cudaFlag);
  triangleResidueLDA = new Array<real4>(nSpaceDim + 1, cudaFlag);
  triangleResidueTotal = new Array<real4>(1, cudaFlag);
  triangleBlendFactor = new Array<real4>(1, cudaFlag);
  triangleShockSensor = new Array<real>(1, cudaFlag);
    
  try {
    // Create mesh object
    mesh = new Mesh(verboseLevel, debugLevel, cudaFlag,
		    fileName, device, restartNumber);
  }
  catch (...) {
    std::cout << "Mesh creation failed" << std::endl;

    // Clean up; destructor will not be called
    delete vertexState;
    delete vertexStateOld;
    delete vertexPotential;
    delete vertexParameterVector;
    delete vertexStateDiff;
    
    delete triangleResidueN;
    delete triangleResidueLDA;
    delete triangleResidueTotal;
    delete triangleBlendFactor;
    delete triangleShockSensor;
   
    throw;
  }

  try {
    // Initialize simulation
    Init(fileName, restartNumber);
  }
  catch (...) {
    std::cout << "Simulation initialization failed" << std::endl;

    // Clean up; destructor will not be called
    delete vertexState;
    delete vertexStateOld;
    delete vertexPotential;
    delete vertexParameterVector;
    delete vertexStateDiff;
    
    delete triangleResidueN;
    delete triangleResidueLDA;
    delete triangleResidueTotal;
    delete triangleBlendFactor;
    delete triangleShockSensor;
   
    delete mesh;
    
    throw;
  }

  std::cout << "Predicates: " << nPredicatesFast << " "
	    << nPredicatesSlow << std::endl;
}

// #########################################################################
// Destructor for simulation object
// #########################################################################

Simulation::~Simulation()
{
  delete vertexState;
  delete vertexStateOld;
  delete vertexPotential;
  delete vertexParameterVector;
  delete vertexStateDiff;
  
  delete triangleResidueN;
  delete triangleResidueLDA;
  delete triangleResidueTotal;
  delete triangleBlendFactor;
  delete triangleShockSensor;

  delete mesh;
}

// #########################################################################
/*! Read the input file and set up the simulation accordingly. It will create 
  a Mesh object, allocate memory on CPU and GPU, and set the initial conditions. 
  Returns 0 if successful, 1 if an error occurred along the way. 
  \param fileName Name of input file
  \param restartNumber Number of save file to restore from*/
// #########################################################################

void Simulation::Init(const char *fileName, int restartNumber)
{
  //-----------------
  // Read input file
  //-----------------

  std::cout << "Setting up Astrix simulation using parameter file \'"
       << fileName << "\'" << std::endl;

  // Parameters to be read from input file, set to invalid values
  problemDef = PROBLEM_UNDEFINED;
  maxSimulationTime = -1.0;
  saveIntervalTime = -1.0;
  saveIntervalTimeFine = -1.0;
  integrationOrder = -1;
  massMatrix = -1;
  selectiveLumpFlag = -1;
  intScheme = SCHEME_UNDEFINED;
  specificHeatRatio = -1.0;

  // Open parameter file
  std::ifstream inFile(fileName);
  if (!inFile.is_open()) {
    std::cout << "Error opening file " << fileName << std::endl;
    throw std::runtime_error("");
  }
  
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
      if (secondWord == "RT") problemDef = PROBLEM_RT;
      if (secondWord == "KH") problemDef = PROBLEM_KH;
      if (secondWord == "RIEMANN") problemDef = PROBLEM_RIEMANN;
      if (secondWord == "SOD") problemDef = PROBLEM_SOD;
      if (secondWord == "BLAST") problemDef = PROBLEM_BLAST;
      if (secondWord == "VORTEX") problemDef = PROBLEM_VORTEX;
      if (secondWord == "YEE") problemDef = PROBLEM_YEE;
      if (secondWord == "ADVECT") problemDef = PROBLEM_ADVECT;
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

    // SpecificHeatRatio
    if (firstWord == "specificHeatRatio") {
      if (!secondWord.empty() &&
	  secondWord.find_first_not_of("0123456789-.e") == std::string::npos)
	specificHeatRatio = atof(secondWord.c_str());
    }

  }

  // Check validity of parameters
  if (problemDef == PROBLEM_UNDEFINED) {
    std::cout << "Invalid value for problemDefinition" << std::endl;
    throw std::runtime_error("");
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
  if (specificHeatRatio < 0.0) {
    std::cout << "Invalid value for specificHeatRatio" << std::endl;
    throw std::runtime_error("");
  }
    
  // Close simulation parameter file
  inFile.close();

  //-----------------------------------------------------------------
  // Done reading input file
  //-----------------------------------------------------------------
  
  int nVertex = mesh->GetNVertex();
  int nTriangle = mesh->GetNTriangle();

  // Allocate memory
  vertexState->SetSize(nVertex);
  vertexStateOld->SetSize(nVertex);
  vertexPotential->SetSize(nVertex);
  vertexStateDiff->SetSize(nVertex);
  vertexParameterVector->SetSize(nVertex);
  
  triangleResidueN->SetSize(nTriangle);
  triangleResidueLDA->SetSize(nTriangle);
  triangleResidueTotal->SetSize(nTriangle);
  if (intScheme == SCHEME_B)
    triangleBlendFactor->SetSize(nTriangle);
  if (intScheme == SCHEME_BX)
    triangleShockSensor->SetSize(nTriangle);

  CalcPotential();

  if (restartNumber == 0) {
    // Start at t = 0.0
    simulationTime = 0.0;
    
    // Set initial conditions
    SetInitial();
    
    if (mesh->IsAdaptive() == 1) {
      ReplaceEnergyWithPressure();
      Coarsen(-1);
      Refine();
      Coarsen(-1);
      Refine();
      ReplacePressureWithEnergy();
    }
  } else {
    try {
      Restore(restartNumber);
    }
    catch (...) {
      std::cout << "Restoring failed!" << std::endl;
      throw;
    }
  }
  
  if (verboseLevel > 0) {
    std::cout << "Done creating simulation." << std::endl;
    std::cout << "Memory allocated on host: "
	      << ((real)(Array<real>::memAllocatedHost) +
		  (real)(Array<int>::memAllocatedHost) +
		  (real)(Array<unsigned int>::memAllocatedHost))/
      (real) (1073741824) << " Gb, on device: "
	      << ((real)(Array<real>::memAllocatedDevice) +
		  (real)(Array<int>::memAllocatedDevice) +
		  (real)(Array<unsigned int>::memAllocatedDevice))/
      (real) (1073741824) << " Gb" << std::endl;
  }
}

}
