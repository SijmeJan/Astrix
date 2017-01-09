/*! \file profile.cpp
\brief Output simple profiling information*/
#include <iostream>
#include <fstream>

#include "./profile.h"

namespace astrix {
  
//#########################################################################
// 
//#########################################################################

void WriteProfileFile(const char *fileName, int nElement,
		      float elapsedTime, int cudaFlag)
{
  int maxElement = 0;
  float maxTime = 0.0;
  int maxElementCPU = 0;
  float maxTimeCPU = 0.0;
  int GPUonlyFlag = 0;
  
  // Read current file if it exists
  std::ifstream infile;
  infile.open(fileName);
  if (infile.is_open()) {
    // Get current number of elements and elapsed time
    infile >> maxElement >> maxTime;
    // Save for later
    maxElementCPU = maxElement;
    maxTimeCPU = maxTime;
    // If using device, we need the next line if it exists
    if (cudaFlag == 1)
      if (!(infile >> maxElement >> maxTime))
	GPUonlyFlag = 1;
  }
  infile.close();
  
  if (maxElement < nElement) {
    std::ofstream outfile;
    outfile.open(fileName);

    if (cudaFlag == 1 && GPUonlyFlag == 0)
      outfile << maxElementCPU << " " << maxTimeCPU << std::endl;
    outfile << nElement << " " << elapsedTime << std::endl;
    outfile.close();
  }
}

}
