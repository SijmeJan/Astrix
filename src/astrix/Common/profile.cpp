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
  int maxElementCPU = 0;
  float maxTimeCPU = 0.0;
  int maxElementGPU = 0;
  float maxTimeGPU = 0.0;
  
  // Read current file if it exists
  std::ifstream infile;
  infile.open(fileName);
  if (infile.is_open()) {
    // Get current number of elements and elapsed time
    infile >> maxElementCPU >> maxTimeCPU;
    infile >> maxElementGPU >> maxTimeGPU;
  }
  infile.close();

  if (cudaFlag == 1) {
    if (maxElementGPU < nElement) {
      maxElementGPU = nElement;
      maxTimeGPU = elapsedTime;
    }
  } else {
    if (maxElementCPU < nElement) {
      maxElementCPU = nElement;
      maxTimeCPU = elapsedTime;
    }
  }
  
  std::ofstream outfile;
  outfile.open(fileName);

  outfile << maxElementCPU << " " << maxTimeCPU << std::endl; 
  outfile << maxElementGPU << " " << maxTimeGPU << std::endl;
  outfile.close();
}

}
