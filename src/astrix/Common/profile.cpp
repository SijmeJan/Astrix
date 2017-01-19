/*! \file profile.cpp
\brief Output simple profiling information

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
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
  float totalTimeCPU = 0.0;
  int maxElementGPU = 0;
  float maxTimeGPU = 0.0;
  float totalTimeGPU = 0.0;

  // Read current file if it exists
  std::ifstream infile;
  infile.open(fileName);
  if (infile.is_open()) {
    // Get current number of elements and elapsed time
    infile >> maxElementCPU >> maxTimeCPU >> totalTimeCPU;
    infile >> maxElementGPU >> maxTimeGPU >> totalTimeGPU;
  }
  infile.close();

  if (cudaFlag == 1) {
    totalTimeGPU += elapsedTime;
    if (maxElementGPU < nElement) {
      maxElementGPU = nElement;
      maxTimeGPU = elapsedTime;
    }
  } else {
    totalTimeCPU += elapsedTime;
    if (maxElementCPU < nElement) {
      maxElementCPU = nElement;
      maxTimeCPU = elapsedTime;
    }
  }

  std::ofstream outfile;
  outfile.open(fileName);

  outfile << maxElementCPU << " "
          << maxTimeCPU << " "
          << totalTimeCPU << std::endl;
  outfile << maxElementGPU << " "
          << maxTimeGPU << " "
          << totalTimeGPU << std::endl;
  outfile.close();
}

}
