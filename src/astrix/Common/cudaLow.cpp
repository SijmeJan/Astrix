/*! \file cudaLow.cpp
\brief CUDA error handling*/
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <cuda_runtime_api.h>

#include "./cudaLow.h"

namespace astrix {
  
//#########################################################################
// Cuda error handling
//#########################################################################

void gpuAssert(cudaError_t code, char *file, int line)
{
   if (code != cudaSuccess) {
     // Print error
     fprintf(stderr,"GPUassert: %s %s %d\n",
	     cudaGetErrorString(code), file, line);
     // Exit
     exit(code);
   }
}

}
