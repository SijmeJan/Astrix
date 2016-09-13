/*! \file device.cpp
\brief Functions for Device class
*/
#include <iostream>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include "device.h"

namespace astrix {
  
//###########################################################################
// Initialise CUDA
//###########################################################################

Device::Device(int _cudaFlag)
{
  cudaFlag = _cudaFlag;
  
  // Check for CUDA capable devices
  deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess && cudaFlag == 1) {
    std::cout << "  cudaGetDeviceCount failed: " << error_id << std::endl
	      << "-> " << cudaGetErrorString(error_id) << std::endl;
    throw std::runtime_error("");
  }
  std::cout << "DeviceCount: " << deviceCount << std::endl;
  
  if (cudaFlag == 1) {
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
      std::cout << "  There is no device supporting CUDA."
		<< std::endl;
      throw std::runtime_error("");
    } else {
      std::cout << "  Found " << deviceCount
		<< " CUDA Capable device(s)" << std::endl;
    }

    // Output some device properties
    for (int dev = 0; dev < deviceCount; ++dev) {      
      cudaGetDeviceProperties(&prop, dev);
      
      std::cout << "  Device " << dev << ": " << prop.name << std::endl;
      std::cout << "    Compute capability: "
		<< prop.major << "."
		<< prop.minor << std::endl;
      std::cout << "    Global device memory: "
		<< static_cast<float>(prop.totalGlobalMem)/1048576.0f
		<< " MB (" << prop.totalGlobalMem << " bytes)"
		<< std::endl;
      std::cout << "    Shared memory per block: "
		<< prop.sharedMemPerBlock
		<< " bytes" << std::endl;
      std::cout << "    Max number of threads per block: "
		<< prop.maxThreadsPerBlock << std::endl;
      std::cout << "    Registers per block: "
		<< prop.regsPerBlock << std::endl;
      if (prop.deviceOverlap)
	std::cout << "    Kernel overlap (use of streams): yes" << std::endl;
      else
	std::cout << "    Kernel overlap (use of streams): no" << std::endl;
      if (prop.canMapHostMemory)
	std::cout << "    Can map host memory: yes" << std::endl;
      else
	std::cout << "    Can map host memory: no" << std::endl;
      if (prop.integrated)
	std::cout << "    Integrated: yes" << std::endl;
      else
	std::cout << "    Integrated: no" << std::endl;
      std::cout << "    Memory clock rate: " << prop.memoryClockRate
		<< " KHz" << std::endl;
      std::cout << "    Memory bus width: " << prop.memoryBusWidth
		<< " bits" << std::endl;
      float bandWidth = (float)(prop.memoryClockRate)*1000.0*
	(float)(prop.memoryBusWidth/8)*2.0/(1024.0*1024.0*1024.0);
      std::cout << "    Theoretical bandwidth: "
		<< bandWidth << " GB/s" << std::endl;
    }
  } else {
    std::cout << "Not using CUDA device" << std::endl;
  }
}

//###########################################################################
// Safely exit CUDA
//###########################################################################

Device::~Device()
{
  if (cudaFlag == 1) cudaDeviceReset();
}

//###########################################################################
// Return number of CUDA devices
//###########################################################################

int Device::GetDeviceCount()
{
  return deviceCount;
}


//###########################################################################
// Return flag if CUDA is being used
//###########################################################################

int Device::GetCudaFlag()
{
  return cudaFlag;
}
  
}
