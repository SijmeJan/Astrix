/*! \file device.h
\brief Header file containing Device class definition
*/

#ifndef ASTRIX_DEVICE_H
#define ASTRIX_DEVICE_H

namespace astrix {

//! Simple class containing information about device
/*! This class is used to hold some very basic information about the machine the simulation is run on: whether we want to use any CUDA capable device, and how many CUDA-capable devices there are in total.
*/ 

class Device
{
 public:
  //! Constructor
  /*! Construct Device object. Count number of CUDA-capable devices and display capabilities on screen. By default, device 0 is used.
    \param _cudaFlag Flag whether to run on CUDA device. If set to zero, still count CUDA devices but do not use them to for computation.*/

  Device(int _cudaFlag);
  //! Destructor
  /*! Free Device object. If using CUDA, reset device for clean exit.*/
  ~Device();

  //! Return number of CUDA devices
  int GetDeviceCount();
  //! Return flag whether using CUDA
  int GetCudaFlag();

  cudaDeviceProp prop;
 private:
  //! Flag whether using CUDA device
  int cudaFlag;
  //! Number of CUDA-capable devices
  int deviceCount;
};

}

#endif
