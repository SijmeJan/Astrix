/*! \file device.h
\brief Header file containing Device class definition

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
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

  explicit Device(int _cudaFlag);
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

}  // namespace astrix

#endif  // ASTRIX_DEVICE_H
