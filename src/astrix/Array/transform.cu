// -*-c++-*-
/*! \file transform.cu
\brief Functions for transforming Array to host or device
*/
#include <iostream>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//###################################################
// Copy data to device
//###################################################

template <class T>
void Array<T>::CopyToDevice()
{  
  // Make sure we have enough space...
  if (deviceVec != 0) gpuErrchk(cudaFree(deviceVec));
  gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceVec), 
		       nDims*realSize*sizeof(T)));

  gpuErrchk(cudaMemcpy(deviceVec, hostVec,
		       nDims*realSize*sizeof(T),
		       cudaMemcpyHostToDevice));
}

//###################################################
// Copy data from device
//###################################################

template <class T>
void Array<T>::CopyToHost()
{
  free(hostVec);
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));
  gpuErrchk(cudaMemcpy(hostVec, deviceVec,
		       nDims*realSize*sizeof(T),
		       cudaMemcpyDeviceToHost));
}

//###################################################
// Transform into device vector
//###################################################

template <class T>
void Array<T>::TransformToDevice()
{
  // Return if already living on device
  if (cudaFlag == 1) return;

  // Free any allocated device memory
  if (deviceVec != 0) gpuErrchk(cudaFree(deviceVec));

  // Allocate fresh device memory
  gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceVec), 
		       nDims*realSize*sizeof(T)));

  // Copy to device
  gpuErrchk(cudaMemcpy(deviceVec, hostVec,
		       nDims*realSize*sizeof(T),
		       cudaMemcpyHostToDevice));

  // Now living on device
  cudaFlag = 1;
}

//###################################################
// Transform into host vector
//###################################################

template <class T>
void Array<T>::TransformToHost()
{
  // Return if already living on host
  if (cudaFlag == 0) return;

  // Free memory allocated on host
  free(hostVec);

  // Allocate fresh host memory
  hostVec = (T *)malloc(nDims*realSize*sizeof(T));

  // Copy to host
  gpuErrchk(cudaMemcpy(hostVec, deviceVec,
		       nDims*realSize*sizeof(T),
		       cudaMemcpyDeviceToHost));

  // Free device memory
  gpuErrchk(cudaFree(deviceVec));
  deviceVec = 0;

  // Now living on host
  cudaFlag = 0;
}

//###################################################
// Instantiate
//###################################################

template void Array<double>::CopyToDevice();
template void Array<double>::CopyToHost();
template void Array<double>::TransformToHost();
template void Array<double>::TransformToDevice();

//###################################################

template void Array<double4>::CopyToDevice();
template void Array<double4>::CopyToHost();
template void Array<double4>::TransformToHost();
template void Array<double4>::TransformToDevice();

//###################################################

template void Array<double3>::CopyToDevice();
template void Array<double3>::CopyToHost();
template void Array<double3>::TransformToHost();
template void Array<double3>::TransformToDevice();

//###################################################

template void Array<double2>::CopyToDevice();
template void Array<double2>::CopyToHost();
template void Array<double2>::TransformToHost();
template void Array<double2>::TransformToDevice();

  //###################################################

template void Array<float>::CopyToDevice();
template void Array<float>::CopyToHost();
template void Array<float>::TransformToHost();
template void Array<float>::TransformToDevice();

//###################################################

template void Array<float4>::CopyToDevice();
template void Array<float4>::CopyToHost();
template void Array<float4>::TransformToHost();
template void Array<float4>::TransformToDevice();

//###################################################

template void Array<float3>::CopyToDevice();
template void Array<float3>::CopyToHost();
template void Array<float3>::TransformToHost();
template void Array<float3>::TransformToDevice();

//###################################################

template void Array<float2>::CopyToDevice();
template void Array<float2>::CopyToHost();
template void Array<float2>::TransformToHost();
template void Array<float2>::TransformToDevice();

//###################################################

template void Array<int>::CopyToDevice();
template void Array<int>::CopyToHost();
template void Array<int>::TransformToHost();
template void Array<int>::TransformToDevice();

//###################################################

template void Array<int3>::CopyToDevice();
template void Array<int3>::CopyToHost();
template void Array<int3>::TransformToHost();
template void Array<int3>::TransformToDevice();

//###################################################

template void Array<int2>::CopyToDevice();
template void Array<int2>::CopyToHost();
template void Array<int2>::TransformToHost();
template void Array<int2>::TransformToDevice();

//###################################################

template void Array<unsigned int>::CopyToDevice();
template void Array<unsigned int>::CopyToHost();
template void Array<unsigned int>::TransformToHost();
template void Array<unsigned int>::TransformToDevice();

}
