// -*-c++-*-
/*! \file reduce.cu
\brief Reduction functions
*/
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//######################################################
// Find minimum
//######################################################

template <class T>
T Array<T>::Minimum()
{
  T result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    
    typename thrust::device_vector<T>::iterator iter =
      thrust::min_element(dev_ptr, dev_ptr + size);
    result = *iter;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter =
      thrust::min_element(hostVec, hostVec + size);
    result = *iter;
  }

  return result;
}

//######################################################
// Find minimum of dimension n
//######################################################

template <class T>
T Array<T>::Minimum(int N)
{
  T result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(&deviceVec[N*realSize]);
    
    typename thrust::device_vector<T>::iterator iter =
      thrust::min_element(dev_ptr, dev_ptr + size);
    result = *iter;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter =
      thrust::min_element(hostVec + N*realSize, hostVec + N*realSize + size);
    result = *iter;
  }

  return result;
}

//######################################################
// Find maximum
//######################################################

template <class T>
T Array<T>::Maximum()
{
  T result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);

    typename thrust::device_vector<T>::iterator iter = 
      thrust::max_element(dev_ptr, dev_ptr + size);
    result = *iter;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter = 
      thrust::max_element(hostVec, hostVec + size);
    result = *iter;
  }

  return result;
}

//######################################################
// Find maximum of dimension n
//######################################################

template <class T>
T Array<T>::Maximum(int N)
{
  T result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(&deviceVec[N*realSize]);

    typename thrust::device_vector<T>::iterator iter = 
      thrust::max_element(dev_ptr, dev_ptr + size);
    result = *iter;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter = 
      thrust::max_element(hostVec + N*realSize, hostVec + N*realSize + size);
    result = *iter;
  }

  return result;
}

//######################################################
// Find maximum of dimension n
//######################################################

template <class T>
struct compare_x
{
  __host__ __device__
  bool operator()(T lhs, T rhs)
  {
    return lhs.x < rhs.x;
  }
};

template <class T>
struct compare_y
{
  __host__ __device__
  bool operator()(T lhs, T rhs)
  {
    return lhs.y < rhs.y;
  }
};

template <class T>
template <class S>
S Array<T>::MaximumComb(int N)
{
  S result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(&deviceVec[0]);

    typename thrust::device_vector<T>::iterator iter;

    if (N == 0)
      iter = thrust::max_element(dev_ptr, dev_ptr + size, compare_x<T>());
    if (N == 1)
      iter = thrust::max_element(dev_ptr, dev_ptr + size, compare_y<T>());
    T res = *iter;
    if (N == 0) result = res.x;
    if (N == 1) result = res.y;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter;

    if (N == 0)
      iter = thrust::max_element(hostVec, hostVec + size, compare_x<T>());
    if (N == 1)
      iter = thrust::max_element(hostVec, hostVec + size, compare_y<T>());
    if (N == 0) result = (*iter).x;
    if (N == 1) result = (*iter).y;
  }

  return result;
}

template <class T>
template <class S>
S Array<T>::MinimumComb(int N)
{
  S result = 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(&deviceVec[0]);

    typename thrust::device_vector<T>::iterator iter;

    if (N == 0)
      iter = thrust::min_element(dev_ptr, dev_ptr + size, compare_x<T>());
    if (N == 1)
      iter = thrust::min_element(dev_ptr, dev_ptr + size, compare_y<T>());

    T res = *iter;
    if (N == 0) result = res.x;
    if (N == 1) result = res.y;
  }
  if (cudaFlag == 0) {
    typename thrust::host_vector<T>::iterator iter;

    if (N == 0)
      iter = thrust::min_element(hostVec, hostVec + size, compare_x<T>());
    if (N == 1)
      iter = thrust::min_element(hostVec, hostVec + size, compare_y<T>());
    if (N == 0) result = (*iter).x;
    if (N == 1) result = (*iter).y;
  }

  return result;
}

//######################################################
// Find total sum
//######################################################

template <class T>
T Array<T>::Sum()
{
  T result = (T) 0;

  if (cudaFlag == 1) {
    thrust::device_ptr<T> dev_ptr(deviceVec);
    
    result =
      thrust::reduce(dev_ptr, dev_ptr + size, (T) 0, thrust::plus<T>());
  }
  if (cudaFlag == 0) {
    result =
      thrust::reduce(hostVec, hostVec + size, (T) 0, thrust::plus<T>());
  }

  return result;
}

//###################################################
// Instantiate
//###################################################

template float Array<float>::Minimum();
template float Array<float>::Minimum(int N);
template float Array<float>::Maximum();
template float Array<float>::Maximum(int N);
template float Array<float>::Sum();

//###################################################

template double Array<double>::Minimum();
template double Array<double>::Minimum(int N);
template double Array<double>::Maximum();
template double Array<double>::Maximum(int N);
template double Array<double>::Sum();

//###################################################

template int Array<int>::Minimum();
template int Array<int>::Maximum();
template int Array<int>::Sum();

//###################################################

template unsigned int Array<unsigned int>::Sum();

template float Array<float2>::MinimumComb(int N);
template float Array<float2>::MaximumComb(int N);

template double Array<double2>::MinimumComb(int N);
template double Array<double2>::MaximumComb(int N);
}
