/*! \file array.h
\brief Header file for Array class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_ARRAY_H
#define ASTRIX_ARRAY_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace astrix {

//! Basic vector-like class used in Astrix
/*! An Array object can be thought of as a vector that can either live on the
host or on the device.*/
template <class T> class Array
{
 public:
  //! Basic constructor for 1D Array on host
  /*! Construct 1D array on host of size \a dynArrayStep.*/
  Array();
  //! Constructor for multidimensional Array, possibly on device
  /*! Construct Array of dimensions \a _nDims, on device if \a _cudaFlag = 1, of size \a dynArrayStep.
    \param _nDims Number of dimensions
    \param _cudaFlag Create Array on host (=0) or on device (=1)
  */
  Array(unsigned int _nDims, int _cudaFlag);
  //! Constructor for multidimensional Array of specified size, possibly on device
  /*! Construct Array of dimensions \a _nDims, on device if \a _cudaFlag = 1, of size \a _size.
    \param _nDims Number of dimensions
    \param _cudaFlag Create Array on host (=0) or on device (=1)
    \param _size Size of every dimension
  */
  Array(unsigned int _nDims, int _cudaFlag, unsigned int _size);
  //! Constructor for multidimensional Array of specified size and dynArrayStep, possibly on device
  /*! Construct Array of dimensions \a _nDims, on device if \a _cudaFlag = 1, of size \a _size, with dynamical size increase step \a _dynArrayStep.
    \param _nDims Number of dimensions
    \param _cudaFlag Create Array on host (=0) or on device (=1)
    \param _size Size of every dimension
    \param _dynArrayStep Increase physical size of array in these steps*/
  Array(unsigned int _nDims, int _cudaFlag,
        unsigned int _size, int _dynArrayStep);
  //! Construct Array from ASCII file
  /*! Construct Array from ASCII file. Dimensions are determined from file.
    \param inputFile Name of inputfile*/
  Array(std::string inputFile);

  //! Destructor, releases allocated memory
  /*! Destroy Array object, releasing both host and device memory*/
  ~Array();

  //! Total amount of memory (bytes) allocated on host in all Array's
  static int64_t memAllocatedHost;
  //! Total amount of memory (bytes) allocated on device in all Array's
  static int64_t memAllocatedDevice;

  //! Transform from host vector to device vector
  void TransformToDevice();
  //! Transform from device vector to host vector
  void TransformToHost();

  //! Return whether data currently resides on host or device
  int GetCudaFlag() const { return cudaFlag; }
  //! Return size of array
  unsigned int GetSize() const;
  //! Return realSize of array
  unsigned int GetRealSize() const;
  //! Return number of dimensions
  unsigned int GetDimension() const;
  //! Set size of Array on host
  void SetSizeHost(unsigned int _size);
  //! Set size of Array on device
  void SetSizeDevice(unsigned int _size);
  //! Set size of Array, either on host or device depending on \a cudaFlag
  void SetSize(unsigned int _size);

  //! Set all Array entries to \a value
  void SetToValue(T value);
  //! Set Array entries from \a startIndex to \a endIndex to \a value
  void SetToValue(T value, unsigned int startIndex, unsigned int endIndex);
  //! Set \a a[0] = 0, \a a[1] = 1, etc.
  void SetToSeries();
  //! Set \a a[\a i] = \a i for all \a i from \a startIndex to \a endIndex
  void SetToSeries(unsigned int startIndex, unsigned int endIndex);

  //! Set all entries equal to those in Array \a *B
  void SetEqual(const Array *B);
  //! Set all entries of dimension N of Array equal to dimension M of Array B
  void SetEqual(const Array *B, unsigned int N, unsigned int M);
  template <class S>
    void SetEqualComb(const Array<S> *B, unsigned int N, unsigned int M);
  //! Set all entries starting from \a startPosition equal to those of Array \a *B, i.e. \a a[\a startPosition] = \a b[0] etc.
  void SetEqual(const Array *B, int startPosition);

  //! Reindex array: a[i] = a[reindex[i]]
  void Reindex(unsigned int *reindex);
  //! Reindex array: a[i] = a[reindex[i]] for the first \a N elements
  void Reindex(unsigned int *reindex, unsigned int N);
  //! Inverse reindex array: a[i] = reindex[a[i]]
  /*! Set a[i] = reindex[a[i]]. If a[i] = -1, leave it at -1. If a[i] >= maxValue, subtract maxValue n times until a[i] < maxValue, and set a[i] = a[reindex[a[i]-n*maxValue]] + n*maxValue*/
  void InverseReindex(unsigned int *reindex, int maxValue, bool ignoreValue);
  void InverseReindex(int *reindex);

  //! Compact; keep only entries where keepFlag == 1
  void Compact(int nKeep, Array<int> *keepFlag,
               Array<int> *keepFlagScan);

  //! Copy data from host to device
  void CopyToDevice();
  //! Copy data from device to host
  void CopyToHost();

  //! Set a[position] = value
  void SetSingleValue(T value, int position);
  //! Set a[position] = value for dimension \a N
  void SetSingleValue(T value, int position, unsigned int N);
  //! Real a[position] into *value
  void GetSingleValue(T *value, int position);
  //! Read a[position] for dimension \a N into *value
  void GetSingleValue(T *value, int position, unsigned int N);

  //! Add value to all entries from startIndex to endIndex
  void AddValue(T value, unsigned int startIndex, unsigned int endIndex);

  //! Create float2/double2 array from 2D float/double array
  template <class S>
    void MakeIntrinsic2D(Array<S> *result);

  //! Sort array, together with \a arrayB
  void Sort(Array<T> *arrayB);
  //! Create index array for sorting
  template<class S>
    void SortByKey(Array<S> *indexArray);
  //! Create index array for sorting dimension N
  template<class S>
    void SortByKey(Array<S> *indexArray, unsigned int N);

  //! Perform exclusive scan
  T ExclusiveScan(Array<T> *result);
  //! Perform exclusive scan on dimension \a N
  T ExclusiveScan(Array<T> *result, unsigned int N);

  //! Set out[i] = in[map[i]]
  void Gather(Array<T> *in, Array<int> *map, int maxIndex);
  //! If map[i] != value then out[i] = in[map[i]]
  void GatherIf(Array<T> *in, Array<int> *map, int value, int maxIndex);
  //! Set out[map[i]] = in[i]
  void Scatter(Array<T> *in, Array<int> *map, int maxIndex);
  //! Set out[map[i]] = i
  template<class S>
    void ScatterSeries(Array<S> *map, unsigned int maxIndex);

  //! Set to random values using rand()
  void SetToRandom();

  //! Set a[i] = a[i] - b[i]
  void SetToDiff(Array<T> *A, Array<T> *B);

  //! Return minimum of array
  T Minimum();
  //! Return minimum of dimension N of array
  T Minimum(int N);
  //! Return maximum of array
  T Maximum();
  //! Return maximum of dimension N of array
  T Maximum(int N);
  template <class S>
    S MinimumComb(int N);
  template <class S>
    S MaximumComb(int N);

  //! Return sum of elements
  T Sum();

  //! Swap values 0 and 1
  void Invert();

  //! Join with Array \a A
  void Concat(Array<T> *A);

  //! Remove every entry start+i*step, compact array
  int RemoveEvery(int start, int step);
  //! Remove every entry start+i*step, compact array and inverse reindex A
  template<class S>
    int RemoveEvery(int start, int step, Array<S> *A);
  //! Remove entries equal to \a value from Array
  int RemoveValue(T value);
  int RemoveValue(T value, int maxIndex);

  template<class S>
    int SelectLargerThan(T value, Array<S> *A);
  template<class S>
    int SelectWhereDifferent(Array<T> *A, Array<S> *B);

  //! At a unique entry of \a A, \a i, (ignoring \a ignoreValue) set \a hostVec[B[i]] = \a value.
  void ScatterUnique(Array<int> *A, Array<int> *B,
                     int maxIndex, int ignoreValue, T value);

  //! Shuffle array (non-random!)
  void Shuffle();

  T InnerProduct(Array<T> *A);
  void LinComb(T a1, Array<T> *A1);
  void LinComb(T a1, Array<T> *A1,
               T a2, Array<T> *A2);
  void LinComb(T a1, Array<T> *A1,
               T a2, Array<T> *A2,
               T a3, Array<T> *A3);

  //! Return pointer to host memory
  T* GetHostPointer() const { return hostVec; }
  //! Return pointer to host memory for dimension \a _dim
  T* GetHostPointer(unsigned int _dim) const
  { return &(hostVec[_dim*realSize]); }
  //! Return pointer to device memory
  T* GetDevicePointer() const { return deviceVec; }
  //! Return pointer to device memory for dimension \a _dim
  T* GetDevicePointer(unsigned int _dim) const
  { return &(deviceVec[_dim*realSize]); }
  //! Return pointer to either host or device memory, depending on cudaFlag
  T* GetPointer() const {
    if (cudaFlag == 0) {
      return hostVec;
    } else {
      return deviceVec;
    }
  }
  //! Return pointer to either host or device memory for dimension \a _dim
  T* GetPointer(unsigned int _dim) const {
    if (cudaFlag == 0) {
      return &(hostVec[_dim*realSize]);
    } else {
      return &(deviceVec[_dim*realSize]);
    }
  }

  int dynArrayStep;
 private:
  //! Size of array
  unsigned int size;
  //! Physical size of array (larger than \a size because of \a dynArrayStep)
  unsigned int realSize;
  //! Number of dimensions of array
  unsigned int nDims;
  //! Flag whether to use device memory or host memory
  int cudaFlag;

  //! Pointer to host memory
  T *hostVec;
  //! Pointer to device memory
  T *deviceVec;
};

template <typename T>
int64_t Array<T>::memAllocatedHost = 0;
template <typename T>
int64_t Array<T>::memAllocatedDevice = 0;

}  // namespace astrix
#endif
