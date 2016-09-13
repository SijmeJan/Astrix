/*! \file cudaLow.h
\brief Header file for CUDA error handling.
*/
#ifndef ASTRIX_CUDA_LOW_H
#define ASTRIX_CUDA_LOW_H

//! Macro handling device errors through gpuAssert
/*! Every CUDA function should be called using this macro, so that upon error the program exists indicating where the error occurred.*/
#define gpuErrchk(ans) { \
    gpuAssert((ans), const_cast<char *>(__FILE__), __LINE__); }

namespace astrix{

//! Handle device errors
/*! Handle device errors by simply printing error and exiting. Called by macro gpuErrchk. 
  \param code CUDA error code
  \param *file Pointer to source file name where error occurred
  \param line Line number where error occurred
*/  
void gpuAssert(cudaError_t code, char *file, int line);

}

#endif
