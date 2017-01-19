/*! \file cudaLow.h
\brief Header file for CUDA error handling.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
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
