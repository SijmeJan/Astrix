/*! \file vtk.h
\brief Header file for VTK class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_VTK_H
#define ASTRIX_VTK_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace astrix {

class Mesh;

//! VTK: object to write legacy VTK output
class VTK
{
 public:
  //! Constructor for VTK object.
  VTK();
  //! Destructor for VTK object
  ~VTK();

  void Write(const char *fileName, Mesh *mesh, realNeq *state);
 private:
  std::ofstream outFile;

  int swapEndian;

  void write_unstructured_mesh(const char *fileName, int nPoints, float *pts,
                               int nCells, int *conn,
                               int nVars, int *varDim, int *centering,
                               const char * const *varNames, float **vars);

  void write_variables(int nVars, int *varDim, int *centering,
                       const char * const * varName, float **vars,
                       int nPoints, int nCells);

  void ForceBigEndian(unsigned char *bytes);

  template <class T>
    void writeSingle(T val);
};

}  // namespace astrix

#endif  // ASTRIX_VTK_H
