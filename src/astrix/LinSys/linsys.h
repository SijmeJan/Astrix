/*! \file linsys.h
\brief Header file for LinSys class

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.*/
#ifndef ASTRIX_LINSYS_H
#define ASTRIX_LINSYS_H

namespace astrix {

class Mesh;
template <class T> class Array;
class Device;

class LinSys
{
 public:
  //! Constructor for LinSys object.
  LinSys(int _verboseLevel, int _debugLevel, Device *_device);
  //! Destructor, releases all dynamically allocated memory
  ~LinSys();

  void BiCGStab(Array<real> *x, Array<real> *b);

 private:
  Device *device;

  //! Flag whether to use CUDA
  int cudaFlag;

  //! How much to output to screen
  int verboseLevel;
  //! Level of debugging
  int debugLevel;

  //! Mesh on which to do simulation
  Mesh *mesh;

  void MultiplyByMatrix(Array<real> *vIn, Array<real> *vOut);
};

}  // namespace astrix

#endif  // ASTRIX_LINSYS_H
