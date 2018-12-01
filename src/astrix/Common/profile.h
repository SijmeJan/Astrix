/*! \file profile.h
\brief Header file for writing profile info to file.

*/ /* \section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_PROFILE_H
#define ASTRIX_PROFILE_H

namespace astrix {

//! Write kernel profiling info to file
/*! Write simple statistics to text file
  \param *fileName Output file name
  \param X Number of elements processed
  \param T Elapsed time
*/
void WriteProfileFile(const char *fileName, int X, float T, int cudaFlag);

}  // namespace astrix

#endif
