/*! \file unique_triangle.h
\brief Header for finding unique triangles.

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef ASTRIX_UNIQUE_TRIANGLE
#define ASTRIX_UNIQUE_TRIANGLE

namespace astrix {

void FindUniqueTriangleAffected(Array<int> *triangleAffected,
                                Array<int> *triangleAffectedIndex,
                                Array<int> *uniqueFlag,
                                int maxIndex, int cudaFlag);

}

#endif
