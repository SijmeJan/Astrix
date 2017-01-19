// -*-c++-*-
/*! \file shuffle.cu
\brief Functions for shuffling array

\section LICENSE
Copyright (c) 2017 Sijme-Jan Paardekooper

This file is part of Astrix.

Astrix is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Astrix is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Astrix.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "./array.h"
#include "../Common/cudaLow.h"

namespace astrix {

//###################################################
//###################################################

template <class T>
void Array<T>::Shuffle()
{
  Array<unsigned int> *randomNumbers = new Array<unsigned int>(1, 0, size);
  randomNumbers->SetToRandom();
  if (cudaFlag == 1) randomNumbers->TransformToDevice();

  Array<unsigned int> *randomPermutation =
    new Array<unsigned int>(1, cudaFlag, size);
  randomPermutation->SetToSeries();

  randomNumbers->SortByKey(randomPermutation);

  Reindex(randomPermutation->GetPointer());

  delete randomNumbers;
  delete randomPermutation;
}

//###################################################
// Instantiate
//###################################################

template void Array<int>::Shuffle();
template void Array<unsigned int>::Shuffle();

}
