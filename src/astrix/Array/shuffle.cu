// -*-c++-*-
/*! \file shuffle.cu
\brief Functions for shuffling array
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
