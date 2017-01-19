// -*-c++-*-
/*! \file unique_triangle.cu
\brief Header for finding unique triangles.*/
#include <iostream>

#include "../../Array/array.h"
#include "../../Common/cudaLow.h"

namespace astrix {

//############################################################################
/*! \brief Kernel finding unique set of affected triangles

This function selects unique entries in array \a triangleAffected. If \a triangleAffected[i-1] == \a trangleAffected[i] then \a uniqueFlag[triangleAffectedIndex[i]] is set to zero.

 * @param maxIndex number of entries in \a triangleAffected to consider
 * @param *pTriangleAffected list of affected triangles.
 * @param *pTriangleAffectedIndex list of indices of points to be inserted.
 * @param *pUniqueFlag pointer to output array: entries will be unity if unique, zero otherwise.*/
//############################################################################

__global__ void
devFindUniqueTriangleAffected(int maxIndex,
                              int *pTriangleAffected,
                              int *pTriangleAffectedIndex,
                              int *pUniqueFlag)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x + 1;

  while (i < maxIndex) {
    if (pTriangleAffected[i-1] == pTriangleAffected[i] &&
        pTriangleAffected[i] != -1) {
      int j = pTriangleAffectedIndex[i];
      pUniqueFlag[j] = 0;
    }

    i += gridDim.x*blockDim.x;
  }
}

//############################################################################
/*! \brief Find unique set of affected triangles

This function selects unique entries in array \a triangleAffected. If \a triangleAffected[i-1] == \a trangleAffected[i] then \a uniqueFlag[triangleAffectedIndex[i]] is set to zero.

 * @param *triangleAffected list of affected triangles.
 * @param *triangleAffectedIndex list of indices of points to be inserted.
 * @param *uniqueFlag pointer to output array: entries will be unity if unique, zero otherwise.
 * @param maxIndex number of entries in \a triangleAffected to consider
 * @param cudaFlag flag indicating whether to use GPU (1) or CPU (0).*/
//############################################################################

void FindUniqueTriangleAffected(Array<int> *triangleAffected,
                                Array<int> *triangleAffectedIndex,
                                Array<int> *uniqueFlag,
                                int maxIndex, int cudaFlag)
{
  int *pTriangleAffected = triangleAffected->GetPointer();
  int *pTriangleAffectedIndex = triangleAffectedIndex->GetPointer();
  int *pUniqueFlag = uniqueFlag->GetPointer();

  uniqueFlag->SetToValue(1);

  if (cudaFlag == 1) {
    int nBlocks = 128;
    int nThreads = 128;

    // Base nThreads and nBlocks on maximum occupancy
    cudaOccupancyMaxPotentialBlockSize
      (&nBlocks, &nThreads, devFindUniqueTriangleAffected, (size_t) 0, 0);

    devFindUniqueTriangleAffected<<<nBlocks, nThreads>>>
      (maxIndex,
       pTriangleAffected,
       pTriangleAffectedIndex,
       pUniqueFlag);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
  } else {
    for (int i = 1; i < maxIndex; i++) {
      if (pTriangleAffected[i - 1] == pTriangleAffected[i] &&
          pTriangleAffected[i] != -1) {
        int j = pTriangleAffectedIndex[i];
        pUniqueFlag[j] = 0;
      }
    }
  }
}

}
